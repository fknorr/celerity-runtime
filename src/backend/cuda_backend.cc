#include "backend/sycl_backend.h"

#include <cuda_runtime.h>

#include "../tracy.h"
#include "log.h"
#include "nd_memory.h"
#include "ranges.h"
#include "system_info.h"
#include "utils.h"
#include "workaround.h"


#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                                                                                            \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		utils::panic(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                     \
	}

namespace celerity::detail::cuda_backend_detail {

void nd_copy_device(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range,
    const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size, const cudaStream_t stream) //
{
	const auto layout = layout_nd_copy(source_range, dest_range, offset_in_source, offset_in_dest, copy_range);
	if(layout.contiguous_range == 0) return;

	if(layout.num_complex_strides == 0) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_1d", ForestGreen, "cudaMemcpyAsync")
		CELERITY_CUDA_CHECK(cudaMemcpyAsync, static_cast<std::byte*>(dest_base) + layout.linear_offset_in_dest * elem_size,
		    static_cast<const std::byte*>(source_base) + layout.linear_offset_in_source * elem_size, layout.contiguous_range * elem_size, cudaMemcpyDefault,
		    stream);
	} else if(layout.num_complex_strides == 1) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_2d", ForestGreen, "cudaMemcpy2DAsync")
		CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, static_cast<std::byte*>(dest_base) + layout.linear_offset_in_dest * elem_size,
		    layout.strides[0].dest_stride * elem_size, static_cast<const std::byte*>(source_base) + layout.linear_offset_in_source * elem_size,
		    layout.strides[0].source_stride * elem_size, layout.contiguous_range * elem_size, layout.strides[0].count, cudaMemcpyDefault, stream);
	} else {
		assert(layout.num_complex_strides == 2);
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_3d", ForestGreen, "cudaMemcpy3DAsync")
		// Arriving in the 3D case means no dimensionality reduction was possible, and cudaMemcpy3D is more closely aligned to the parameters to nd_copy_device
		// than to nd_copy_layout, so we don't compute cudaMemcpy3DParms from `layout`.
		cudaMemcpy3DParms parms = {};
		parms.srcPos = make_cudaPos(offset_in_source[2] * elem_size, offset_in_source[1], offset_in_source[0]);
		parms.srcPtr = make_cudaPitchedPtr(
		    const_cast<void*>(source_base), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
		parms.dstPos = make_cudaPos(offset_in_dest[2] * elem_size, offset_in_dest[1], offset_in_dest[0]);
		parms.dstPtr = make_cudaPitchedPtr(dest_base, dest_range[2] * elem_size, dest_range[2], dest_range[1]);
		parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
		parms.kind = cudaMemcpyDefault;
		CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, stream);
	}
}

void nd_copy_device(cudaStream_t stream, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const box<3>& copy_box, const size_t elem_size) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	nd_copy_device(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
	    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size, stream);
}

void nd_copy_device(cudaStream_t stream, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const region<3>& copy_region, const size_t elem_size) //
{
	for(const auto& copy_box : copy_region.get_boxes()) {
		nd_copy_device(stream, source_base, dest_base, source_box, dest_box, copy_box, elem_size);
	}
}


struct cuda_native_event_deleter {
	void operator()(const cudaEvent_t evt) const { CELERITY_CUDA_CHECK(cudaEventDestroy, evt); }
};

using unique_cuda_native_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_native_event_deleter>;

class cuda_event final : public async_event_impl {
  public:
	cuda_event(unique_cuda_native_event after) : m_after(std::move(after)) {}
	cuda_event(unique_cuda_native_event before, unique_cuda_native_event after) : m_before(std::move(before)), m_after(std::move(after)) {}

	bool is_complete() override {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::query_event", ForestGreen, "cudaEventQuery")
		switch(const auto result = cudaEventQuery(m_after.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: utils::panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

	std::optional<std::chrono::nanoseconds> get_native_execution_time() override {
		assert(is_complete());
		if(m_before == nullptr) return std::nullopt;
		float ms = NAN;
		CELERITY_CUDA_CHECK(cudaEventElapsedTime, &ms, m_before.get(), m_after.get());
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float, std::milli>(ms));
	}

  private:
	unique_cuda_native_event m_before; // not null iff profiling is enabled
	unique_cuda_native_event m_after;
};

static unique_cuda_native_event record_native_event(const cudaStream_t stream, bool enable_profiling) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::record_event", ForestGreen, "cudaEventRecord")
	cudaEvent_t event;
	CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, enable_profiling ? cudaEventDefault : cudaEventDisableTiming);
	CELERITY_CUDA_CHECK(cudaEventRecord, event, stream);
	return unique_cuda_native_event(event);
}

async_event copy_region(sycl::queue& queue, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
#if CELERITY_WORKAROUND(HIPSYCL)
	auto event = queue.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle handle) {
		const auto stream = handle.get_native_queue<sycl::backend::cuda>();
		cuda_backend_detail::nd_copy_device(stream, source_base, dest_base, source_box, dest_box, copy_region, elem_size);
	});
	sycl_backend_detail::flush_queue(queue);
	return make_async_event<sycl_event>(std::move(event), enable_profiling);
#elif CELERITY_WORKAROUND(DPCPP)
	const auto stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
	auto before = enable_profiling ? record_native_event(stream, enable_profiling) : nullptr;
	cuda_backend_detail::copy_region_async(stream, source_base, dest_base, source_box, dest_box, copy_region, elem_size);
	auto after = record_native_event(stream, enable_profiling);
	return make_async_event<cuda_event>(std::move(before), std::move(after));
#else
#error Unavailable for this SYCL implementation
#endif
}

#if CELERITY_WORKAROUND(DPCPP)
constexpr sycl::backend sycl_cuda_backend = sycl::backend::ext_oneapi_cuda;
#else
constexpr sycl::backend sycl_cuda_backend = sycl::backend::cuda;
#endif

bool can_enable_peer_access(const int id_device, const int id_peer) {
	// RTX 30xx and 40xx GPUs do not support peer access, but Nvidia Driver < 550 incorrectly reports that it does, causing kernel panics when enabling it
	cudaDeviceProp props{};
	CELERITY_CUDA_CHECK(cudaGetDeviceProperties, &props, id_device);
	std::string_view device_name(props.name);
	if(device_name.find("RTX 30") != std::string::npos || device_name.find("RTX 40") != std::string::npos) {
		CELERITY_DEBUG("Overriding CUDA reporting of peer access capabilities for \"{}\"", device_name);
		return false;
	}

	int can_access = -1;
	CELERITY_CUDA_CHECK(cudaDeviceCanAccessPeer, &can_access, id_device, id_peer);
	assert(can_access == 0 || can_access == 1);
	return can_access != 0;
}

void enable_peer_access(const int id_device, const int id_peer) {
	int id_before = -1;
	CELERITY_CUDA_CHECK(cudaGetDevice, &id_before);
	CELERITY_CUDA_CHECK(cudaSetDevice, id_device);
	const auto enabled = cudaDeviceEnablePeerAccess(id_peer, 0);
	if(enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled) { utils::panic("cudaDeviceEnablePeerAccess: {}", cudaGetErrorString(enabled)); }
	CELERITY_CUDA_CHECK(cudaSetDevice, id_before);
}

} // namespace celerity::detail::cuda_backend_detail

namespace celerity::detail {

sycl_cuda_backend::sycl_cuda_backend(const std::vector<sycl::device>& devices, const bool enable_profiling) : sycl_backend(devices, enable_profiling) {
#if !CELERITY_DISABLE_CUDA_PEER_ACCESS
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
			const int id_i = sycl::get_native<cuda_backend_detail::sycl_cuda_backend>(devices[i]);
			const int id_j = sycl::get_native<cuda_backend_detail::sycl_cuda_backend>(devices[j]);

			// system_info mandates that copy_peers is reflexive
			if(cuda_backend_detail::can_enable_peer_access(id_i, id_j) && cuda_backend_detail::can_enable_peer_access(id_j, id_i)) {
				cuda_backend_detail::enable_peer_access(id_i, id_j);
				cuda_backend_detail::enable_peer_access(id_j, id_i);

				const memory_id mid_i = first_device_memory_id + i;
				const memory_id mid_j = first_device_memory_id + j;
				get_system_info().memories[mid_i].copy_peers.set(mid_j);
				get_system_info().memories[mid_j].copy_peers.set(mid_i);
			} else {
				CELERITY_DEBUG("CUDA does not provide peer access between D{} and D{}, device-to-device copies will be staged in host memory", i, j);
			}
		}
	}
#else
	if(devices.size() > 1) { CELERITY_DEBUG("CUDA peer access is disabled in build configuration, device-to-device copies will be staged in host memory"); }
#endif
}

async_event sycl_cuda_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) //
{
	return cuda_backend_detail::copy_region(
	    get_device_queue(device, device_lane), source_base, dest_base, source_box, dest_box, copy_region, elem_size, is_profiling_enabled());
}

} // namespace celerity::detail
