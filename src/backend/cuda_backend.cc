#include "backend/sycl_backend.h"

#include <cuda_runtime.h>

#include "../tracy.h"
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

void nd_copy_async(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range,
    const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size, const cudaStream_t stream) {
	assert(all_true(offset_in_source + copy_range <= source_range));
	assert(all_true(offset_in_dest + copy_range <= dest_range));

	if(copy_range.size() == 0) return;

	// TODO copied from nd_copy_host - this works but is not optimal, it will still do a 2D copy on a [1,1,1] range if there is a dim1 offset
	int linear_dim = 0;
	for(int d = 1; d < 3; ++d) {
		if(source_range[d] != copy_range[d] || dest_range[d] != copy_range[d]) { linear_dim = d; }
	}

	const auto first_source_elem = static_cast<const std::byte*>(source_base) + get_linear_index(source_range, offset_in_source) * elem_size;
	const auto first_dest_elem = static_cast<std::byte*>(dest_base) + get_linear_index(dest_range, offset_in_dest) * elem_size;

	switch(linear_dim) {
	case 0: {
		const auto copy_bytes = (copy_range[0] * copy_range[1] * copy_range[2]) * elem_size;
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_1d", ForestGreen, "cudaMemcpyAsync")
		CELERITY_CUDA_CHECK(cudaMemcpyAsync, first_dest_elem, first_source_elem, copy_bytes, cudaMemcpyDefault, stream);
		break;
	}
	case 1: {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_2d", ForestGreen, "cudaMemcpy2DAsync")
		CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, first_dest_elem, dest_range[1] * dest_range[2] * elem_size, first_source_elem,
		    source_range[1] * source_range[2] * elem_size, copy_range[1] * copy_range[2] * elem_size, copy_range[0], cudaMemcpyDefault, stream);
		break;
	}
	case 2: {
		cudaMemcpy3DParms parms = {};
		parms.srcPos = make_cudaPos(offset_in_source[2] * elem_size, offset_in_source[1], offset_in_source[0]);
		parms.srcPtr = make_cudaPitchedPtr(
		    const_cast<void*>(source_base), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
		parms.dstPos = make_cudaPos(offset_in_dest[2] * elem_size, offset_in_dest[1], offset_in_dest[0]);
		parms.dstPtr = make_cudaPitchedPtr(dest_base, dest_range[2] * elem_size, dest_range[2], dest_range[1]);
		parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
		parms.kind = cudaMemcpyDefault;
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::memcpy_3d", ForestGreen, "cudaMemcpy3DAsync")
		CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, stream);
		break;
	}
	default: assert(!"unreachable");
	}
}

void copy_region_async(cudaStream_t stream, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const region<3>& copy_region, const size_t elem_size) {
	for(const auto& copy_box : copy_region.get_boxes()) {
		assert(source_box.covers(copy_box));
		assert(dest_box.covers(copy_box));
		nd_copy_async(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
		    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size, stream);
	}
}


struct cuda_event_deleter {
	void operator()(const cudaEvent_t evt) const { CELERITY_CUDA_CHECK(cudaEventDestroy, evt); }
};

using unique_cuda_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_event_deleter>;

unique_cuda_event make_cuda_event() {
	cudaEvent_t event;
	CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
	return unique_cuda_event(event);
}

class event_impl final : public async_event_impl {
  public:
	event_impl(unique_cuda_event evt) : m_evt(std::move(evt)) {}

	bool is_complete() override {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::query_event", ForestGreen, "cudaEventQuery")
		switch(const auto result = cudaEventQuery(m_evt.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: utils::panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

  private:
	unique_cuda_event m_evt;
};

static async_event record_event(const cudaStream_t stream) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cuda::record_event", ForestGreen, "cudaEventRecord")
	auto event = make_cuda_event();
	CELERITY_CUDA_CHECK(cudaEventRecord, event.get(), stream);
	return make_async_event<event_impl>(std::move(event));
}

async_event copy_region(sycl::queue& queue, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const region<3>& copy_region, const size_t elem_size) {
#if CELERITY_WORKAROUND(HIPSYCL)
	auto event = queue.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle handle) {
		const auto stream = handle.get_native_queue<sycl::backend::cuda>();
		cuda_backend_detail::copy_region_async(stream, source_base, dest_base, source_box, dest_box, copy_region, elem_size);
	});
	sycl_backend_detail::flush_queue(queue);
	return make_async_event<sycl_event>(std::move(event), false /* enable_profiling */);
#elif CELERITY_WORKAROUND(DPCPP)
	const auto stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
	cuda_backend_detail::copy_region_async(stream, source_base, dest_base, source_box, dest_box, copy_region, elem_size);
	return record_event(stream);
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
	for(size_t i = 0; i < devices.size(); ++i) {
		for(size_t j = i + 1; j < devices.size(); ++j) {
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
			}
		}
	}
#endif
}

async_event sycl_cuda_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) //
{
	return cuda_backend_detail::copy_region(get_device_queue(device, device_lane), source_base, dest_base, source_box, dest_box, copy_region, elem_size);
}

} // namespace celerity::detail
