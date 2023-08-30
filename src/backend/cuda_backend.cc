#include "backend/cuda_backend.h"

#include <cuda_runtime.h>

#include "log.h"
#include "ranges.h"

#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                                                                                            \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		CELERITY_CRITICAL(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                \
		abort();                                                                                                                                               \
	}

namespace celerity::detail::backend_detail {

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<0>& /* source_range */,
    const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */, const range<0>& /* copy_range */) {
	(void)queue;
	const auto ret = cudaMemcpy(target_base_ptr, source_base_ptr, elem_size, cudaMemcpyDefault);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	cudaStreamSynchronize(0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<1>& source_range,
    const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range) {
	(void)queue;
	const size_t line_size = elem_size * copy_range[0];
	CELERITY_CUDA_CHECK(cudaMemcpy, static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<2>& source_range,
    const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range) {
	(void)queue;
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	CELERITY_CUDA_CHECK(cudaMemcpy2D, static_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
	    static_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size, copy_range[0],
	    cudaMemcpyDefault);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<3>& source_range,
    const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	cudaMemcpy3DParms parms = {};
	parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
	parms.srcPtr = make_cudaPitchedPtr(
	    const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
	parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
	parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
	parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
	parms.kind = cudaMemcpyDefault;
	CELERITY_CUDA_CHECK(cudaMemcpy3D, &parms);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const ize_t elem_size,
    const range<0>& /* source_range */, const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */,
    const range<0>& /* copy_range */) {
	CELERITY_CUDA_CHECK(cudaMemcpyAsync, target_base_ptr, source_base_ptr, elem_size, cudaMemcpyDefault, stream);
}

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const ize_t elem_size,
    const range<1>& source_range, const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range) {
	const size_t line_size = elem_size * copy_range[0];
	CELERITY_CUDA_CHECK(cudaMemcpyAsync, static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault, stream);
}

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const size_t elem_size,
    const range<2>& source_range, const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range) {
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, static_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
	    static_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size, copy_range[0],
	    cudaMemcpyDefault, stream);
}

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const size_t elem_size,
    const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	cudaMemcpy3DParms parms = {};
	parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
	parms.srcPtr = make_cudaPitchedPtr(
	    const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
	parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
	parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
	parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
	parms.kind = cudaMemcpyDefault;
	CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, stream);
}

struct cuda_stream_deleter {
	void operator()(const cudaStream_t stream) { CELERITY_CUDA_CHECK(cudaStreamDestroy, stream); }
};

using unique_cuda_stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, cuda_stream_deleter>;

struct cuda_event_deleter {
	void operator()(const cudaEvent_t evt) { CELERITY_CUDA_CHECK(cudaEventDestroy, evt); }
};

using unique_cuda_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_event_deleter>;

} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

class cuda_event : public queue::event {
  public:
	cuda_event(backend_detail::uniqe_cuda_event evt) : m_evt(std::move(evt)) {}

	bool is_complete() const override {
		switch(cudaEventQuery(m_evt.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: CELERITY_CRITICAL("cudaEventQuery: {}", cudaGetErrorString(cuda_check_result)); abort();
		}
	}

  private:
	backend_detail::unique_cuda_event m_evt;
};

struct cuda_queue::impl {
	enum copy_direction : size_t { from_host, to_host, to_peer, count };
	using device_streams = std::array<unique_cuda_stream, copy_direction::count>;
	std::unordered_map<device_id, device_streams> streams;
};

cuda_queue::cuda_queue() : m_impl(std::make_unique<impl>()) {}

void cuda_queue::add_device(device_id device, sycl::queue& queue) {
	assert(m_impl->streams.find(device_id) == m_impl->streams.end());
	const auto cuda_device_id = 0; // queue.get???();
	int cuda_device_id_before;
	CELERITY_CUDA_CHECK(cudaGetDevice, &cuda_device_id_before);
	CELERITY_CUDA_CHECK(cudaSetDevice, cuda_device_id);
	impl::device_streams device_streams;
	for(size_t i = 0; i < impl::copy_direction::count; ++i) {
		cudaStream_t stream;
		CELERITY_CUDA_CHECK(cudaStreamCreate, &stream);
		device_streams[i] = backend_detail::unique_cuda_stream(stream);
	}
	m_impl.streams.emplace(device, std::move(device_streams));
	CELERITY_CUDA_CHECK(cudaSetDevice, cuda_device_id_before);
}

std::unique_ptr<event> cuda_queue::memcpy_strided_device(const int dims, const memory_id source, const memory_id dest, const void* const source_base_ptr,
    void* const target_base_ptr, const size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range,
    const id<3>& target_offset, const range<3>& copy_range) {
	assert(source != host_memory_id || target != host_memory_id);
	const auto stream = source == host_memory_id   ? m_impl->streams.at(to_device_id(target))[from_host].get()
	                    : target == host_memory_id ? m_impl->streams.at(to_device_id(source))[to_host].get()
	                                               : m_impl->streams.at(to_device_id(source))[to_peer].get();

	cudaEvent_t raw_evt;
	CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &raw_evt, cudaEventDisableTiming);
	auto evt = backend_detail::unique_cuda_event(raw_evt);

	const auto dispatch_memcpy = [&](const auto dims) {
		backend_detail::memcpy_strided_device_generic(evt.get(), source_base_ptr, target_base_ptr, elem_size, range_cast<dims.value>(source_range),
		    id_cast<dims.value>(source_offset), range_cast<dims.value>(target_range), id_cast<dims.value>(target_offset), range_cast<dims.value>(copy_range));
	};
	switch(dims) {
	case 0: dispatch_memcpy(std::integral_constant<int, 0>()); break;
	case 1: dispatch_memcpy(std::integral_constant<int, 1>()); break;
	case 2: dispatch_memcpy(std::integral_constant<int, 2>()); break;
	case 3: dispatch_memcpy(std::integral_constant<int, 3>()); break;
	default: abort();
	}

	CELERITY_CUDA_CHECK(cudaEventRecord, raw_evt, stream);
	return std::make_unique<cuda_event>(std::move(evt));
}

} // namespace celerity::detail::backend
