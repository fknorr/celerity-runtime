#include "backend/cuda_backend.h"

#include <cuda_runtime.h>

#include "log.h"
#include "ranges.h"
#include "types.h"
#include "utils.h"


#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                                                                                            \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		utils::panic(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                     \
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

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const size_t elem_size,
    const range<0>& /* source_range */, const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */,
    const range<0>& /* copy_range */) {
	CELERITY_CUDA_CHECK(cudaMemcpyAsync, target_base_ptr, source_base_ptr, elem_size, cudaMemcpyDefault, stream);
}

void memcpy_strided_device_cuda(const cudaStream_t stream, const void* const source_base_ptr, void* const target_base_ptr, const size_t elem_size,
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

using cuda_device_id = celerity::detail::backend::cuda_queue::cuda_device_id;

struct cuda_set_device_guard {
	explicit cuda_set_device_guard(cuda_device_id cudid) {
		CELERITY_CUDA_CHECK(cudaGetDevice, &cudid_before);
		CELERITY_CUDA_CHECK(cudaSetDevice, cudid);
	}
	cuda_set_device_guard(const cuda_set_device_guard&) = delete;
	cuda_set_device_guard(cuda_set_device_guard&&) = delete;
	cuda_set_device_guard& operator=(const cuda_set_device_guard&) = delete;
	cuda_set_device_guard& operator=(cuda_set_device_guard&&) = delete;
	~cuda_set_device_guard() { CELERITY_CUDA_CHECK(cudaSetDevice, cudid_before); }

	cuda_device_id cudid_before = -1;
};

struct cuda_stream_deleter {
	void operator()(const cudaStream_t stream) const { CELERITY_CUDA_CHECK(cudaStreamDestroy, stream); }
};

using unique_cuda_stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, cuda_stream_deleter>;

unique_cuda_stream make_cuda_stream(const cuda_device_id id) {
	cudaStream_t stream;
	CELERITY_CUDA_CHECK(cudaStreamCreateWithFlags, &stream, cudaStreamNonBlocking);
	return unique_cuda_stream(stream);
}

struct cuda_event_deleter {
	void operator()(const cudaEvent_t evt) const { CELERITY_CUDA_CHECK(cudaEventDestroy, evt); }
};

using unique_cuda_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_event_deleter>;

unique_cuda_event make_cuda_event() {
	cudaEvent_t event;
	CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
	return backend_detail::unique_cuda_event(event);
}

} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

class cuda_event final : public event {
  public:
	cuda_event(backend_detail::unique_cuda_event evt) : m_evt(std::move(evt)) {}

	static std::unique_ptr<cuda_event> record(const cudaStream_t stream) {
		auto event = backend_detail::make_cuda_event();
		CELERITY_CUDA_CHECK(cudaEventRecord, event.get(), stream);
		return std::make_unique<cuda_event>(std::move(event));
	}

	bool is_complete() const override {
		switch(const auto result = cudaEventQuery(m_evt.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: utils::panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

  private:
	backend_detail::unique_cuda_event m_evt;
};

// TODO dispatch "host" operations to thread queue and replace this type's implementation with a std::future
class cuda_host_event final : public event {
  public:
	bool is_complete() const override { return true; }
};

struct cuda_queue::impl {
	struct device {
		cuda_device_id cuda_id;
		sycl::queue sycl_queue;
	};
	struct memory {
		cuda_device_id cuda_id;
		backend_detail::unique_cuda_stream copy_from_host_stream;
		backend_detail::unique_cuda_stream copy_to_host_stream;
		std::unordered_map<memory_id, backend_detail::unique_cuda_stream> copy_from_peer_stream;
	};

	std::unordered_map<device_id, device> devices;
	std::unordered_map<memory_id, memory> memories;
};

cuda_queue::cuda_queue(const std::vector<device_config>& devices) : m_impl(std::make_unique<impl>()) {
	for(const auto& config : devices) {
		assert(m_impl->devices.count(config.device_id) == 0);
		assert(m_impl->memories.count(config.native_memory) == 0); // TODO handle devices that share memory

		const cuda_device_id cuda_id = config.sycl_device.hipSYCL_device_id().get_id();
		backend_detail::cuda_set_device_guard set_device(cuda_id);

		impl::device dev{cuda_id, sycl::queue(config.sycl_device, backend::handle_sycl_errors)};
		m_impl->devices.emplace(config.device_id, std::move(dev));

		impl::memory mem;
		mem.cuda_id = cuda_id;
		mem.copy_from_host_stream = backend_detail::make_cuda_stream(cuda_id);
		mem.copy_to_host_stream = backend_detail::make_cuda_stream(cuda_id);
		for(const auto& other_config : devices) {
			// device can be its own "peer" - buffer resizes need to copy within the device's memory
			mem.copy_from_peer_stream.emplace(other_config.native_memory, backend_detail::make_cuda_stream(cuda_id));
		}
		m_impl->memories.emplace(config.native_memory, std::move(mem));
	}
}

cuda_queue::~cuda_queue() = default;

std::pair<void*, std::unique_ptr<event>> cuda_queue::malloc(const memory_id where, const size_t size, [[maybe_unused]] const size_t alignment) {
	void* ptr;
	if(where == host_memory_id) {
		CELERITY_CUDA_CHECK(cudaMallocHost, &ptr, size, cudaHostAllocDefault);
	} else {
		const auto& mem = m_impl->memories.at(where);
		backend_detail::cuda_set_device_guard set_device(mem.cuda_id);
		// We _want_ to use cudaMallocAsync / cudaMallocFromPoolAsync for asynchronicity and stream ordering here, but according to
		// https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2 memory allocated through that API cannot be used with GPUDirect
		// RDMA (although NVIDIA plans to support this at an unspecified time in the future).
		// When we eventually switch to cudaMallocAsync, remember to call cudaMemPoolSetAccess to allow d2d copies (see the same article).
		CELERITY_CUDA_CHECK(cudaMalloc, &ptr, size);
	}
	assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
	return {ptr, std::make_unique<cuda_host_event>()};
}

std::unique_ptr<event> cuda_queue::free(const memory_id where, void* const allocation) {
	if(where == host_memory_id) {
		CELERITY_CUDA_CHECK(cudaFreeHost, allocation);
	} else {
		const auto& mem = m_impl->memories.at(where);
		backend_detail::cuda_set_device_guard set_device(mem.cuda_id);
		CELERITY_CUDA_CHECK(cudaFree, allocation);
	}
	return std::make_unique<cuda_host_event>();
}

std::unique_ptr<event> cuda_queue::memcpy_strided_device(const int dims, const memory_id source, const memory_id dest, const void* const source_base_ptr,
    void* const target_base_ptr, const size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range,
    const id<3>& target_offset, const range<3>& copy_range) {
	const impl::memory* memory = nullptr;
	cudaStream_t stream = nullptr;
	if(source == host_memory_id) {
		assert(dest != host_memory_id);
		memory = &m_impl->memories.at(dest);
		stream = memory->copy_from_host_stream.get();
	} else if(dest == host_memory_id) {
		assert(source != host_memory_id);
		memory = &m_impl->memories.at(source);
		stream = memory->copy_from_host_stream.get();
	} else {
		memory = &m_impl->memories.at(dest);
		stream = memory->copy_from_peer_stream.at(source).get();
	}

	backend_detail::cuda_set_device_guard set_device(memory->cuda_id);

	const auto dispatch_memcpy = [&](const auto dims) {
		backend_detail::memcpy_strided_device_cuda(stream, source_base_ptr, target_base_ptr, elem_size, range_cast<dims.value>(source_range),
		    id_cast<dims.value>(source_offset), range_cast<dims.value>(target_range), id_cast<dims.value>(target_offset), range_cast<dims.value>(copy_range));
	};
	switch(dims) {
	case 0: dispatch_memcpy(std::integral_constant<int, 0>()); break;
	case 1: dispatch_memcpy(std::integral_constant<int, 1>()); break;
	case 2: dispatch_memcpy(std::integral_constant<int, 2>()); break;
	case 3: dispatch_memcpy(std::integral_constant<int, 3>()); break;
	default: abort();
	}

	return cuda_event::record(stream);
}

std::unique_ptr<event> cuda_queue::launch_kernel(device_id did, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range,
    const std::vector<void*>& reduction_ptrs, bool is_reduction_initializer) {
	return launch_sycl_kernel(m_impl->devices.at(did).sycl_queue, launcher, execution_range, reduction_ptrs, is_reduction_initializer);
}

} // namespace celerity::detail::backend
