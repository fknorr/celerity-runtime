#include "backend/backend.h"

#include "ranges.h"
#include "types.h"
#include <hipSYCL/sycl/usm.hpp>

namespace celerity::detail::backend_detail {

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<0>& /* source_range */, const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */,
    const range<0>& /* copy_range */) {
	return {queue.memcpy(target_base_ptr, source_base_ptr, elem_size)};
}

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<1>& source_range, const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range) {
	const size_t line_size = elem_size * copy_range[0];
	return {queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size)};
}

// TODO Optimize for contiguous copies?
std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<2>& source_range, const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range) {
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	const size_t line_size = elem_size * copy_range[1];
	std::vector<sycl::event> wait_list{copy_range[0]};
	for(size_t i = 0; i < copy_range[0]; ++i) {
		auto e = queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
		    static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		wait_list[i] = std::move(e);
	}
	return wait_list;
}

// TODO Optimize for contiguous copies?
std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
	const auto source_base_offset =
	    get_linear_index(source_range, source_offset) - get_linear_index(range<2>{source_range[1], source_range[2]}, id<2>{source_offset[1], source_offset[2]});
	const auto target_base_offset =
	    get_linear_index(target_range, target_offset) - get_linear_index(range<2>{target_range[1], target_range[2]}, id<2>{target_offset[1], target_offset[2]});

	std::vector<sycl::event> wait_list;
	wait_list.reserve(copy_range[0] * copy_range[1]);
	for(size_t i = 0; i < copy_range[0]; ++i) {
		const auto* const source_ptr = static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
		auto* const target_ptr = static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
		const auto events = memcpy_strided_device_generic(queue, source_ptr, target_ptr, elem_size, range<2>{source_range[1], source_range[2]},
		    id<2>{source_offset[1], source_offset[2]}, range<2>{target_range[1], target_range[2]}, id<2>{target_offset[1], target_offset[2]},
		    range<2>{copy_range[1], copy_range[2]});
		wait_list.insert(wait_list.end(), events.begin(), events.end());
	}
	return wait_list;
}

} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

std::vector<sycl::device> get_device_vector(const std::vector<std::pair<device_id, sycl::device>>& devices) {
	std::vector<sycl::device> vector;
	vector.reserve(devices.size());
	for(auto& [did, dev] : devices) {
		vector.push_back(dev);
	}
	return vector;
}

generic_queue::generic_queue(const std::vector<device_config>& devices) {
	m_memory_queues.emplace(host_memory_id, sycl::queue());

	for(const auto& config : devices) {
		assert(m_device_queues.count(config.device_id) == 0);
		assert(m_memory_queues.count(config.native_memory) == 0); // TODO handle devices that share memory

		sycl::queue queue(config.sycl_device);
		m_device_queues.emplace(config.device_id, queue);
		m_memory_queues.emplace(config.native_memory, queue);
	}
}

std::pair<void*, std::unique_ptr<event>> generic_queue::malloc(const memory_id where, const size_t size, [[maybe_unused]] const size_t alignment) {
	void* ptr;
	if(where == host_memory_id) {
		ptr = sycl::aligned_alloc_host(alignment, size, m_memory_queues.at(host_memory_id));
	} else {
		ptr = sycl::aligned_alloc_device(alignment, size, m_memory_queues.at(where));
	}
	return {ptr, std::make_unique<sycl_event>()}; // synchronous
}

std::unique_ptr<event> generic_queue::free(const memory_id where, void* const allocation) {
	sycl::free(allocation, m_memory_queues.at(where));
	return std::make_unique<sycl_event>(); // synchronous
}

std::unique_ptr<event> generic_queue::memcpy_strided_device(const int dims, const memory_id source, const memory_id target, const void* const source_base_ptr,
    void* const target_base_ptr, const size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range,
    const id<3>& target_offset, const range<3>& copy_range) {
	assert(source != host_memory_id || target != host_memory_id);
	auto& queue = m_memory_queues.at(source == host_memory_id ? target : source);
	const auto dispatch_memcpy = [&](const auto dims) {
		return std::make_unique<sycl_event>(backend_detail::memcpy_strided_device_generic(queue, source_base_ptr, target_base_ptr, elem_size,
		    range_cast<dims.value>(source_range), id_cast<dims.value>(source_offset), range_cast<dims.value>(target_range), id_cast<dims.value>(target_offset),
		    range_cast<dims.value>(copy_range)));
	};
	switch(dims) {
	case 0: return dispatch_memcpy(std::integral_constant<int, 0>());
	case 1: return dispatch_memcpy(std::integral_constant<int, 1>());
	case 2: return dispatch_memcpy(std::integral_constant<int, 2>());
	case 3: return dispatch_memcpy(std::integral_constant<int, 3>());
	default: abort();
	}
}

std::unique_ptr<event> generic_queue::launch_kernel(device_id did, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range,
    const std::vector<void*>& reduction_ptrs, bool is_reduction_initializer) {
	return launch_sycl_kernel(m_device_queues.at(did), launcher, execution_range, reduction_ptrs, is_reduction_initializer);
}

} // namespace celerity::detail::backend
