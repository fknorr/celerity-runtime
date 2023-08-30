#include "backend/generic_backend.h"

#include "ranges.h"

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

class generic_event : public queue::event {
  public:
	generic_event(std::vector<sycl::event> wait_list) : m_incomplete(std::move(wait_list)) {}

	bool is_complete() const override {
		const auto last_incomplete = std::remove_if(m_incomplete.begin(), m_incomplete.end(),
		    [](const sycl::event& evt) { return evt.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; });
		m_incomplete.erase(last_incomplete, m_incomplete.end());
		return m_incomplete.empty();
	}

  private:
	mutable std::vector<sycl::event> m_incomplete;
};

void generic_queue::add_device(const device_id device, sycl::queue& queue) { m_device_queues.emplace(device, &queue); }

std::unique_ptr<queue::event> generic_queue::memcpy_strided_device(const int dims, const memory_id source, const memory_id target,
    const void* const source_base_ptr, void* const target_base_ptr, const size_t elem_size, const range<3>& source_range, const id<3>& source_offset,
    const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	assert(source != host_memory_id || target != host_memory_id);
	auto& queue = *m_device_queues.at(to_device_id(source == host_memory_id ? target : source));
	const auto dispatch_memcpy = [&](const auto dims) {
		return std::make_unique<generic_event>(backend_detail::memcpy_strided_device_generic(queue, source_base_ptr, target_base_ptr, elem_size,
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

} // namespace celerity::detail::backend
