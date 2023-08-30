#pragma once

#include "ranges.h"

#include "backend/operations.h"
#include "backend/queue.h"
#include "backend/type.h"

namespace celerity::detail::backend_detail {

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<0>& source_range, const id<0>& source_offset, const range<0>& target_range, const id<0>& target_offset, const range<0>& copy_range);

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<1>& source_range, const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range);

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<2>& source_range, const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range);

std::vector<sycl::event> memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range);

template <>
struct backend_operations<backend::type::generic> {
	template <typename... Args>
	static void memcpy_strided_device(Args&&... args) {
		sycl::event::wait(memcpy_strided_device_generic(args...));
	}
};

} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

class generic_queue : public queue {
  public:
	class event : public queue::event {
	  public:
		event(std::vector<sycl::event> wait_list) : m_incomplete(std::move(wait_list)) {}

		bool is_complete() const override {
			const auto last_incomplete = std::remove_if(m_incomplete.begin(), m_incomplete.end(), [](const sycl::event& evt) {
				return evt.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
			});
			m_incomplete.erase(last_incomplete, m_incomplete.end());
			return m_incomplete.empty();
		}

	  private:
		mutable std::vector<sycl::event> m_incomplete;
	};

	void add_device(device_id device, sycl::queue& queue) override;

	std::unique_ptr<queue::event> memcpy_strided_device(int dims, memory_id source, memory_id target, const void* source_base_ptr, void* target_base_ptr,
	    size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset,
	    const range<3>& copy_range) override;

  private:
	std::unordered_map<device_id, sycl::queue*> m_device_queues;
};

} // namespace celerity::detail::backend
