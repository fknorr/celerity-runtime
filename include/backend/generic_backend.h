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
	explicit generic_queue(const std::vector<device_config>& devices);

	std::pair<void*, std::unique_ptr<event>> malloc(memory_id where, size_t size, size_t alignment) override;

	std::unique_ptr<event> free(memory_id where, void* allocation) override;

	std::unique_ptr<event> memcpy_strided_device(int dims, memory_id source, memory_id target, const void* source_base_ptr, void* target_base_ptr,
	    size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset,
	    const range<3>& copy_range) override;

	std::unique_ptr<event> launch_kernel(
	    device_id did, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	std::unordered_map<device_id, sycl::queue> m_device_queues;
	std::unordered_map<memory_id, sycl::queue> m_memory_queues;
};

} // namespace celerity::detail::backend
