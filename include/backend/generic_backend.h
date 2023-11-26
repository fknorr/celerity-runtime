#pragma once

#include "ranges.h"

#include "backend/queue.h"
#include "backend/type.h"


namespace celerity::detail::backend {

class generic_queue : public queue {
  public:
	explicit generic_queue(const std::vector<device_config>& devices);

	void* malloc(memory_id where, size_t size, size_t alignment) override;

	void free(memory_id where, void* allocation) override;

	async_event memcpy_strided_device(int dims, memory_id source, memory_id target, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset,
	    const range<3>& copy_range) override;

	async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const subrange<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	std::unordered_map<device_id, sycl::queue> m_device_queues;
	std::unordered_map<memory_id, sycl::queue> m_memory_queues;
};

} // namespace celerity::detail::backend
