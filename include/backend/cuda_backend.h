#pragma once

#include "backend/queue.h"
#include "backend/type.h"
#include "ranges.h"


namespace celerity::detail::backend {

class cuda_queue final : public queue {
  public:
	using cuda_device_id = int;

	explicit cuda_queue(const std::vector<device_config>& devices);
	~cuda_queue() override;

	void init() override;

	void* malloc(memory_id where, size_t size, size_t alignment) override;

	void free(memory_id where, void* allocation) override;

	async_event memcpy_strided_device(int dims, memory_id source, memory_id dest, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset,
	    const range<3>& copy_range) override;

	async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const subrange<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail::backend
