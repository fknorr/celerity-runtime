#pragma once

#include "backend/queue.h"


namespace celerity::detail::backend {

class cuda_queue final : public queue {
  public:
	using cuda_device_id = int;

	explicit cuda_queue(const std::vector<device_config>& devices);
	~cuda_queue() override;

	void init() override;

	void* alloc(memory_id where, size_t size, size_t alignment) override;

	void free(memory_id where, void* allocation) override;

	async_event nd_copy(memory_id source_mid, memory_id dest_mid, const void* source_base, void* dest_base, const range<3>& source_range,
	    const range<3>& dest_range, const id<3>& source_offset, const id<3>& dest_offset, const range<3>& copy_range, size_t elem_size) override;

	async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const subrange<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail::backend
