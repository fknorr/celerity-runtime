#pragma once

#include "launcher.h"
#include "ranges.h"
#include "types.h"
#include "workaround.h"

#include <vector>

namespace celerity::detail::backend {

class event {
  public:
	event() = default;
	event(const event&) = delete;
	event(event&&) = delete;
	event& operator=(const event&) = delete;
	event& operator=(event&&) = delete;
	virtual ~event() = default;

	virtual bool is_complete() const = 0;
};

class sycl_event : public event {
  public:
	sycl_event() = default;
	sycl_event(std::vector<sycl::event> wait_list);

	bool is_complete() const override;

  private:
	mutable std::vector<sycl::event> m_incomplete;
};

struct device_config {
	device_id device_id = 0;
	memory_id native_memory = 1;
	sycl::device sycl_device;
};

class queue {
  public:
	queue() = default;
	queue(const queue&) = delete;
	queue& operator=(const queue&) = delete;
	virtual ~queue() = default;

	[[nodiscard]] virtual std::pair<void*, std::unique_ptr<event>> malloc(memory_id where, size_t size, size_t alignment) = 0;

	[[nodiscard]] virtual std::unique_ptr<event> free(memory_id where, void* allocation) = 0;

	[[nodiscard]] virtual std::unique_ptr<event> memcpy_strided_device(int dims, memory_id source, memory_id dest, const void* source_base_ptr,
	    void* target_base_ptr, size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range,
	    const id<3>& target_offset, const range<3>& copy_range) = 0;

	[[nodiscard]] virtual std::unique_ptr<event> launch_kernel(device_id did, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range,
	    const std::vector<void*>& reduction_ptrs, bool is_reduction_initializer) = 0;
};

std::unique_ptr<event> launch_sycl_kernel(sycl::queue& queue, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range,
    const std::vector<void*>& reduction_ptrs, bool is_reduction_initializer);

void handle_sycl_errors(const sycl::exception_list& errors);

} // namespace celerity::detail::backend
