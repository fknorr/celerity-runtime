#pragma once

#include "async_event.h"
#include "launcher.h"
#include "types.h"

#include <vector>

#include <sycl/sycl.hpp>

namespace celerity::detail {

struct system_info;

class backend {
  public:
	backend() = default;
	backend(const backend&) = delete;
	backend(backend&&) = delete;
	backend& operator=(const backend&) = delete;
	backend& operator=(backend&&) = delete;
	virtual ~backend() = default;

	virtual const system_info& get_system_info() const = 0;

	virtual void* debug_alloc(size_t size) = 0;

	virtual void debug_free(void* ptr) = 0;

	virtual async_event enqueue_host_alloc(size_t size, size_t alignment) = 0;

	virtual async_event enqueue_device_alloc(device_id memory_device, size_t size, size_t alignment) = 0;

	virtual async_event enqueue_host_free(void* ptr) = 0;

	virtual async_event enqueue_device_free(device_id memory_device, void* ptr) = 0;

	virtual async_event enqueue_host_function(size_t host_lane, std::function<void()> fn) = 0;

	virtual async_event enqueue_device_kernel(device_id device, size_t device_lane, const device_kernel_launcher& launcher, const box<3>& execution_range,
	    const std::vector<void*>& reduction_ptrs) = 0;

	virtual async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
	    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) = 0;

	virtual async_event enqueue_host_copy(size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) = 0;
};

} // namespace celerity::detail
