#pragma once

#include "ranges.h"
#include "types.h"
#include "workaround.h"

namespace celerity::detail::backend {

class queue {
  public:
	class event {
	  public:
		event() = default;
		event(const event&) = delete;
		event& operator=(const event&) = delete;
		virtual ~event() = default;

		virtual bool is_complete() const = 0;
	};

	virtual ~queue() = default;

	virtual void add_device(device_id device, sycl::queue& queue) = 0;

	virtual std::unique_ptr<event> memcpy_strided_device(int dims, memory_id source, memory_id dest, const void* source_base_ptr, void* target_base_ptr,
	    size_t elem_size, const range<3>& source_range, const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset,
	    const range<3>& copy_range) = 0;
};

} // namespace celerity::detail::backend
