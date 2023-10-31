#pragma once

#include "ranges.h"

namespace celerity {
namespace detail {

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<0>& source_range, const id<0>& source_offset,
	    const range<0>& target_range, const id<0>& target_offset, const range<0>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<1>& source_range, const id<1>& source_offset,
	    const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<2>& source_range, const id<2>& source_offset,
	    const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<3>& source_range, const id<3>& source_offset,
	    const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range);

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr);

}}