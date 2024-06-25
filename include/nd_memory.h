#pragma once

#include "grid.h"
#include "ranges.h"

#include <string.h>

namespace celerity::detail {

struct nd_copy_layout {
	struct stride_dimension {
		size_t source_stride = 1;
		size_t dest_stride = 1;
		size_t count = 1;

		friend bool operator==(const stride_dimension& lhs, const stride_dimension& rhs) {
			return lhs.source_stride == rhs.source_stride && lhs.dest_stride == rhs.dest_stride && lhs.count == rhs.count;
		}
		friend bool operator!=(const stride_dimension& lhs, const stride_dimension& rhs) { return !(lhs == rhs); }
	};

	size_t linear_offset_in_source = 0;
	size_t linear_offset_in_dest = 0;
	int num_complex_strides = 0;
	stride_dimension strides[2];
	size_t contiguous_range = 0;

	friend bool operator==(const nd_copy_layout& lhs, const nd_copy_layout& rhs) {
		return lhs.linear_offset_in_source == rhs.linear_offset_in_source && lhs.linear_offset_in_dest == rhs.linear_offset_in_dest
		       && lhs.contiguous_range == rhs.contiguous_range && lhs.num_complex_strides == rhs.num_complex_strides && lhs.strides[0] == rhs.strides[0]
		       && lhs.strides[1] == rhs.strides[1];
	}

	friend bool operator!=(const nd_copy_layout& lhs, const nd_copy_layout& rhs) { return !(lhs == rhs); }
};

inline nd_copy_layout layout_nd_copy(
    const range<3>& source_range, const range<3>& dest_range, const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range) {
	assert(all_true(offset_in_source + copy_range <= source_range));
	assert(all_true(offset_in_dest + copy_range <= dest_range));

	if(copy_range.size() == 0) return {};

	nd_copy_layout layout;
	layout.linear_offset_in_source = get_linear_index(source_range, offset_in_source);
	layout.linear_offset_in_dest = get_linear_index(dest_range, offset_in_dest);
	layout.contiguous_range = 1;
	size_t* current_range = &layout.contiguous_range;
	size_t next_source_step = 1;
	size_t next_dest_step = 1;
	bool contiguous = true;
	for(int d = 2; d >= 0; --d) {
		if(!contiguous && copy_range[d] != 1) {
			++layout.num_complex_strides;
			layout.strides[1] = layout.strides[0];
			layout.strides[0] = {next_source_step, next_dest_step, 1};
			current_range = &layout.strides[0].count;
			contiguous = true;
		}
		next_source_step *= source_range[d];
		next_dest_step *= dest_range[d];
		*current_range *= copy_range[d];
		if(source_range[d] != copy_range[d] || dest_range[d] != copy_range[d]) { contiguous = false; }
	}

	return layout;
}

template <typename F>
inline void for_each_contiguous_chunk(const nd_copy_layout& layout, F&& f) {
	if(layout.contiguous_range == 0) return;

	size_t source_offset_0 = layout.linear_offset_in_source;
	size_t dest_offset_0 = layout.linear_offset_in_dest;
	for(size_t i = 0; i < layout.strides[0].count; ++i) {
		size_t source_offset_1 = source_offset_0;
		size_t dest_offset_1 = dest_offset_0;
		for(size_t j = 0; j < layout.strides[1].count; ++j) {
			f(source_offset_1, dest_offset_1, layout.contiguous_range);
			source_offset_1 += layout.strides[1].source_stride;
			dest_offset_1 += layout.strides[1].dest_stride;
		}
		source_offset_0 += layout.strides[0].source_stride;
		dest_offset_0 += layout.strides[0].dest_stride;
	}
}

inline void nd_copy_host(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range,
    const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size) //
{
	const auto layout = layout_nd_copy(source_range, dest_range, offset_in_source, offset_in_dest, copy_range);
	for_each_contiguous_chunk(layout, [&](const size_t chunk_offset_in_source, const size_t chunk_offset_in_dest, const size_t chunk_size) {
		memcpy(static_cast<std::byte*>(dest_base) + chunk_offset_in_dest * elem_size,
		    static_cast<const std::byte*>(source_base) + chunk_offset_in_source * elem_size, chunk_size * elem_size);
	});
}

inline void nd_copy_host(
    const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	nd_copy_host(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
	    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size);
}

inline void nd_copy_host(const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region,
    const size_t elem_size) //
{
	for(const auto& copy_box : copy_region.get_boxes()) {
		nd_copy_host(source_base, dest_base, source_box, dest_box, copy_box, elem_size);
	}
}

} // namespace celerity::detail
