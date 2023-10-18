#pragma once

#include "types.h"

#include <functional>
#include <numeric>

namespace celerity::detail {

using host_reduction_fn = std::function<void(void* dest, const void* src, size_t src_count, bool include_dest)>;

template <typename Scalar, typename BinaryOp>
host_reduction_fn make_host_reduction_fn(const BinaryOp op, const Scalar identity) {
	return [=](void* const dest, const void* const src, const size_t src_count, const bool include_dest) {
		const auto v_dest = static_cast<Scalar*>(dest);
		const auto v_src = static_cast<const Scalar*>(src);
		*v_dest = std::reduce(v_src, v_src + src_count, include_dest ? *v_dest : identity, op);
	};
}

struct reduction_info {
	reduction_id rid = 0;
	buffer_id bid = 0;
	bool init_from_buffer = false;
};

} // namespace celerity::detail
