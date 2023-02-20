#pragma once

#include "grid.h"
#include "ranges.h"

namespace celerity {

namespace detail {
	std::ostream& print_chunk3(std::ostream& os, chunk<3> chnk3);
	std::ostream& print_subrange3(std::ostream& os, subrange<3> subr3);
} // namespace detail

template <int Dims>
std::ostream& operator<<(std::ostream& os, chunk<Dims> chnk) {
	return detail::print_chunk3(os, detail::chunk_cast<3>(chnk));
}

template <int Dims>
std::ostream& operator<<(std::ostream& os, subrange<Dims> subr) {
	return detail::print_subrange3(os, detail::subrange_cast<3>(subr));
}

} // namespace celerity

namespace celerity::detail {

inline GridRegion<3> merge_regions(const std::vector<GridRegion<3>>& regions) {
	GridRegion<3> merged;
	for(const auto& r : regions) {
		merged = GridRegion<3>::merge(merged, r);
	}
	return merged;
}

} // namespace celerity::detail
