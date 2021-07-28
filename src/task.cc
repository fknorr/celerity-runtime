#include "task.h"

#include <algorithm>

namespace celerity {
namespace detail {

	std::unordered_set<buffer_id> buffer_access_map::get_accessed_buffers() const {
		std::unordered_set<buffer_id> result;
		for(auto& [bid, _] : map) {
			result.emplace(bid);
		}
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> buffer_access_map::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(auto [first, last] = map.equal_range(bid); first != last; ++first) {
			result.insert(first->second->get_access_mode());
		}
		return result;
	}

	template <int KernelDims>
	subrange<3> apply_range_mapper(range_mapper_base const* rm, chunk<KernelDims> chnk) {
		switch(rm->get_buffer_dimensions()) {
		case 1: return subrange_cast<3>(rm->map_1(chnk));
		case 2: return subrange_cast<3>(rm->map_2(chnk));
		case 3: return rm->map_3(chnk);
		default: assert(false);
		}
		return subrange<3>{};
	}

	GridRegion<3> buffer_access_map::get_requirements_for_access(
	    buffer_id bid, cl::sycl::access::mode mode, int kernel_dims, const subrange<3>& sr, const cl::sycl::range<3>& global_size) const {
		auto [first, last] = map.equal_range(bid);
		if(first == map.end()) { return {}; }

		GridRegion<3> result;
		for(auto iter = first; iter != last; ++iter) {
			auto range = iter->second.get();
			if(range->get_access_mode() != mode) continue;

			chunk<3> chnk{sr.offset, sr.range, global_size};
			subrange<3> req;
			switch(kernel_dims) {
			case 0:
				[[fallthrough]]; // cl::sycl::range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
				                 // anyway, we require range mappers to take at least 1d subranges
			case 1: req = apply_range_mapper<1>(range, chunk_cast<1>(chnk)); break;
			case 2: req = apply_range_mapper<2>(range, chunk_cast<2>(chnk)); break;
			case 3: req = apply_range_mapper<3>(range, chunk_cast<3>(chnk)); break;
			default: assert(!"Unreachable");
			}
			result = GridRegion<3>::merge(result, subrange_to_grid_box(req));
		}

		return result;
	}

} // namespace detail
} // namespace celerity
