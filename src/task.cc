#include "task.h"

#include <algorithm>

namespace celerity {
namespace detail {

	split_constraints command_group_task::get_split_constraints() const {
		return split_constraints{
		    get_geometry(),
		    get_hint<experimental::hints::tiled_split>() != nullptr,
		};
	}

	std::unordered_set<buffer_id> buffer_access_map::get_accessed_buffers() const {
		std::unordered_set<buffer_id> result;
		for(auto& [bid, _] : m_accesses) {
			result.emplace(bid);
		}
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> buffer_access_map::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(auto& [_, rm] : m_accesses) {
			result.insert(rm->get_access_mode());
		}
		return result;
	}

	GridRegion<3> buffer_access_map::get_mode_requirements(
	    const buffer_id bid, const access_mode mode, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		GridRegion<3> result;
		for(size_t i = 0; i < m_accesses.size(); ++i) {
			if(m_accesses[i].first != bid || m_accesses[i].second->get_access_mode() != mode) continue;
			result = GridRegion<3>::merge(result, get_requirements_for_nth_access(i, kernel_dims, sr, global_size));
		}
		return result;
	}

	GridBox<3> buffer_access_map::get_requirements_for_nth_access(
	    const size_t n, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		const auto& [_, rm] = m_accesses[n];
		return subrange_to_grid_box(apply_range_mapper(rm.get(), chunk<3>{sr.offset, sr.range, global_size}, kernel_dims));
	}

	std::vector<GridBox<3>> buffer_access_map::get_required_contiguous_boxes(
	    const buffer_id bid, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		std::vector<GridBox<3>> boxes;
		for(const auto& [a_bid, a_rm] : m_accesses) {
			if(a_bid != bid) continue;
			auto new_box = subrange_to_grid_box(apply_range_mapper(a_rm.get(), chunk<3>{sr.offset, sr.range, global_size}, kernel_dims));
			// If the new box intersects with any of the existing boxes, their union must be contiguous in the result
			const auto first_intersecting = std::partition(boxes.begin(), boxes.end(), [&](const auto& other_box) { return !other_box.intersects(new_box); });
			if(first_intersecting != boxes.end()) {
				for(auto it = first_intersecting; it != boxes.end(); ++it) {
					new_box = GridBox<3>::span(new_box, *it);
				}
				boxes.erase(first_intersecting, boxes.end());
			}
			boxes.push_back(new_box);
		}
		return boxes;
	}

	void side_effect_map::add_side_effect(const host_object_id hoid, const experimental::side_effect_order order) {
		// TODO for multiple side effects on the same hoid, find the weakest order satisfying all of them
		emplace(hoid, order);
	}
} // namespace detail
} // namespace celerity
