#pragma once

#include "grid.h"

#include <algorithm>
#include <vector>

namespace celerity::detail {

class bounding_box_set : private box_vector<3> {
  private:
	using vector = box_vector<3>;

  public:
	using typename vector::const_iterator;
	using typename vector::value_type;
	using iterator = const_iterator;

	using vector::empty;
	using vector::size;
	using vector::swap;

	iterator begin() const { return vector::begin(); } // only export const overload
	iterator end() const { return vector::end(); }     // only export const overload

	void insert(const box<3>& box) {
		auto new_box = box;
		for(;;) { // replacing two partially overlapping boxes with their span can create new intersections in space previously not covered
			const auto first_intersecting =
			    std::partition(vector::begin(), vector::end(), [&](const auto& other_box) { return box_intersection(other_box, new_box).empty(); });
			if(first_intersecting == vector::end()) break;
			for(auto it = first_intersecting; it != vector::end(); ++it) {
				new_box = bounding_box(new_box, *it);
			}
			vector::erase(first_intersecting, end());
		}
		vector::push_back(new_box);
	}

	template <typename Iterator>
	void insert(const Iterator first, const Iterator last) {
		while(first != last) {
			bounding_box_set::insert(*first++);
		}
	}

	vector into_vector() && { return std::move(*this); }
};

} // namespace celerity::detail
