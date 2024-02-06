#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>

namespace celerity::detail {

/// Like a simple std::unordered_map, but implemented by indexing into a vector with the integral key type.
// TODO I'm taking bikeshedding suggestions for the name. dense_map? integral_map? vector_map?
template <typename KeyId, typename Value>
class dense_map : private std::vector<Value> {
  private:
	using vector = std::vector<Value>;

  public:
	dense_map() = default;
	explicit dense_map(const size_t size) : vector(size) {}

	using vector::begin, vector::end, vector::cbegin, vector::cend, vector::empty, vector::size, vector::resize;

	Value& operator[](const KeyId key) {
		assert(key < size());
		return vector::operator[](static_cast<size_t>(key));
	}

	const Value& operator[](const KeyId key) const {
		assert(key < size());
		return vector::operator[](static_cast<size_t>(key));
	}
};

} // namespace celerity::detail