#include "split.h"

#include <array>
#include <tuple>

#include "grid.h"

namespace {

using namespace celerity;
using namespace celerity::detail;

void sanity_check_split(const chunk<3>& full_chunk, const std::vector<chunk<3>>& split) {
	region<3> reconstructed_chunk;
	for(auto& chnk : split) {
		assert(region_intersection(reconstructed_chunk, box<3>(chnk)).empty());
		reconstructed_chunk = region_union(box<3>(chnk), reconstructed_chunk);
	}
	assert(region_difference(reconstructed_chunk, box<3>(full_chunk)).empty());
}

template <int Dims>
std::tuple<std::array<size_t, Dims>, std::array<size_t, Dims>, std::array<size_t, Dims>> compute_small_and_large_chunks(
    const chunk<3>& full_chunk, const range<3>& granularity, const std::array<size_t, Dims>& actual_num_chunks) {
	std::array<size_t, Dims> small_chunk_size{};
	std::array<size_t, Dims> large_chunk_size{};
	std::array<size_t, Dims> num_large_chunks{};
	for(int d = 0; d < Dims; ++d) {
		const size_t ideal_chunk_size = full_chunk.range[d] / actual_num_chunks[d];
		small_chunk_size[d] = (ideal_chunk_size / granularity[d]) * granularity[d];
		large_chunk_size[d] = small_chunk_size[d] + granularity[d];
		num_large_chunks[d] = (full_chunk.range[d] - small_chunk_size[d] * actual_num_chunks[d]) / granularity[d];
	}
	return {small_chunk_size, large_chunk_size, num_large_chunks};
}

} // namespace

namespace celerity::detail {

std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks) {
#ifndef NDEBUG
	assert(num_chunks > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_chunk.range[d] % granularity[d] == 0);
	}
#endif

	// Due to split granularity requirements or if num_workers > global_size[0],
	// we may not be able to create the requested number of chunks.
	const std::array<size_t, 1> actual_num_chunks = {std::min(num_chunks, full_chunk.range[0] / granularity[0])};
	const auto [small_chunk_size, large_chunk_size, num_large_chunks] = compute_small_and_large_chunks<1>(full_chunk, granularity, actual_num_chunks);

	std::vector<chunk<3>> result(actual_num_chunks[0], {full_chunk.offset, full_chunk.range, full_chunk.global_size});
	for(auto i = 0u; i < num_large_chunks[0]; ++i) {
		result[i].range[0] = large_chunk_size[0];
		result[i].offset[0] += i * large_chunk_size[0];
	}
	for(auto i = num_large_chunks[0]; i < actual_num_chunks[0]; ++i) {
		result[i].range[0] = small_chunk_size[0];
		result[i].offset[0] += num_large_chunks[0] * large_chunk_size[0] + (i - num_large_chunks[0]) * small_chunk_size[0];
	}

#ifndef NDEBUG
	sanity_check_split(full_chunk, result);
#endif

	return result;
}

// TODO: Make the split dimensions configurable for 3D chunks?
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks) {
#ifndef NDEBUG
	assert(num_chunks > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_chunk.range[d] % granularity[d] == 0);
	}
#endif

	const auto assign_factors = [&full_chunk, &granularity, &num_chunks](const size_t factor) {
		assert(num_chunks % factor == 0);
		const size_t max_chunks[2] = {full_chunk.range[0] / granularity[0], full_chunk.range[1] / granularity[1]};
		const size_t f0 = factor;
		const size_t f1 = num_chunks / factor;

		// Decide in which direction to split by first checking which
		// factor assignment produces more chunks under the given constraints.
		const std::array<size_t, 2> split0 = {std::min(f0, max_chunks[0]), std::min(f1, max_chunks[1])};
		const std::array<size_t, 2> split1 = {std::min(f1, max_chunks[0]), std::min(f0, max_chunks[1])};
		const auto count0 = split0[0] * split0[1];
		const auto count1 = split1[0] * split1[1];

		if(count0 == count1) {
			// If we're tied for the number of chunks we can create, try some heuristics to decide.

			// If domain is square(-ish), prefer splitting along slower dimension.
			// (These bounds have been chosen arbitrarily!)
			const double squareishness = std::sqrt(full_chunk.range.size()) / static_cast<double>(full_chunk.range[0]);
			if(squareishness > 0.95 && squareishness < 1.05) { return (f0 >= f1) ? split0 : split1; }

			// For non-square domains, prefer split that produces shorter edges (compare sum of circumferences)
			const auto circ0 = full_chunk.range[0] / split0[0] + full_chunk.range[1] / split0[1];
			const auto circ1 = full_chunk.range[0] / split1[0] + full_chunk.range[1] / split1[1];
			return circ0 < circ1 ? split0 : split1;

			// TODO: Yet another heuristic we may want to consider is how even chunk sizes are,
			// i.e., how balanced the workload is.
		}
		if(count0 > count1) { return split0; }
		return split1;
	};

	// Factorize num_chunks
	// Try to find factors as close to the square root as possible, that also produce
	// (or come close to) the requested number of chunks (under the given constraints).
	size_t f = std::floor(std::sqrt(num_chunks));
	std::array<size_t, 2> best_f_counts = {0, 0};
	while(f >= 1) {
		while(f > 1 && num_chunks % f != 0) {
			f--;
		}
		const auto counts = assign_factors(f);
		if(counts[0] * counts[1] > best_f_counts[0] * best_f_counts[1]) { best_f_counts = counts; }
		if(counts[0] * counts[1] == num_chunks) { break; }
		f--;
	}
	const auto actual_num_chunks = best_f_counts;
	const auto [small_chunk_size, large_chunk_size, num_large_chunks] = compute_small_and_large_chunks<2>(full_chunk, granularity, actual_num_chunks);

	std::vector<chunk<3>> result(actual_num_chunks[0] * actual_num_chunks[1], {full_chunk.offset, full_chunk.range, full_chunk.global_size});
	id<3> offset = full_chunk.offset;

	for(size_t j = 0; j < actual_num_chunks[0]; ++j) {
		range<2> chunk_size = {(j < num_large_chunks[0]) ? large_chunk_size[0] : small_chunk_size[0], 0};
		for(size_t i = 0; i < actual_num_chunks[1]; ++i) {
			chunk_size[1] = (i < num_large_chunks[1]) ? large_chunk_size[1] : small_chunk_size[1];
			auto& chnk = result[j * actual_num_chunks[1] + i];
			chnk.offset = offset;
			chnk.range[0] = chunk_size[0];
			chnk.range[1] = chunk_size[1];
			offset[1] += chunk_size[1];
		}
		offset[0] += chunk_size[0];
		offset[1] = full_chunk.offset[1];
	}

#ifndef NDEBUG
	sanity_check_split(full_chunk, result);
#endif

	return result;
}
} // namespace celerity::detail
