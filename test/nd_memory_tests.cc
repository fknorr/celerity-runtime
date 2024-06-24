#include "nd_memory.h"
#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;


TEMPLATE_TEST_CASE_SIG("nd_copy_host works correctly in all source- and destination layouts", "[memory]", ((int Dims), Dims), 0, 1, 2, 3) {
	const auto copy_range = test_utils::truncate_range<Dims>({5, 6, 7});

	int source_shift[Dims];
	int dest_shift[Dims];
	if constexpr(Dims > 0) { source_shift[0] = GENERATE(values({-2, 0, 2})), dest_shift[0] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 1) { source_shift[1] = GENERATE(values({-2, 0, 2})), dest_shift[1] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 2) { source_shift[2] = GENERATE(values({-2, 0, 2})), dest_shift[2] = GENERATE(values({-2, 0, 2})); }

	range<Dims> source_range = ones;
	range<Dims> dest_range = ones;
	id<Dims> offset_in_source = zeros;
	id<Dims> offset_in_dest = zeros;
	for(int i = 0; i < Dims; ++i) {
		source_range[i] = copy_range[i] + std::abs(source_shift[i]);
		offset_in_source[i] = std::max(0, source_shift[i]);
		dest_range[i] = copy_range[i] + std::abs(dest_shift[i]);
		offset_in_dest[i] = std::max(0, dest_shift[i]);
	}

	CAPTURE(source_range, dest_range, offset_in_source, offset_in_dest, copy_range);

	std::vector<int> source(source_range.size());
	std::iota(source.begin(), source.end(), 1);

	std::vector<int> expected_dest(dest_range.size());
	test_utils::for_each_in_range(copy_range, [&](const id<Dims> id) {
		const auto linear_index_in_source = get_linear_index(source_range, offset_in_source + id);
		const auto linear_index_in_dest = get_linear_index(dest_range, offset_in_dest + id);
		expected_dest[linear_index_in_dest] = source[linear_index_in_source];
	});

	std::vector<int> dest(dest_range.size());
	nd_copy_host(source.data(), dest.data(), range_cast<3>(source_range), range_cast<3>(dest_range), id_cast<3>(offset_in_source), id_cast<3>(offset_in_dest),
	    range_cast<3>(copy_range), sizeof(int));

	CHECK(dest == expected_dest);
}

TEST_CASE("layout_strided_nd_copy selects the minimum number of strides", "[memory]") {
	// all contiguous
	CHECK(layout_strided_nd_copy({1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}) == strided_nd_copy_layout{0, 0, 1, {}});
	CHECK(layout_strided_nd_copy({1, 3, 1}, {1, 1, 1}, {0, 2, 0}, {0, 0, 0}, {1, 1, 1}) == strided_nd_copy_layout{2, 0, 1, {}});
	CHECK(layout_strided_nd_copy({1, 1, 1}, {1, 3, 1}, {0, 0, 0}, {0, 2, 0}, {1, 1, 1}) == strided_nd_copy_layout{0, 2, 1, {}});
	CHECK(layout_strided_nd_copy({5, 3, 2}, {1, 1, 2}, {0, 0, 0}, {0, 0, 0}, {1, 1, 2}) == strided_nd_copy_layout{0, 0, 2, {}});
	CHECK(layout_strided_nd_copy({5, 3, 2}, {1, 1, 2}, {2, 1, 0}, {0, 0, 0}, {1, 1, 2}) == strided_nd_copy_layout{14, 0, 2, {}});
	CHECK(layout_strided_nd_copy({1, 1, 2}, {5, 3, 2}, {0, 0, 0}, {2, 1, 0}, {1, 1, 2}) == strided_nd_copy_layout{0, 14, 2, {}});
	CHECK(layout_strided_nd_copy({5, 1, 3}, {2, 1, 3}, {0, 0, 0}, {0, 0, 0}, {2, 1, 3}) == strided_nd_copy_layout{0, 0, 6, {}});
	CHECK(layout_strided_nd_copy({5, 2, 3}, {7, 2, 3}, {2, 0, 0}, {1, 0, 0}, {2, 2, 3}) == strided_nd_copy_layout{12, 6, 12, {}});
	CHECK(layout_strided_nd_copy({5, 2, 3}, {4, 2, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 3}) == strided_nd_copy_layout{0, 0, 12, {}});

	// one stride
	CHECK(layout_strided_nd_copy({1, 2, 3}, {1, 2, 1}, {0, 0, 0}, {0, 0, 0}, {1, 2, 1}) == strided_nd_copy_layout{0, 0, 1, {{3, 1, 2}}});
	CHECK(layout_strided_nd_copy({1, 2, 3}, {1, 2, 1}, {0, 0, 1}, {0, 0, 0}, {1, 2, 1}) == strided_nd_copy_layout{1, 0, 1, {{3, 1, 2}}});
	CHECK(layout_strided_nd_copy({5, 2, 3}, {4, 2, 1}, {0, 0, 0}, {0, 0, 0}, {2, 1, 1}) == strided_nd_copy_layout{0, 0, 1, {{6, 2, 2}}});
	CHECK(layout_strided_nd_copy({4, 3, 2}, {4, 3, 2}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}) == strided_nd_copy_layout{0, 0, 4, {{6, 6, 2}}});
	CHECK(layout_strided_nd_copy({4, 5, 2}, {4, 5, 2}, {0, 0, 0}, {0, 0, 0}, {2, 4, 2}) == strided_nd_copy_layout{0, 0, 8, {{10, 10, 2}}});
	CHECK(layout_strided_nd_copy({4, 5, 6}, {4, 5, 6}, {0, 0, 0}, {0, 0, 0}, {2, 4, 6}) == strided_nd_copy_layout{0, 0, 24, {{30, 30, 2}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 3, 1}) == strided_nd_copy_layout{0, 0, 1, {{3, 3, 3}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 2, 2}) == strided_nd_copy_layout{0, 0, 2, {{3, 3, 2}}});
	CHECK(layout_strided_nd_copy({4, 1, 4}, {4, 1, 4}, {0, 0, 0}, {0, 0, 0}, {2, 1, 2}) == strided_nd_copy_layout{0, 0, 2, {{4, 4, 2}}});

	// two strides
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}) == strided_nd_copy_layout{0, 0, 2, {{3, 3, 2}, {9, 9, 2}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {1, 0, 0}, {0, 0, 0}, {2, 2, 2}) == strided_nd_copy_layout{9, 0, 2, {{3, 3, 2}, {9, 9, 2}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {1, 1, 0}, {0, 0, 0}, {2, 2, 2}) == strided_nd_copy_layout{12, 0, 2, {{3, 3, 2}, {9, 9, 2}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {1, 0, 0}, {2, 2, 2}) == strided_nd_copy_layout{0, 9, 2, {{3, 3, 2}, {9, 9, 2}}});
	CHECK(layout_strided_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {1, 1, 0}, {2, 2, 2}) == strided_nd_copy_layout{0, 12, 2, {{3, 3, 2}, {9, 9, 2}}});
	CHECK(layout_strided_nd_copy({2, 3, 4}, {3, 6, 5}, {0, 0, 0}, {0, 0, 0}, {2, 3, 4}) == strided_nd_copy_layout{0, 0, 4, {{4, 5, 3}, {12, 30, 2}}});
}