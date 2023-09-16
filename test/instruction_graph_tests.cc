#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "distributed_graph_generator_test_utils.h"
#include "instruction_graph_generator.h"
#include "print_graph.h"
#include "recorders.h"
#include "task_ring_buffer.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;

TEST_CASE("trivial graph", "[instruction graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	ictx.device_compute<class UKN(kernel)>(test_range).submit();
}

TEST_CASE("graph with only writes", "[instruction graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
}

TEST_CASE("resize and overwrite", "[instruction graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer(range<1>(256));
	ictx.device_compute<class UKN(writer)>(range<1>(1)).discard_write(buf1, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute<class UKN(writer)>(range<1>(1)).discard_write(buf1, acc::fixed<1>({64, 196})).submit();
	// TODO assert that we do not copy the overwritten buffer portion
}

TEST_CASE("communication-free dataflow", "[instruction graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::one_to_one()).submit();
}

TEST_CASE("communication-free dataflow with copies", "[instruction graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("simple communication", "[instruction graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("large graph", "[instruction graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);

	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	auto buf2 = ictx.create_buffer(test_range);

	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf2, acc::all()).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf2, acc::all()).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf2, acc::all()).submit();
}

TEST_CASE("recv split", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);

	const auto reverse_one_to_one = [](chunk<1> ck) -> subrange<1> { return {ck.global_size[0] - ck.range[0] - ck.offset[0], ck.range[0]}; };

	const range<1> test_range = {256};
	auto buf = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf, reverse_one_to_one).submit();
}

TEST_CASE("transitive copy dependencies", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);

	const auto reverse_one_to_one = [](chunk<1> ck) -> subrange<1> { return {ck.global_size[0] - ck.range[0] - ck.offset[0], ck.range[0]}; };

	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	auto buf2 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf2, reverse_one_to_one).submit();
}

TEST_CASE("RSim pattern", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);
	size_t width = 1000;
	size_t n_iters = 3;
	auto buf = ictx.create_buffer(range<2>(n_iters, width));

	const auto access_up_to_ith_line_all = [&](size_t i) { //
		return celerity::access::fixed<2>({{0, 0}, {i, width}});
	};
	const auto access_ith_line_1to1 = [](size_t i) {
		return [i](celerity::chunk<2> chnk) { return celerity::subrange<2>({i, chnk.offset[0]}, {1, chnk.range[0]}); };
	};

	for(size_t i = 0; i < n_iters; ++i) {
		ictx.device_compute<class UKN(rsim)>(range<2>(width, width))
		    .read(buf, access_up_to_ith_line_all(i))
		    .discard_write(buf, access_ith_line_1to1(i))
		    .submit();
	}
}

TEST_CASE("hello world pattern (host initialization)", "[instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const std::string input_str = "Ifmmp!Xpsme\"\x01";
	auto buf = ictx.create_buffer<1>(input_str.size(), true /* host initialized */);

	ictx.device_compute(buf.get_range()).read_write(buf, acc::one_to_one()).submit();
	// ictx.fence(buf); TODO currently unimplemented
}

// TODO a test with impossible requirements (overlapping writes maybe?)
// TODO an oversubscribed host task with side effects
