#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "distributed_graph_generator_test_utils.h"
#include "instruction_graph_generator.h"
#include "recorders.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;

TEST_CASE("A command group without data access compiles to a trivial graph", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	ictx.device_compute(test_range).name("kernel").submit();
	ictx.finish();

	const auto iq = ictx.query<instruction_record>();
	CHECK(iq.count() == 3);
	CHECK(iq.count<epoch_instruction_record>() == 2);
	CHECK(iq.count<launch_instruction_record>() == 1);

	const auto kernel = iq.select<launch_instruction_record>("kernel");
	CHECK(kernel.predecessors().all<epoch_instruction_record>());
	CHECK(kernel.successors().all<epoch_instruction_record>());
}

TEST_CASE("allocations and kernels are split between devices", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf1, acc::one_to_one()).submit();
	ictx.finish();

	const auto iq = ictx.query<instruction_record>();

	CHECK(iq.select<alloc_instruction_record>().predecessors().all<epoch_instruction_record>());

	const auto writers = iq.select<launch_instruction_record>();
	CHECK(writers.count() == 2);
	CHECK(writers.all("writer"));
	CHECK(writers[0].unique().device_id != writers[1].unique().device_id);

	for(const auto& w : writers.each()) {
		CHECK(w.predecessors().all<alloc_instruction_record>());
		const auto alloc = w.predecessors().unique<alloc_instruction_record>();
		CHECK(alloc.memory_id == ictx.get_native_memory(w.unique().device_id.value()));
		CHECK(alloc.buffer_allocation.value().box == w.unique().access_map.front().accessed_box_in_buffer);

		CHECK(w.successors().all<free_instruction_record>());
		const auto free = w.successors().unique<free_instruction_record>();
		CHECK(free.memory_id == alloc.memory_id);
	}

	CHECK(iq.select<free_instruction_record>().successors().all<epoch_instruction_record>());
}

TEST_CASE("resize and overwrite", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer(range<1>(256));
	ictx.device_compute(range<1>(1)).name("1st writer").discard_write(buf1, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range<1>(1)).name("2nd writer").discard_write(buf1, acc::fixed<1>({64, 196})).submit();
	ictx.finish();

	const auto first_writer_iq = ictx.query<launch_instruction_record>("1st writer");
	const auto first_alloc_iq = first_writer_iq.predecessors<alloc_instruction_record>().check_count(1);
	CHECK(first_alloc_iq.unique().buffer_allocation.value().box == box_cast<3>(box<1>(0, 128)));

	const auto second_writer_iq = ictx.query<launch_instruction_record>("2nd writer");
	const auto second_alloc_iq = second_writer_iq.predecessors<alloc_instruction_record>();
	CHECK(second_alloc_iq.count() == 1); // does not depend on copy
	CHECK(second_alloc_iq.unique().buffer_allocation.value().box == box_cast<3>(box<1>(0, 256)));

	const auto resize_copy_iq = ictx.query<copy_instruction_record>().check_count(1);
	CHECK(resize_copy_iq.unique().box == box_cast<3>(box<1>(0, 64)));
	const auto resize_copy_pred_iq = resize_copy_iq.predecessors();
	CHECK(resize_copy_pred_iq.count() == 2);
	CHECK(resize_copy_pred_iq.select<launch_instruction_record>() == first_writer_iq);
	CHECK(resize_copy_pred_iq.select<alloc_instruction_record>() == second_alloc_iq);
}

TEST_CASE("communication-free dataflow", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::one_to_one()).submit();
}

TEST_CASE("communication-free dataflow with copies", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("simple communication", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("large graph", "[instruction_graph_generator][instruction-graph]") {
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
	const auto [num_nodes, my_nid] = GENERATE(values<std::pair<size_t, node_id>>({{1, 0}, {2, 0}, {2, 1}}));
	CAPTURE(num_nodes);
	CAPTURE(my_nid);

	test_utils::idag_test_context ictx(num_nodes, my_nid, 1 /* devices */);
	const std::string input_str = "Ifmmp!Xpsme\"\x01";
	auto buf = ictx.create_buffer<1>(input_str.size(), true /* host initialized */);

	ictx.device_compute(buf.get_range()).read_write(buf, acc::one_to_one()).submit();
	ictx.fence(buf);
}

TEST_CASE("matmul pattern", "[instruction_graph_generator][instruction-graph]") {
	const size_t mat_size = 128;

	const auto my_nid = GENERATE(values<node_id>({0, 1}));
	CAPTURE(my_nid);

	test_utils::idag_test_context ictx(4 /* nodes */, my_nid, 1 /* devices */);

	const auto range = celerity::range<2>(mat_size, mat_size);
	auto mat_a_buf = ictx.create_buffer(range);
	auto mat_b_buf = ictx.create_buffer(range);
	auto mat_c_buf = ictx.create_buffer(range);

	const auto set_identity = [&](test_utils::mock_buffer<2> mat) {
		ictx.device_compute<class set_identity>(mat.get_range()) //
		    .discard_write(mat, celerity::access::one_to_one())
		    .submit();
	};

	set_identity(mat_a_buf);
	set_identity(mat_b_buf);

	const auto multiply = [&](test_utils::mock_buffer<2> mat_a, test_utils::mock_buffer<2> mat_b, test_utils::mock_buffer<2> mat_c) {
		const size_t group_size = 8;
		ictx.device_compute<class multiply>(celerity::nd_range<2>{range, {group_size, group_size}})
		    .read(mat_a, celerity::access::slice<2>(1))
		    .read(mat_b, celerity::access::slice<2>(0))
		    .discard_write(mat_c, celerity::access::one_to_one())
		    .submit();
	};

	multiply(mat_a_buf, mat_b_buf, mat_c_buf);
	multiply(mat_b_buf, mat_c_buf, mat_a_buf);

	const auto verify = [&](test_utils::mock_buffer<2> mat_c, test_utils::mock_host_object passed_obj) {
		ictx.host_task(mat_c.get_range()) //
		    .read(mat_c, celerity::access::one_to_one())
		    .affect(passed_obj)
		    .submit();
	};

	auto passed_obj = ictx.create_host_object();
	verify(mat_a_buf, passed_obj);

	ictx.fence(passed_obj);
}

TEST_CASE("await-push of disconnected subregions does not allocate their bounding-box", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);

	auto buf = ictx.create_buffer(range(1024));
	const auto acc_first = acc::fixed(subrange<1>(0, 1));
	const auto acc_last = acc::fixed(subrange<1>(1023, 1));
	ictx.device_compute<class writer_1>(range(1)).discard_write(buf, acc_first).submit();
	ictx.device_compute<class writer_2>(range(1)).discard_write(buf, acc_last).submit();
	ictx.device_compute<class reader>(buf.get_range()).read(buf, acc_first).read(buf, acc_last).submit();
}

TEST_CASE("collective host tasks", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.collective_host_task(default_collective_group).submit();
	ictx.collective_host_task(collective_group()).submit();
}

TEST_CASE("syncing pattern", "[instruction-graph]") {
	const auto my_nid = GENERATE(values<node_id>({0, 1}));
	CAPTURE(my_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, my_nid, 2 /* devices */);

	auto buf = ictx.create_buffer<1>(512);
	ictx.device_compute(buf.get_range()).discard_write(buf, acc::one_to_one()).submit();
	ictx.collective_host_task().read(buf, acc::all()).submit();
	ictx.epoch(epoch_action::barrier);
}

TEST_CASE("allreduce from identity", "[instruction-graph]") {
	const auto num_nodes = GENERATE(values<size_t>({1, 2}));
	const auto num_devices = GENERATE(values<size_t>({1, 2}));
	CAPTURE(num_nodes, num_devices);

	test_utils::idag_test_context ictx(num_nodes, 0, num_devices);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range<1>(256)).reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range<1>(256)).read(buf, acc::all()).submit();
}

TEST_CASE("allreduce including current buffer value", "[instruction-graph]") {
	const auto num_nodes = GENERATE(values<size_t>({1, 2}));
	const auto my_nid = GENERATE(values<node_id>({0, 1}));
	const auto num_devices = GENERATE(values<size_t>({1, 2}));
	if(my_nid >= num_nodes) return;
	CAPTURE(num_nodes, my_nid, num_devices);

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);

	auto buf = ictx.create_buffer<1>(1, true /* host initialized */);
	ictx.device_compute(range<1>(256)).reduce(buf, true /* include_current_buffer_value */).submit();
	ictx.device_compute(range<1>(256)).read(buf, acc::all()).submit();
}

TEST_CASE("reduction example pattern", "[instruction-graph]") {
	const auto num_nodes = GENERATE(values<size_t>({1, 2}));
	const auto my_nid = GENERATE(values<node_id>({0, 1}));
	const auto num_devices = GENERATE(values<size_t>({1, 2}));
	if(my_nid >= num_nodes) return;
	CAPTURE(num_nodes, my_nid, num_devices);

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);

	const celerity::range image_size{1536, 2048};
	auto srgb_255_buf = ictx.create_buffer<sycl::uchar4, 2>(image_size, true /* host initialized */);
	auto rgb_buf = ictx.create_buffer<sycl::float4, 2>(image_size);
	auto min_buf = ictx.create_buffer<float, 0>({});
	auto max_buf = ictx.create_buffer<float, 0>({});

	ictx.device_compute<class linearize_and_accumulate>(image_size)
	    .read(srgb_255_buf, acc::one_to_one())
	    .discard_write(rgb_buf, acc::one_to_one())
	    .reduce(min_buf, false)
	    .reduce(max_buf, false)
	    .submit();

	ictx.master_node_host_task() //
	    .read(min_buf, acc::all())
	    .read(max_buf, acc::all())
	    .submit();

	ictx.device_compute<class correct_and_compress>(image_size)
	    .read(rgb_buf, acc::one_to_one())
	    .read(min_buf, acc::all())
	    .read(max_buf, acc::all())
	    .discard_write(srgb_255_buf, acc::one_to_one())
	    .submit();

	ictx.master_node_host_task() //
	    .read(srgb_255_buf, acc::all())
	    .submit();
}

TEST_CASE("local reduction can be initialized to a buffer value that is not present locally", "[instruction-graph]") {
	const size_t num_nodes = 2;
	const node_id my_nid = 0;
	const auto num_devices = 2;

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);

	auto buf = ictx.create_buffer(range<1>(1));

	ictx.device_compute(range<1>(num_nodes)) //
	    .discard_write(buf, [](const chunk<1> ck) { return subrange(id(0), range(ck.offset[0] + ck.range[0] > 1 ? 1 : 0)); })
	    .submit();
	ictx.device_compute(range<1>(1)) //
	    .reduce(buf, true /* include_current_buffer_value */)
	    .submit();
}

TEST_CASE("local reductions only include values from participating devices", "[instruction-graph]") {
	const size_t num_nodes = 1;
	const node_id my_nid = 0;
	const auto num_devices = 4;

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);

	auto buf = ictx.create_buffer(range<1>(1));

	ictx.device_compute(range<1>(num_devices / 2)) //
	    .reduce(buf, false /* include_current_buffer_value */)
	    .submit();
}

TEST_CASE("global reduction without a local contribution does not read a stale local value", "[instruction-graph]") {
	const size_t num_nodes = 3;
	const node_id my_nid = GENERATE(values<node_id>({0, 1, 2}));
	const auto num_devices = 1;

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);

	auto buf = ictx.create_buffer(range<1>(1));

	ictx.device_compute(range<1>(2)) //
	    .reduce(buf, false /* include_current_buffer_value */)
	    .submit();
	ictx.device_compute(range<1>(num_nodes)) //
	    .read(buf, acc::all())
	    .submit();
}

TEST_CASE("instruction_graph_generator throws in tests if it detects an uninitialized read", "[instruction_graph_generator]") {
	const size_t num_devices = 2;
	const range<1> device_range{num_devices};

	test_utils::idag_test_context ictx(1, 0, num_devices);
	ictx.get_task_manager().set_uninitialized_read_policy(error_policy::ignore);    // otherwise we get task-level errors first
	ictx.get_graph_generator().set_uninitialized_read_policy(error_policy::ignore); // otherwise we get command-level errors first

	SECTION("on a fully uninitialized buffer") {
		auto buf = ictx.create_buffer<1>({1});
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::all()).submit()),
		    "Instruction is trying to read B0 {[0,0,0] - [1,1,1]}, which is neither found locally nor has been await-pushed before.");
	}

	SECTION("on a partially, locally initialized buffer") {
		auto buf = ictx.create_buffer<1>(device_range);
		ictx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::all()).submit()),
		    "Instruction is trying to read B0 {[1,0,0] - [2,1,1]}, which is neither found locally nor has been await-pushed before.");
	}

	SECTION("on a partially, remotely initialized buffer") {
		auto buf = ictx.create_buffer<1>(device_range);
		ictx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::one_to_one()).submit()),
		    "Instruction is trying to read B0 {[1,0,0] - [2,1,1]}, which is neither found locally nor has been await-pushed before.");
	}
}

TEST_CASE("distributed_graph_generator throws in tests if it detects overlapping writes", "[instruction_graph_generator]") {
	const size_t num_devices = 2;
	test_utils::idag_test_context ictx(1, 0, num_devices);
	auto buf = ictx.create_buffer<2>({20, 20});

	SECTION("on all-write") {
		CHECK_THROWS_WITH((ictx.device_compute(buf.get_range()).discard_write(buf, acc::all()).submit()),
		    "Task T1 \"celerity::detail::unnamed_kernel\" has overlapping writes on N0 in B0 {[0,0,0] - [20,20,1]}. Choose a non-overlapping range mapper for "
		    "the write access or constrain the split to make the access non-overlapping.");
	}

	SECTION("on neighborhood-write") {
		CHECK_THROWS_WITH((ictx.device_compute(buf.get_range()).discard_write(buf, acc::neighborhood(1, 1)).submit()),
		    "Task T1 \"celerity::detail::unnamed_kernel\" has overlapping writes on N0 in B0 {[9,0,0] - [11,20,1]}. Choose a non-overlapping range mapper for "
		    "the write access or constrain the split to make the access non-overlapping.");
	}
}


// TODO a test with impossible requirements (overlapping writes maybe?)
// TODO an oversubscribed host task with side effects
