#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;


TEST_CASE("a command group without data access compiles to a trivial graph", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	ictx.device_compute(test_range).name("kernel").submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count() == 3);
	CHECK(all_instrs.count<epoch_instruction_record>() == 2);
	CHECK(all_instrs.count<launch_instruction_record>() == 1);

	const auto kernel = all_instrs.select_unique<launch_instruction_record>("kernel");
	CHECK(kernel->access_map.empty());
	CHECK(kernel->execution_range == subrange<3>(zeros, range_cast<3>(test_range)));
	CHECK(kernel->device_id == device_id(0));
	CHECK(kernel.predecessors().is_unique<epoch_instruction_record>());
	CHECK(kernel.successors().is_unique<epoch_instruction_record>());
}


TEST_CASE("side-effects introduce dependencies between host-task instructions", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);
	auto ho1 = ictx.create_host_object();
	auto ho2 = ictx.create_host_object(false /* owns instance */);
	ictx.master_node_host_task().name("affect ho1 (a)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (a)").affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 (b)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (b)").affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 + ho2").affect(ho1).affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 (c)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (c)").affect(ho2).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto affect_ho1_a = all_instrs.select_unique<launch_instruction_record>("affect ho1 (a)");
	const auto affect_ho2_a = all_instrs.select_unique<launch_instruction_record>("affect ho2 (a)");
	const auto affect_ho1_b = all_instrs.select_unique<launch_instruction_record>("affect ho1 (b)");
	const auto affect_ho2_b = all_instrs.select_unique<launch_instruction_record>("affect ho2 (b)");
	const auto affect_both = all_instrs.select_unique<launch_instruction_record>("affect ho1 + ho2");
	const auto affect_ho1_c = all_instrs.select_unique<launch_instruction_record>("affect ho1 (c)");
	const auto affect_ho2_c = all_instrs.select_unique<launch_instruction_record>("affect ho2 (c)");
	// only ho1 owns its instance, so only one destroy_host_object_instruction is generated
	const auto destroy_ho1 = all_instrs.select_unique<destroy_host_object_instruction_record>();

	CHECK(affect_ho1_a.predecessors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho2_a.predecessors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho1_a.successors() == affect_ho1_b);
	CHECK(affect_ho2_a.successors() == affect_ho2_b);
	CHECK(affect_ho1_b.successors() == affect_both);
	CHECK(affect_ho2_b.successors() == affect_both);
	CHECK(affect_both.successors() == union_of(affect_ho1_c, affect_ho2_c));
	CHECK(affect_ho1_c.successors() == destroy_ho1);
	CHECK(destroy_ho1.successors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho2_c.successors().is_unique<epoch_instruction_record>());
}

TEST_CASE("collective-group instructions follow a single global total order", "[instruction_graph_generator][instruction-graph]") {
	const auto local_nid = GENERATE(values<node_id>({0, 1}));

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);
	ictx.set_horizon_step(999);

	// collective-groups are not explicitly registered with graph generators, so both IDAG tests and the runtime use the same mechanism to declare them
	experimental::collective_group custom_cg_1;
	experimental::collective_group custom_cg_2;

	ictx.collective_host_task().name("default-group (a)").submit();
	ictx.collective_host_task().name("default-group (b)").submit();
	ictx.collective_host_task(custom_cg_1).name("custom-group 1 (a)").submit();
	ictx.collective_host_task(custom_cg_2).name("custom-group 2").submit();
	ictx.collective_host_task(custom_cg_1).name("custom-group 1 (b)").submit();
	ictx.collective_host_task().name("default-group (c)").submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto init_epoch = all_instrs.select_unique(task_manager::initial_epoch_task);
	// the default collective group does not use the default communicator (aka MPI_COMM_WORLD) because host tasks are executed on a different thread
	const auto clone_for_default_group = init_epoch.successors().assert_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_default_group->origin_collective_group_id == root_collective_group_id);
	CHECK(clone_for_default_group->new_collective_group_id != clone_for_default_group->origin_collective_group_id);
	const auto default_cgid = clone_for_default_group->new_collective_group_id;

	const auto default_group_a = all_instrs.select_unique<launch_instruction_record>("default-group (a)");
	CHECK(default_group_a->collective_group_id == default_cgid);
	CHECK(default_group_a.predecessors() == clone_for_default_group);

	const auto default_group_b = all_instrs.select_unique<launch_instruction_record>("default-group (b)");
	CHECK(default_group_b->collective_group_id == default_cgid);
	CHECK(default_group_b.predecessors() == default_group_a); // collective-group ordering

	// even though "default-group (c)" is submitted last, it only depends on its predecessor in the same group
	const auto default_group_c = all_instrs.select_unique<launch_instruction_record>("default-group (c)");
	CHECK(default_group_c->collective_group_id == default_cgid);
	CHECK(default_group_c.predecessors() == default_group_b); // collective-group ordering
	CHECK(default_group_c.successors().is_unique<epoch_instruction_record>());

	// clone-collective-group instructions are ordered, because cloning an MPI communicator is a collective operation as well
	const auto clone_for_custom_group_1 = clone_for_default_group.successors().select_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_custom_group_1->origin_collective_group_id == root_collective_group_id);
	CHECK(clone_for_custom_group_1->new_collective_group_id != clone_for_custom_group_1->origin_collective_group_id);
	const auto custom_cgid_1 = clone_for_custom_group_1->new_collective_group_id;
	CHECK(custom_cgid_1 != default_cgid);

	const auto custom_group_1_a = all_instrs.select_unique<launch_instruction_record>("custom-group 1 (a)");
	CHECK(custom_group_1_a->collective_group_id == custom_cgid_1);
	CHECK(custom_group_1_a.predecessors() == clone_for_custom_group_1);

	const auto custom_group_1_b = all_instrs.select_unique<launch_instruction_record>("custom-group 1 (b)");
	CHECK(custom_group_1_b->collective_group_id == custom_cgid_1);
	CHECK(custom_group_1_b.predecessors() == custom_group_1_a); // collective-group ordering
	CHECK(custom_group_1_b.successors().is_unique<epoch_instruction_record>());

	// clone-collective-group instructions are ordered, because cloning an MPI communicator is a collective operation as well
	const auto clone_for_custom_group_2 = clone_for_custom_group_1.successors().select_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_custom_group_2->origin_collective_group_id == root_collective_group_id);
	CHECK(clone_for_custom_group_2->new_collective_group_id != clone_for_custom_group_2->origin_collective_group_id);
	const auto custom_cgid_2 = clone_for_custom_group_2->new_collective_group_id;
	CHECK(custom_cgid_2 != default_cgid);
	CHECK(custom_cgid_2 != custom_cgid_1);

	const auto custom_group_2 = all_instrs.select_unique<launch_instruction_record>("custom-group 2");
	CHECK(custom_group_2->collective_group_id == custom_cgid_2);
	CHECK(custom_group_2.predecessors() == clone_for_custom_group_2);
	CHECK(custom_group_2.successors().is_unique<epoch_instruction_record>());
}

TEMPLATE_TEST_CASE_SIG("buffer fences export data to user memory", "[instruction_graph_generator][instruction-graph][fence]", ((int Dims), Dims), 0, 1, 2, 3) {
	constexpr static auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto export_subrange = GENERATE(values({subrange<Dims>({}, full_range), test_utils::truncate_subrange<Dims>({{48, 64, 72}, {128, 128, 128}})}));

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int>(full_range);
	ictx.device_compute(full_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.fence(buf, export_subrange);
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto fence = all_instrs.select_unique<fence_instruction_record>();
	const auto& fence_buffer_info = std::get<fence_instruction_record::buffer_variant>(fence->variant);
	CHECK(fence_buffer_info.bid == buf.get_id());
	CHECK(fence_buffer_info.box.get_offset() == id_cast<3>(export_subrange.offset));
	CHECK(fence_buffer_info.box.get_range() == range_cast<3>(export_subrange.range));

	// At some point in the future we want to track user-memory allocations in the executor, and emit normal copy-instructions from host-memory instead of
	// these specialized export instructions.
	const auto xport = fence.predecessors().assert_unique<export_instruction_record>();
	CHECK(xport->buffer == buf.get_id());
	CHECK(xport->dimensions == Dims);
	CHECK(xport->offset_in_buffer == id_cast<3>(export_subrange.offset));
	CHECK(xport->copy_range == range_cast<3>(export_subrange.range));
	CHECK(xport->element_size == sizeof(int));

	// exports are done from host memory
	const auto coherence_copy = xport.predecessors().assert_unique<copy_instruction_record>();
	CHECK(coherence_copy->copy_range == xport->copy_range);
	CHECK(coherence_copy->buffer == buf.get_id());
	CHECK(coherence_copy->dimensions == Dims);
	CHECK(coherence_copy->box.get_offset() == xport->offset_in_buffer);
	CHECK(coherence_copy->box.get_range() == xport->copy_range);
}

TEST_CASE("host-object fences introduce the appropriate dependencies", "[instruction_graph_generator][instruction-graph][fence]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto ho = ictx.create_host_object();
	ictx.master_node_host_task().name("task 1").affect(ho).submit();
	ictx.fence(ho);
	ictx.master_node_host_task().name("task 2").affect(ho).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto task_1 = all_instrs.select_unique<launch_instruction_record>("task 1");
	const auto fence = all_instrs.select_unique<fence_instruction_record>();
	const auto task_2 = all_instrs.select_unique<launch_instruction_record>("task 2");

	CHECK(task_1.successors() == fence);
	CHECK(fence.successors() == task_2);

	const auto& ho_fence_info = std::get<fence_instruction_record::host_object_variant>(fence->variant);
	CHECK(ho_fence_info.hoid == ho.get_id());
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

TEST_CASE("instruction_graph_generator throws in tests if it detects overlapping writes", "[instruction_graph_generator]") {
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
