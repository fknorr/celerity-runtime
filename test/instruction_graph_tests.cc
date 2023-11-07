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


struct reverse_one_to_one {
	template <int Dims>
	subrange<Dims> operator()(chunk<Dims> ck) const {
		subrange<Dims> sr;
		for(int d = 0; d < Dims; ++d) {
			sr.offset[d] = ck.global_size[d] - ck.range[d] - ck.offset[d];
			sr.range[d] = ck.range[d];
		}
		return sr;
	}
};


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

TEMPLATE_TEST_CASE_SIG(
    "allocations and kernels are split between devices", "[instruction_graph_generator][instruction-graph][allocation]", ((int Dims), Dims), 1, 2, 3) //
{
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256}); // dim0 split
	auto buf1 = ictx.create_buffer(full_range);
	ictx.device_compute(full_range).name("writer").discard_write(buf1, acc::one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	CHECK(all_instrs.select_all<alloc_instruction_record>().predecessors().all_match<epoch_instruction_record>());

	// we have two writer instructions, one per device, each operating on their separate allocations on separate memories.
	const auto all_writers = all_instrs.select_all<launch_instruction_record>();
	CHECK(all_writers.count() == 2);
	CHECK(all_writers.all_match("writer"));
	CHECK(all_writers[0]->device_id != all_writers[1]->device_id);
	CHECK(
	    region_union(box(all_writers[0]->execution_range), box(all_writers[1]->execution_range)) == region(box(subrange<3>(zeros, range_cast<3>(full_range)))));

	for(const auto& writer : all_writers.iterate()) {
		CAPTURE(writer);
		REQUIRE(writer->access_map.size() == 1);

		// instruction_graph_generator guarantees the default dim0 split
		CHECK(writer->execution_range.range == range_cast<3>(half_range));
		CHECK((writer->execution_range.offset[0] == 0 || writer->execution_range.offset[0] == half_range[0]));

		// the IDAG allocates appropriate boxes on the memories native to each executing device.
		const auto alloc = writer.predecessors().assert_unique<alloc_instruction_record>();
		CHECK(alloc->memory_id == ictx.get_native_memory(writer->device_id.value()));
		CHECK(writer->access_map.front().allocation_id == alloc->allocation_id);
		CHECK(alloc->buffer_allocation.value().box == writer->access_map.front().accessed_box_in_buffer);

		const auto free = writer.successors().assert_unique<free_instruction_record>();
		CHECK(free->memory_id == alloc->memory_id);
		CHECK(free->allocation_id == alloc->allocation_id);
		CHECK(free->size == alloc->size);
	}

	CHECK(all_instrs.select_all<free_instruction_record>().successors().all_match<epoch_instruction_record>());
}

// This test may fail in the future if we implement a more sophisticated allocator that decides to merge some allocations.
// When this happens, consider replacing the subranges with a pair that is never reasonable to allocate the bounding-box of.
TEMPLATE_TEST_CASE_SIG("accessing non-overlapping buffer subranges in subsequent kernels causes distinct allocations",
    "[instrution_graph_generator][instruction-graph][allocation]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 128, 128});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(id(half_range), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer(full_range);
	ictx.device_compute(first_half.range, first_half.offset).name("1st").discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute(second_half.range, second_half.offset).name("2nd").discard_write(buf1, acc::one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	CHECK(all_instrs.select_all<alloc_instruction_record>().count() == 2);
	CHECK(all_instrs.select_all<copy_instruction_record>().count() == 0); // no coherence copies needed
	CHECK(all_instrs.select_all<launch_instruction_record>().count() == 2);
	CHECK(all_instrs.select_all<free_instruction_record>().count() == 2);

	const auto first = all_instrs.select_unique<launch_instruction_record>("1st");
	const auto second = all_instrs.select_unique<launch_instruction_record>("2nd");
	REQUIRE(first->access_map.size() == 1);
	REQUIRE(second->access_map.size() == 1);

	// the kernels access distinct allocations
	CHECK(first->access_map.front().allocation_id != second->access_map.front().allocation_id);
	CHECK(first->access_map.front().accessed_box_in_buffer == first->access_map.front().allocated_box_in_buffer);

	// the allocations exactly match the accessed subrange
	CHECK(second->access_map.front().accessed_box_in_buffer == second->access_map.front().allocated_box_in_buffer);

	// kernels are fully concurrent
	CHECK(first.predecessors().is_unique<alloc_instruction_record>());
	CHECK(first.successors().is_unique<free_instruction_record>());
	CHECK(second.predecessors().is_unique<alloc_instruction_record>());
	CHECK(second.successors().is_unique<free_instruction_record>());
}

TEST_CASE("resizing a buffer allocation for a discard-write access preserves only the non-overwritten parts", //
    "[instruction_graph_generator][instruction-graph][allocation]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer(range<1>(256));
	ictx.device_compute(range<1>(1)).name("1st writer").discard_write(buf1, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range<1>(1)).name("2nd writer").discard_write(buf1, acc::fixed<1>({64, 196})).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// part of the buffer is allocated for the first writer
	const auto first_writer = all_instrs.select_unique<launch_instruction_record>("1st writer");
	REQUIRE(first_writer->access_map.size() == 1);
	const auto first_alloc = first_writer.predecessors().select_unique<alloc_instruction_record>();
	const auto first_write_box = first_writer->access_map.front().accessed_box_in_buffer;
	CHECK(first_alloc->buffer_allocation.value().box == first_write_box);

	// first and second writer ranges overlap, so the bounding box has to be allocated (and the old allocation freed)
	const auto second_writer = all_instrs.select_unique<launch_instruction_record>("2nd writer");
	REQUIRE(second_writer->access_map.size() == 1);
	const auto second_alloc = second_writer.predecessors().assert_unique<alloc_instruction_record>();
	const auto second_write_box = second_writer->access_map.front().accessed_box_in_buffer;
	const auto large_alloc_box = bounding_box(first_write_box, second_write_box);
	CHECK(second_alloc->buffer_allocation.value().box == large_alloc_box);

	// The copy must not attempt to preserve ranges that were not written in the old allocation ([128] - [256]) or that were written but are going to be
	// overwritten (without being read) in the command for which the resize was generated ([64] - [128]).
	const auto preserved_region = region_difference(first_write_box, second_write_box);
	REQUIRE(preserved_region.get_boxes().size() == 1);
	const auto preserved_box = preserved_region.get_boxes().front();

	const auto resize_copy = all_instrs.select_unique<copy_instruction_record>();
	CHECK(resize_copy->box == preserved_box);

	const auto resize_copy_preds = resize_copy.predecessors();
	CHECK(resize_copy_preds.count() == 2);
	CHECK(resize_copy_preds.select_unique<launch_instruction_record>() == first_writer);
	CHECK(resize_copy_preds.select_unique<alloc_instruction_record>() == second_alloc);

	// resize-copy and overwriting kernel are concurrent, because they access non-overlapping regions in the same allocation
	CHECK(second_writer.successors().all_match<free_instruction_record>());
	CHECK(resize_copy.successors().all_match<free_instruction_record>());
}

TEST_CASE("data-dependencies are generated between kernels on the same memory", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer<1>(256);
	auto buf2 = ictx.create_buffer<1>(256);
	ictx.device_compute(range(1)).name("write buf1").discard_write(buf1, acc::all()).submit();
	ictx.device_compute(range(1)).name("overwrite buf1 right").discard_write(buf1, acc::fixed<1>({128, 128})).submit();
	ictx.device_compute(range(1)).name("read buf 1, write buf2").read(buf1, acc::all()).discard_write(buf2, acc::all()).submit();
	ictx.device_compute(range(1)).name("read-write buf1 center").read_write(buf1, acc::fixed<1>({64, 128})).submit();
	ictx.device_compute(range(1)).name("read buf2").read(buf2, acc::all()).submit();
	ictx.device_compute(range(1)).name("read buf1+2").read(buf1, acc::all()).read(buf2, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto predecessor_kernels = [](const auto& q) { return q.predecessors().template select_all<launch_instruction_record>(); };
	const auto successor_kernels = [](const auto& q) { return q.successors().template select_all<launch_instruction_record>(); };

	const auto write_buf1 = all_instrs.select_unique<launch_instruction_record>("write buf1");
	CHECK(predecessor_kernels(write_buf1).count() == 0);

	const auto overwrite_buf1_right = all_instrs.select_unique<launch_instruction_record>("overwrite buf1 right");
	CHECK(predecessor_kernels(overwrite_buf1_right) == write_buf1 /* output-dependency on buf1 [128] - [256]*/);

	const auto read_buf1_write_buf2 = all_instrs.select_unique<launch_instruction_record>("read buf 1, write buf2");
	CHECK(predecessor_kernels(read_buf1_write_buf2).contains(overwrite_buf1_right /* true-dependency on buf1 [128] - [256]*/));
	// IDAG might also specify a true-dependency on "write buf1" for buf1 [0] - [128], but this is transitive

	const auto read_write_buf1_center = all_instrs.select_unique<launch_instruction_record>("read-write buf1 center");
	CHECK(predecessor_kernels(read_write_buf1_center).contains(read_buf1_write_buf2 /* anti-dependency on buf1 [64] - [192]*/));
	// IDAG might also specify true-dependencies on "write buf1" and "overwrite buf1 right", but these are transitive

	const auto read_buf2 = all_instrs.select_unique<launch_instruction_record>("read buf2");
	CHECK(predecessor_kernels(read_buf2) == read_buf1_write_buf2 /* true-dependency on buf2 [0] - [256] */);
	// This should not depend on any other kernel instructions, because none other are concerned with buf2.

	const auto read_buf1_buf2 = all_instrs.select_unique<launch_instruction_record>("read buf1+2");
	CHECK(predecessor_kernels(read_buf1_buf2).contains(read_write_buf1_center) /* true-dependency on buf1 [64] - [192] */);
	CHECK(!predecessor_kernels(read_buf1_buf2).contains(read_buf2) /* readers are concurrent */);
	// IDAG might also specify true-dependencies on "write buf1", "overwrite buf1 right", "read buf1, write_buf2", but these are transitive
}

TEST_CASE("data dependencies across memories introduce coherence copies", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf = ictx.create_buffer(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(test_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<launch_instruction_record>("writer");
	const auto all_readers = all_instrs.select_all<launch_instruction_record>("reader");
	const auto coherence_copies = all_instrs.select_all<copy_instruction_record>(
	    [](const copy_instruction_record& copy) { return copy.origin == copy_instruction_record::copy_origin::coherence; });

	CHECK(all_readers.count() == 2);
	for(device_id did = 0; did < 2; ++did) {
		const device_id opposite_did = 1 - did;
		CAPTURE(did, opposite_did);

		const auto reader = all_readers.select_unique(did);
		REQUIRE(reader->access_map.size() == 1);
		const auto opposite_writer = all_writers.select_unique(opposite_did);
		REQUIRE(opposite_writer->access_map.size() == 1);

		// There is one coherence copy per reader kernel, which copies the portion written on the opposite device
		const auto coherence_copy = intersection_of(coherence_copies, reader.predecessors()).assert_unique();
		CHECK(coherence_copy->source_memory == ictx.get_native_memory(opposite_did));
		CHECK(coherence_copy->dest_memory == ictx.get_native_memory(did));
		CHECK(coherence_copy->box == opposite_writer->access_map.front().accessed_box_in_buffer);
	}

	// Coherence copies are not sequenced with respect to each other
	CHECK(coherence_copies.all_concurrent());
}

// This test should become obsolete in the future when we allow the instruction_executor to manage user memory - this feature would remove the need for both
// init_buffer_instruction and export_instruction
TEMPLATE_TEST_CASE_SIG("host-initialization eagerly copies the entire buffer from user memory to a host allocation",
    "[instruction_graph_generator][instruction-graph][allocation]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto buffer_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto buffer_box = box(subrange(id<Dims>(zeros), buffer_range));

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int, Dims>(buffer_range, true /* host_initialized */);
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto init = all_instrs.select_unique<init_buffer_instruction_record>();
	const auto alloc = init.predecessors().assert_unique<alloc_instruction_record>();
	CHECK(alloc->memory_id == host_memory_id);
	CHECK(alloc->buffer_allocation.value().box == box_cast<3>(buffer_box));
	CHECK(alloc->size == buffer_range.size() * sizeof(int));
	CHECK(init->size == alloc->size);
	CHECK(init->buffer_id == buf.get_id());
	CHECK(init->host_allocation_id == alloc->allocation_id);
}

TEMPLATE_TEST_CASE_SIG("buffer subranges are sent and received to satisfy push and await-push commands",
    "[instruction_graph_generator][instruction-graph][send-recv]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id opposite_nid = 1 - local_nid;
	CAPTURE(local_nid, opposite_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);

	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto writer = all_instrs.select_unique<launch_instruction_record>("writer");
	const auto send = all_instrs.select_unique<send_instruction_record>();
	const auto recv = all_instrs.select_unique<receive_instruction_record>();
	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	// we send exactly the part of the buffer that our node has written
	REQUIRE(writer->access_map.size() == 1);
	const auto& write_access = writer->access_map.front();
	CHECK(send->dest_node_id == opposite_nid);
	CHECK(send->transfer_id == expected_trid);
	CHECK(send->send_range == write_access.accessed_box_in_buffer.get_range());
	CHECK(send->offset_in_buffer == write_access.accessed_box_in_buffer.get_offset());
	CHECK(send->element_size == sizeof(int));

	// a pilot is attached to the send
	const auto pilot = ictx.query_outbound_pilots();
	CHECK(pilot.is_unique());
	CHECK(pilot->to == opposite_nid);
	CHECK(pilot->message.trid == send->transfer_id);
	CHECK(pilot->message.tag == send->tag);

	// we receive exactly the part of the buffer that our node has _not_ written
	REQUIRE(reader->access_map.size() == 1);
	const auto& read_access = reader->access_map.front();
	CHECK(recv->transfer_id == expected_trid);
	CHECK(recv->element_size == sizeof(int));
	CHECK(region_intersection(write_access.accessed_box_in_buffer, recv->requested_region).empty());
	CHECK(region_union(write_access.accessed_box_in_buffer, recv->requested_region) == region(read_access.accessed_box_in_buffer));

	// the logical dependencies are (writer -> send, writer -> reader, recv -> reader)
	CHECK(writer.transitive_successors_across<copy_instruction_record>().contains(send));
	CHECK(recv.transitive_successors_across<copy_instruction_record>().contains(reader));
	CHECK(send.is_concurrent_with(recv));
}

TEMPLATE_TEST_CASE_SIG("send and receive instructions are split on multi-device systems to allow compute-transfer overlap",
    "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 1, 2, 3) {
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id opposite_nid = 1 - local_nid;
	CAPTURE(local_nid, opposite_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, reverse_one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	const auto all_writers = all_instrs.select_all<launch_instruction_record>("writer");
	CHECK(all_writers.count() == 2);
	CHECK(all_writers.all_concurrent());

	const auto all_sends = all_instrs.select_all<send_instruction_record>();
	CHECK(all_sends.count() == 2);
	CHECK(all_sends.all_concurrent());

	CHECK(all_pilots.count() == all_sends.count());

	// there is one send per writer instruction (with coherence copies in between)
	for(const auto& send : all_sends.iterate()) {
		CAPTURE(send);

		const auto associated_writer =
		    intersection_of(send.transitive_predecessors_across<copy_instruction_record>(), all_writers).assert_unique<launch_instruction_record>();
		REQUIRE(associated_writer->access_map.size() == 1);
		const auto& write = associated_writer->access_map.front();

		// the send operates on a (host) allocation that is distinct from the (device) allocation that associated_writer writes to, but both instructions need
		// to access the same buffer subrange
		const auto send_box = box(subrange(send->offset_in_buffer, send->send_range));
		CHECK(send_box == write.accessed_box_in_buffer);
		CHECK(send->element_size == sizeof(int));
		CHECK(send->transfer_id == expected_trid);

		CHECK(all_pilots.count(send->dest_node_id, expected_trid, send_box) == 1);
	}

	const auto split_recv = all_instrs.select_unique<split_receive_instruction_record>();
	const auto all_await_recvs = all_instrs.select_all<await_receive_instruction_record>();
	const auto all_readers = all_instrs.select_all<launch_instruction_record>("reader");

	// There is one split-receive instruction which binds the allocation to a transfer id, because we don't know the shape / stride of incoming messages until
	// we receive pilots at runtime, and messages might either match our awaited subregions (and complete them independently), cover both (and need the
	// bounding-box allocation), or anything in between.
	CHECK(split_recv.successors().contains(all_await_recvs));
	CHECK(split_recv->requested_region == region_union(all_await_recvs[0]->received_region, all_await_recvs[1]->received_region));
	CHECK(split_recv->element_size == sizeof(int));
	CHECK(region_intersection(all_await_recvs[0]->received_region, all_await_recvs[1]->received_region).empty());

	CHECK(all_await_recvs.count() == 2); // one per device
	CHECK(all_await_recvs.all_concurrent());
	CHECK(all_readers.all_concurrent());

	// there is one reader per await-receive instruction (with coherence copies in between)
	for(const auto& await_recv : all_await_recvs.iterate()) {
		CAPTURE(await_recv);

		const auto associated_reader =
		    intersection_of(await_recv.transitive_successors_across<copy_instruction_record>(), all_readers).assert_unique<launch_instruction_record>();
		REQUIRE(associated_reader->access_map.size() == 1);
		const auto& read = associated_reader->access_map.front();

		CHECK(await_recv->received_region == region(read.accessed_box_in_buffer));
		CHECK(await_recv->transfer_id == expected_trid);
	}
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

TEMPLATE_TEST_CASE_SIG("buffer fences export data to user memory", "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 0, 1, 2, 3) {
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

TEST_CASE("host-object fences introduce the appropriate dependencies", "[instruction_graph_generator][instruction-graph]") {
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

	const auto &ho_fence_info = std::get<fence_instruction_record::host_object_variant>(fence->variant);
	CHECK(ho_fence_info.hoid == ho.get_id());
}

TEST_CASE("transitive copy dependencies", "[instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);

	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	auto buf2 = ictx.create_buffer(test_range);
	ictx.device_compute(test_range).name("producer").discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute(test_range).name("gather").read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute(test_range).name("consumer").read(buf2, reverse_one_to_one()).submit();
	ictx.finish();

	const auto iq = ictx.query_instructions();
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


// TODO a test with impossible requirements (overlapping writes maybe?)
// TODO an oversubscribed host task with side effects
