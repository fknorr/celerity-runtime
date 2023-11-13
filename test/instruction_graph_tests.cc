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

template <int Dims>
acc::neighborhood<Dims> make_neighborhood(const size_t border) {
	if constexpr(Dims == 1) {
		return acc::neighborhood<1>(border);
	} else if constexpr(Dims == 2) {
		return acc::neighborhood<2>(border, border);
	} else if constexpr(Dims == 3) {
		return acc::neighborhood<3>(border, border, border);
	}
}


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

TEMPLATE_TEST_CASE_SIG("multiple overlapping accessors cause allocation of their bounding box", "[instruction_graph_generator][instruction-graph][allocation]",
    ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto access_range = test_utils::truncate_range<Dims>({128, 128, 128});
	const auto access_offset_1 = test_utils::truncate_id<Dims>({32, 32, 32});
	const auto access_offset_2 = test_utils::truncate_id<Dims>({96, 96, 96});

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(access_range)
	    .discard_write(buf, [=](const chunk<Dims>& ck) { return subrange(ck.offset + access_offset_1, ck.range); })
	    .discard_write(buf, [=](const chunk<Dims>& ck) { return subrange(ck.offset + access_offset_2, ck.range); })
	    .submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto kernel = all_instrs.select_unique<launch_instruction_record>();

	const auto alloc = all_instrs.select_unique<alloc_instruction_record>();
	CHECK(alloc->memory_id == ictx.get_native_memory(kernel->device_id.value()));
	CHECK(alloc->buffer_allocation->buffer_id == buf.get_id());

	// the IDAG must allocate the bounding box for both accessors to map to overlapping, contiguous memory
	const auto expected_box = bounding_box(box(subrange(access_offset_1, access_range)), box(subrange(access_offset_2, access_range)));
	CHECK(alloc->buffer_allocation->box == box_cast<3>(expected_box));
	CHECK(alloc->size == expected_box.get_area() * sizeof(int));
	CHECK(alloc.successors().contains(kernel));

	// alloc and free instructions are always symmetric
	const auto free = all_instrs.select_unique<free_instruction_record>();
	CHECK(free->memory_id == alloc->memory_id);
	CHECK(free->allocation_id == alloc->allocation_id);
	CHECK(free->size == alloc->size);
	CHECK(free->buffer_allocation == alloc->buffer_allocation);
	CHECK(free.predecessors().contains(kernel));
}

TEMPLATE_TEST_CASE_SIG(
    "allocations and kernels are split between devices", "[instruction_graph_generator][instruction-graph][allocation]", ((int Dims), Dims), 1, 2, 3) //
{
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256}); // dim0 split
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(full_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
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
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(first_half.range, first_half.offset).name("1st").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(second_half.range, second_half.offset).name("2nd").discard_write(buf, acc::one_to_one()).submit();
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
	auto buf = ictx.create_buffer(range<1>(256));
	ictx.device_compute(range<1>(1)).name("1st writer").discard_write(buf, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range<1>(1)).name("2nd writer").discard_write(buf, acc::fixed<1>({64, 196})).submit();
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

TEMPLATE_TEST_CASE_SIG(
    "coherence copies of the same data are performed only once", "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(test_utils::truncate_id<Dims>({128, 0, 0}), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);
	auto buf = ictx.create_buffer<int>(full_range, true /* host-initialize to avoid resizes */);

	ictx.device_compute(range(1)).name("write 1st half").discard_write(buf, acc::fixed(first_half)).submit();
	// requires a coherence copy for the first half
	ictx.master_node_host_task().name("read 1st half").read(buf, acc::fixed(first_half)).submit();
	ictx.master_node_host_task().name("read 1st half").read(buf, acc::fixed(first_half)).submit();
	ictx.device_compute(range(1)).name("write 2nd half").discard_write(buf, acc::fixed(second_half)).submit();
	// requires a coherence copy for the second half
	ictx.master_node_host_task().name("read all").read(buf, acc::all()).submit();
	ictx.master_node_host_task().name("read all").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto write_first_half = all_instrs.select_all<launch_instruction_record>("write 1st half");
	const auto read_first_half = all_instrs.select_all<launch_instruction_record>("read 1st half");
	const auto write_second_half = all_instrs.select_all<launch_instruction_record>("write 2nd half");
	const auto read_all = all_instrs.select_all<launch_instruction_record>("read all");
	const auto all_copies = all_instrs.select_all<copy_instruction_record>();

	// there is one device -> host copy for each half
	CHECK(all_copies.count() == 2);
	const auto first_half_copy = intersection_of(all_copies, write_first_half.successors()).assert_unique();
	CHECK(first_half_copy->box == box_cast<3>(box(first_half)));
	const auto second_half_copy = intersection_of(all_copies, write_second_half.successors()).assert_unique();
	CHECK(second_half_copy->box == box_cast<3>(box(second_half)));

	// copies depend on their producers only
	CHECK(first_half_copy.predecessors().contains(write_first_half));
	CHECK(first_half_copy.successors().contains(read_first_half)); // both readers of 1st half
	CHECK(first_half_copy.successors().contains(read_all));        // both readers of full range

	CHECK(second_half_copy.predecessors().contains(write_second_half));
	CHECK(second_half_copy.successors().contains(read_all)); // both readers of full range
}

TEMPLATE_TEST_CASE_SIG(
    "local copies are split to allow overlap between compute and copy", "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(test_utils::truncate_id<Dims>({128, 0, 0}), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int>(full_range);
	// both writers write to the same memory, initializing the full_range on D0 / M1
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::fixed(first_half)).submit();
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::fixed(second_half)).submit();
	// read on host / M0 must create one copy per writer to allow compute-copy overlap
	ictx.master_node_host_task().name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<launch_instruction_record>("writer");
	const auto all_copies = all_instrs.select_all<copy_instruction_record>();
	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");

	CHECK(all_writers.count() == 2);
	CHECK(all_copies.count() == 2);

	// the reader depends on one copy per writer
	for(const auto& copy : all_copies.iterate()) {
		const auto writer = intersection_of(copy.predecessors(), all_writers).assert_unique<launch_instruction_record>();
		REQUIRE(writer->access_map.size() == 1);

		const auto& write = writer->access_map.front();
		CHECK(write.buffer_id == copy->buffer);
		CHECK(write.accessed_box_in_buffer == copy->box);
		copy.successors().contains(reader);
	}
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

TEMPLATE_TEST_CASE_SIG("overlapping requirements generate split-receives with one await per reader-set", "[instruction_graph_generator][instruction-graph]",
    ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 4 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::all()).submit();
	ictx.device_compute(test_range).name("reader").read(buf, make_neighborhood<Dims>(1)).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// We are N1, so we receive the entire buffer from N0.
	const auto split_recv = all_instrs.select_unique<split_receive_instruction_record>();

	// We do not know the send-split, so we create a receive-split that awaits subregions used by single chunks separately from subregions used by multiple
	// chunks (the neighborhood overlap) in order to un-block any instruction as soon as their requirements are fulfilled.
	const auto all_await_recvs = all_instrs.select_all<await_receive_instruction_record>();
	CHECK(split_recv.successors().contains(all_await_recvs));

	// await-receives for the same split-receive must never intersect, but their union must cover the entire received region
	region<3> awaited_region;
	for(const auto& await : all_await_recvs.iterate()) {
		CHECK(region_intersection(awaited_region, await->received_region).empty());
		awaited_region = region_union(awaited_region, await->received_region);
	}
	CHECK(awaited_region == split_recv->requested_region);

	// Each reader instruction sources its input data from multiple await-receive instructions, and by extension, the await-receives operating on the overlap
	// service multiple reader chunks.
	const auto all_readers = all_instrs.select_all<launch_instruction_record>("reader");
	for(const auto& reader : all_readers.iterate()) {
		const auto all_pred_awaits = reader.transitive_predecessors_across<copy_instruction_record>().select_all<await_receive_instruction_record>();
		CHECK(all_pred_awaits.count() > 1);

		// Sanity check: Each reader chunk depends on await-receives for the subranges it reads
		region<3> pred_awaited_region;
		for(const auto& pred_await : all_pred_awaits.iterate()) {
			pred_awaited_region = region_union(pred_awaited_region, pred_await->received_region);
		}

		REQUIRE(reader->access_map.size() == 1);
		const auto& read = reader->access_map.front();
		CHECK(read.buffer_id == buf.get_id());
		CHECK(read.accessed_box_in_buffer == pred_awaited_region);
	}
}

TEST_CASE("an await-push of disconnected subregions does not allocate their bounding-box", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(range(1024));
	const auto acc_first = acc::fixed(subrange<1>(0, 1));
	const auto acc_last = acc::fixed(subrange<1>(1023, 1));
	ictx.host_task(range(1)).name("writer").discard_write(buf, acc::all()).submit(); // remote only
	ictx.host_task(range(2)).name("reader").read(buf, acc_first).read(buf, acc_last).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// since the individual elements (acc_first, acc_last) are bound to different accessors, we can (and should) allocate them separately to avoid allocating
	// the large bounding box. This means we have two allocations, with one receive each.
	const auto all_allocs = all_instrs.select_all<alloc_instruction_record>();
	CHECK(all_allocs.count() == 2);
	const auto all_recvs = all_instrs.select_all<receive_instruction_record>();
	CHECK(all_recvs.count() == 2);
	CHECK(all_recvs.all_concurrent());

	for(const auto& recv : all_recvs.iterate()) {
		CAPTURE(recv);
		CHECK(recv->requested_region.get_area() == 1);

		const auto alloc = recv.predecessors().select_unique<alloc_instruction_record>();
		CHECK(alloc->buffer_allocation->buffer_id == buf.get_id());
		CHECK(region(alloc->buffer_allocation->box) == recv->requested_region);
	}

	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");
	CHECK(reader.predecessors() == all_recvs);
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

TEST_CASE("reductions are equivalent to writes on a single-node single-device setup", "[instruction_graph_generator][instruction-graph][reduction]") {
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto writer = all_instrs.select_unique<launch_instruction_record>("writer");
	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");
	CHECK(writer.successors() == reader);
	CHECK(reader.predecessors() == writer);

	// there is no local (eager) reduce-instruction generated on the reduction-write, nor do we get a reduction_command to generate a global (lazy)
	// reduce-instruction between nodes.
	CHECK(all_instrs.count<send_instruction_record>() == 0);
	CHECK(all_instrs.count<gather_receive_instruction_record>() == 0);
	CHECK(all_instrs.count<copy_instruction_record>() == 0);
	CHECK(all_instrs.count<reduce_instruction_record>() == 0);
}

TEST_CASE("single-node single-device reductions locally include the initial buffer value"
          "[instruction_graph_generator][instruction-graph][reduction]") //
{
	// almost the same setup as above, but now we include the current buffer value, which generates a local reduce-instruction.
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<1>(1, true /* host initialized */);
	ictx.device_compute(range(256)).name("writer").reduce(buf, true /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto writer = all_instrs.select_unique<launch_instruction_record>("writer");
	const auto init_buffer = all_instrs.select_unique<init_buffer_instruction_record>();
	CHECK(writer.is_concurrent_with(init_buffer));

	// initialization happens in a buffer allocation, from which we must copy into the gather allocation
	const auto gather_from_init = init_buffer.successors().assert_unique<copy_instruction_record>();
	CHECK(gather_from_init->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_from_init->box == box<3>(zeros, ones));

	// we also directly perfrom a device-to-host copy into the gather allocation
	const auto gather_from_writer = writer.successors().assert_unique<copy_instruction_record>();
	CHECK(gather_from_writer->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_from_writer->box == box<3>(zeros, ones));
	CHECK(gather_from_writer.is_concurrent_with(gather_from_init));

	const auto gather_alloc = intersection_of(gather_from_init.predecessors(), gather_from_writer.predecessors()).assert_unique<alloc_instruction_record>();
	CHECK(gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(gather_alloc->size == 2 * sizeof(int)); // 1 for the initial value, 1 for the "writer" contribution

	// the local reduction combines both values and writes into the (final) host buffer allocation
	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(local_reduce->scope == reduce_instruction_record::reduction_scope::local);
	CHECK(local_reduce->num_source_values == 2);
	CHECK(local_reduce->memory_id == gather_from_init->dest_memory);
	CHECK(local_reduce->source_allocation_id == gather_from_init->dest_allocation);
	CHECK(local_reduce->memory_id == gather_from_writer->dest_memory);
	CHECK(local_reduce->source_allocation_id == gather_from_writer->dest_allocation);

	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");
	CHECK(reader.transitive_predecessors_across<copy_instruction_record>().contains(local_reduce));
}

TEST_CASE("reduction accesses on a single-node multi-device setup generate local reduce-instructions only",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const size_t num_devices = 2;
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// partial results are written on the device
	const auto all_writers = all_instrs.select_all<launch_instruction_record>("writer");
	CHECK(all_writers.count() == num_devices);
	CHECK(all_writers.all_concurrent());

	// partial results are written to the appropriate positions in a (host) gather buffer
	const auto all_gather_copies = all_writers.successors().select_all<copy_instruction_record>();
	CHECK(all_gather_copies.count() == num_devices);
	CHECK(all_gather_copies.all_concurrent());
	for(const auto& gather_copy : all_gather_copies.iterate()) {
		CAPTURE(gather_copy);
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);

		// the order of reduction inputs must be deterministic because the reduction operator is not necessarily associative
		const auto writer = intersection_of(all_writers, gather_copy.predecessors());
		CHECK(gather_copy->offset_in_dest == id_cast<3>(id(static_cast<size_t>(writer->device_id.value()))));
	}

	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(local_reduce.predecessors().contains(all_gather_copies));

	const auto all_readers = all_instrs.select_all<launch_instruction_record>("reader");
	CHECK(all_readers.all_concurrent());
	CHECK(local_reduce.transitive_successors_across<copy_instruction_record>().contains(all_readers));
}

TEST_CASE("reduction accesses on a multi-node single-device setup generate global reduce-instructions only",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const size_t num_nodes = 2;
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id peer_nid = 1 - local_nid;
	CAPTURE(local_nid, peer_nid);

	test_utils::idag_test_context ictx(num_nodes, local_nid, 1 /* num devices */);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	// there is exactly one (global) reduce-instruction per node.
	const auto reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(reduce->scope == reduce_instruction_record::reduction_scope::global);
	CHECK(reduce->num_source_values == num_nodes);
	CHECK(reduce->buffer_id == buf.get_id());
	CHECK(reduce->box == box<3>(zeros, ones));

	// we send partial results to the peer - this operation anti-depends on the reduce operation, which will overwrite its buffer
	const auto send_to_peer = all_instrs.select_unique<send_instruction_record>();
	CHECK(reduce.predecessors().contains(send_to_peer));
	CHECK(send_to_peer->offset_in_buffer == zeros);
	CHECK(send_to_peer->send_range == ones);
	CHECK(send_to_peer->dest_node_id == peer_nid);
	CHECK(send_to_peer->transfer_id.rid == reduce->reduction_id);
	CHECK(send_to_peer->transfer_id.bid == buf.get_id());

	// we (gather-) copy the local partial result to the appropriate position in the gather buffer
	const auto gather_copy = reduce.predecessors().select_unique<copy_instruction_record>();
	CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_copy->offset_in_source == zeros);
	CHECK(gather_copy->offset_in_dest == id_cast<3>(id(local_nid)));
	CHECK(gather_copy->copy_range == ones);

	// we gather-receive from all peers - this will _not_ write to the position `local_nid`
	const auto gather_recv = all_instrs.select_unique<gather_receive_instruction_record>();
	CHECK(reduce.predecessors().contains(gather_recv));
	CHECK(gather_recv->gather_box == box<3>(zeros, ones));
	CHECK(gather_recv->num_nodes == num_nodes);
	CHECK(gather_recv->allocation_id == gather_copy->dest_allocation);
	CHECK(gather_recv->transfer_id == send_to_peer->transfer_id);

	// fill the gather-buffer before initiating the gather-receive because if the peer decides to not send a payload (but an empty pilot), the gather-recv can
	// simply skip writing to the appropriate position in the gather buffer.
	const auto fill_identity = all_instrs.select_unique<fill_identity_instruction_record>();
	CHECK(fill_identity.successors().contains(union_of(gather_copy, gather_recv)));
	CHECK(fill_identity->reduction_id == reduce->reduction_id);
	CHECK(fill_identity->num_values == num_nodes);
}

TEST_CASE("reduction accesses on a multi-node multi-device setup generate global and local reduce-instructions",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const size_t num_nodes = 2;
	const size_t num_devices = 2;
	test_utils::idag_test_context ictx(num_nodes, 0 /* my nid */, num_devices);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// TODO
}

TEST_CASE("local reductions can be initialized to a buffer value that is not present locally", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 2; // we need a remote writer
	const node_id my_nid = GENERATE(values<node_id>({0, 1}));
	const auto num_devices = 1; // we generate a local reduction even for a single device because there's a remote contribution
	CAPTURE(my_nid);

	constexpr auto item_1_accesses_0 = [](const chunk<1> ck) { return subrange(id(0), range(ck.offset[0] + ck.range[0] > 1 ? 1 : 0)); };

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);
	auto buf = ictx.create_buffer<int>(range(1));
	ictx.device_compute(range<1>(num_nodes)).name("init").discard_write(buf, item_1_accesses_0).submit();
	const auto reduce_tid = ictx.device_compute(range<1>(1)).name("writer").reduce(buf, true /* include_current_buffer_value */).submit();
	// local reductions are generated eagerly, even if there is no subsequent reader
	ictx.finish();

	// the generated push / await-push pair is not in preparation of a reduction command (since there is none in this example), instead, the kernel starting the
	// reduction defines an implicit read-requirement on the buffer on the reduction-initializer node (node 0), and push / await-push commands are generated
	// accordingly to establish coherence.
	const auto expected_trid = transfer_id(reduce_tid, buf.get_id(), no_reduction_id);

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	if(my_nid == 0) {
		// we are the receiver / reducer node
		const auto recv = all_instrs.select_unique<receive_instruction_record>();
		CHECK(recv->transfer_id == expected_trid);
		CHECK(recv->requested_region == region(box<3>(zeros, ones)));
		CHECK(recv->element_size == sizeof(int));

		const auto writer = all_instrs.select_unique<launch_instruction_record>("writer");
		CHECK(writer->access_map.empty()); // we have reductions, not (regular) accesses
		REQUIRE(writer->reduction_map.size() == 1);
		const auto& red_acc = writer->reduction_map.front();
		CHECK(red_acc.buffer_id == buf.get_id());
		CHECK(red_acc.box == box<3>(zeros, ones));

		const auto gather_copies = all_instrs.select_all<copy_instruction_record>();
		for(const auto& copy : gather_copies.iterate()) {
			CHECK(copy->origin == copy_instruction_record::copy_origin::gather);
			CHECK(copy->buffer == buf.get_id());
			CHECK(copy->copy_range == ones);
			CHECK(copy->box == box<3>(zeros, ones));
		}

		const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
		CHECK(local_reduce->box == box<3>(zeros, ones));
		CHECK(local_reduce->reduction_id != no_reduction_id);
		CHECK(local_reduce->reduction_id == red_acc.reduction_id);
		CHECK(local_reduce->num_source_values == 2); // the received remote init value + our contribution
		CHECK(local_reduce.predecessors() == gather_copies);
	} else {
		// we are the initializer / sender node
		const auto send = all_instrs.select_unique<send_instruction_record>();
		CHECK(send->transfer_id == expected_trid);
		CHECK(send->send_range == range_cast<3>(range(1)));
		CHECK(send->offset_in_buffer == zeros);
		CHECK(send->element_size == sizeof(int));

		const auto pilot = all_pilots.assert_unique();
		CHECK(pilot->to == 0);
		CHECK(pilot->message.trid == expected_trid);
		CHECK(pilot->message.box == box<3>(zeros, ones));
	}
}

TEST_CASE("local reductions only include values from participating devices", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 1;
	const node_id my_nid = 0;
	const auto num_devices = 4; // we need multiple, but not all devices to produce partial reduction results

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);
	auto buf = ictx.create_buffer(range<1>(1));
	ictx.device_compute(range<1>(num_devices / 2)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<launch_instruction_record>("writer");
	CHECK(all_writers.count() == num_devices / 2);

	// look up the reduce-instruction first because it defines which memory / allocation we need to gather-copy to
	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();

	// there is one gather-copy for each writer kernel
	for(const auto& writer : all_writers.iterate()) {
		CAPTURE(writer);

		CHECK(writer->access_map.empty()); // we have reductions, not (regular) accesses
		REQUIRE(writer->reduction_map.size() == 1);
		const auto& red_acc = writer->reduction_map.front();

		const auto gather_copy = writer.successors().assert_unique<copy_instruction_record>();
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
		CHECK(gather_copy->source_memory == red_acc.memory_id);
		CHECK(gather_copy->source_allocation == red_acc.allocation_id);

		CHECK(gather_copy->dest_memory == local_reduce->memory_id);
		CHECK(gather_copy->dest_allocation == local_reduce->source_allocation_id);

		// gather-order must be deterministic because the reduction operation is not necessarily associative
		CHECK(gather_copy->offset_in_dest == id_cast<3>(id(static_cast<size_t>(writer->device_id.value()))));

		CHECK(local_reduce.predecessors().contains(gather_copy));
	}
}

TEST_CASE("global reductions without a local contribution do not read stale local values", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 3;
	const node_id local_nid = GENERATE(values<node_id>({0, 1, 2}));
	const auto num_devices = 1;

	test_utils::idag_test_context ictx(num_nodes, local_nid, num_devices);
	auto buf = ictx.create_buffer(range<1>(1));
	ictx.device_compute(range<1>(2)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	const auto reader_tid = ictx.device_compute(range<1>(num_nodes)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	const auto global_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(global_reduce->scope == reduce_instruction_record::reduction_scope::global);
	CHECK(global_reduce->buffer_id == buf.get_id());
	CHECK(global_reduce->box == box<3>(zeros, ones));
	CHECK(global_reduce->num_source_values == num_nodes);

	const auto gather_recv = all_instrs.select_unique<gather_receive_instruction_record>();
	CHECK(global_reduce.predecessors().contains(gather_recv));
	CHECK(gather_recv->transfer_id.rid == global_reduce->reduction_id);
	CHECK(gather_recv->transfer_id.bid == buf.get_id());

	// the gather-receive buffer must be filled with the reduction identity since some nodes might not contribute partial results (and notify us of that fact at
	// runtime via zero-range pilot messages).
	const auto fill_identity = all_instrs.select_unique<fill_identity_instruction_record>();
	CHECK(gather_recv.predecessors().contains(fill_identity));
	CHECK(fill_identity->memory_id == gather_recv->memory_id);
	CHECK(fill_identity->allocation_id == gather_recv->allocation_id);
	CHECK(fill_identity->num_values == gather_recv->num_nodes);
	CHECK(fill_identity->reduction_id == global_reduce->reduction_id);

	if(local_nid < 2) {
		// there is a local contribution, which will be copied to the global gather buffer concurrent with the receive
		const auto gather_copy = global_reduce.predecessors().select_unique<copy_instruction_record>();
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
		CHECK(gather_copy->dest_memory == gather_recv->memory_id);
		CHECK(gather_copy->dest_allocation == gather_recv->allocation_id);
		CHECK(gather_copy->copy_range == ones);
		CHECK(gather_copy->offset_in_dest == id_cast<3>(id(local_nid)));
		CHECK(gather_copy.is_concurrent_with(gather_recv));

		// fill_identity writes the entire buffer, so we need to overwrite one slot with our contribution
		CHECK(gather_copy.predecessors().contains(fill_identity));

		// since every node participates in the global reduction, we need to push our partial results to every peer
		const auto partial_result_sends = all_instrs.select_all<send_instruction_record>();
		CHECK(partial_result_sends.count() == num_nodes - 1);
		for(const auto& send : partial_result_sends.iterate()) {
			CAPTURE(send);
			CHECK(send->transfer_id == gather_recv->transfer_id);
			CHECK(send->send_range == ones);
			CHECK(send.is_concurrent_with(gather_copy));

			const auto pilot = all_pilots.select_unique(send->dest_node_id);
			CHECK(pilot->message.trid == send->transfer_id);
			CHECK(pilot->message.box == box<3>(zeros, ones));
		}

		// global_reduce will overwrite the host buffer, so it must anti-depend on the partial-result send instructions
		CHECK(global_reduce.predecessors().contains(partial_result_sends));
	} else {
		// there is no local contribution, but we still participate in the global reduction
		CHECK(all_instrs.count<send_instruction_record>() == 0);

		// we signal all peers that we are not going to perform a `send` by transmitting zero-ranged pilots
		for(node_id peer = 0; peer < 2; ++peer) {
			const auto pilot = all_pilots.select_unique(peer);
			CHECK(pilot->message.trid == transfer_id(reader_tid, buf.get_id(), global_reduce->reduction_id));
			CHECK(pilot->message.box == box<3>());
		}
	}

	const auto reader = all_instrs.select_unique<launch_instruction_record>("reader");
	CHECK(reader.transitive_predecessors_across<copy_instruction_record>().contains(global_reduce));
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
