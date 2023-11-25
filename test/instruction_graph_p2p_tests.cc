#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;

namespace acc = celerity::access;


TEMPLATE_TEST_CASE_SIG("buffer subranges are sent and received to satisfy push and await-push commands",
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id peer_nid = 1 - local_nid;
	CAPTURE(local_nid, peer_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);

	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
	const auto send = all_instrs.select_unique<send_instruction_record>();
	const auto recv = all_instrs.select_unique<receive_instruction_record>();
	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	// we send exactly the part of the buffer that our node has written
	REQUIRE(writer->access_map.size() == 1);
	const auto& write_access = writer->access_map.front();
	CHECK(send->dest_node_id == peer_nid);
	CHECK(send->transfer_id == expected_trid);
	CHECK(send->send_range == write_access.accessed_box_in_buffer.get_range());
	CHECK(send->offset_in_buffer == write_access.accessed_box_in_buffer.get_offset());
	CHECK(send->element_size == sizeof(int));

	// a pilot is attached to the send
	const auto pilot = ictx.query_outbound_pilots();
	CHECK(pilot.is_unique());
	CHECK(pilot->to == peer_nid);
	CHECK(pilot->message.transfer_id == send->transfer_id);
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
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) {
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id peer_nid = 1 - local_nid;
	CAPTURE(local_nid, peer_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, test_utils::reverse_one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
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
		    intersection_of(send.transitive_predecessors_across<copy_instruction_record>(), all_writers).assert_unique<device_kernel_instruction_record>();
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
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");

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
		    intersection_of(await_recv.transitive_successors_across<copy_instruction_record>(), all_readers).assert_unique<device_kernel_instruction_record>();
		REQUIRE(associated_reader->access_map.size() == 1);
		const auto& read = associated_reader->access_map.front();

		CHECK(await_recv->received_region == region(read.accessed_box_in_buffer));
		CHECK(await_recv->transfer_id == expected_trid);
	}
}

TEMPLATE_TEST_CASE_SIG("overlapping requirements generate split-receives with one await per reader-set",
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 4 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::all()).submit();
	ictx.device_compute(test_range).name("reader").read(buf, test_utils::make_neighborhood<Dims>(1)).submit();
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
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
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

TEST_CASE("an await-push of disconnected subregions does not allocate their bounding-box", "[instruction_graph_generator][instruction-graph][p2p]") {
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

	const auto reader = all_instrs.select_unique<host_task_instruction_record>("reader");
	CHECK(reader.predecessors() == all_recvs);
}
