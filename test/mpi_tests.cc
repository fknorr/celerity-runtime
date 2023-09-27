#include "communicator.h"
#include "host_utils.h"
#include "instruction_graph.h" // for pilot_message TODO
#include "mpi_communicator.h"
#include "spdlog/common.h"
#include "test_utils.h"
#include "types.h"

#include <chrono>
#include <mutex>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/sink.h>

using namespace celerity;
using namespace celerity::detail;


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives pilot messages", "[mpi]") {
	class test_delegate final : public communicator::delegate {
	  public:
		void inbound_pilot_received(const inbound_pilot& pilot) override {
			std::lock_guard lock(mutex);
			pilots_received.push_back(pilot);
		}

		std::mutex mutex;
		std::vector<inbound_pilot> pilots_received;
	};

	test_delegate test;
	mpi_communicator comm(MPI_COMM_WORLD, &test);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_pilot_message = [&](const node_id sender, const node_id receiver) {
		const auto p2p_id = 1 + sender * comm.get_num_nodes() + receiver;
		const int tag = static_cast<int>(p2p_id) * 13;
		const buffer_id bid = p2p_id * 11;
		const transfer_id trid = p2p_id * 17;
		const box<3> box = {id{p2p_id, p2p_id * 2, p2p_id * 3}, id{p2p_id * 4, p2p_id * 5, p2p_id * 6}};
		return outbound_pilot{receiver, pilot_message{tag, bid, trid, box}};
	};

	for(node_id to = 0; to < comm.get_num_nodes(); ++to) {
		if(to == comm.get_local_node_id()) continue;
		comm.send_outbound_pilot(make_pilot_message(comm.get_local_node_id(), to));
	}

	for(;;) {
		{
			std::lock_guard lock(test.mutex);
			if(test.pilots_received.size() == comm.get_num_nodes() - 1) break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1)); // don't contend the mutex
	}

	std::lock_guard lock(test.mutex);
	for(const auto& pilot : test.pilots_received) {
		const auto expect = make_pilot_message(pilot.from, comm.get_local_node_id());
		CHECK(pilot.message.tag == expect.message.tag);
		CHECK(pilot.message.buffer == expect.message.buffer);
		CHECK(pilot.message.transfer == expect.message.transfer);
		CHECK(pilot.message.box == expect.message.box);
	}
}


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives payloads", "[mpi]") {
	mpi_communicator comm(MPI_COMM_WORLD, /* delegate= */ nullptr);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_tag = [&](const node_id sender, const node_id receiver) { //
		return static_cast<int>(1 + sender * comm.get_num_nodes() + receiver);
	};

	const communicator::stride stride{{12, 11, 11}, {{1, 0, 3}, {5, 4, 6}}, sizeof(int)};

	std::vector<std::vector<int>> send_buffers;
	std::vector<std::vector<int>> receive_buffers;
	std::vector<std::unique_ptr<communicator::event>> events;
	for(node_id other = 0; other < comm.get_num_nodes(); ++other) {
		if(other == comm.get_local_node_id()) continue;

		auto& send = send_buffers.emplace_back(stride.allocation.size());
		std::iota(send.begin(), send.end(), make_tag(comm.get_local_node_id(), other));
		auto& receive = receive_buffers.emplace_back(stride.allocation.size());
		events.push_back(comm.send_payload(other, make_tag(comm.get_local_node_id(), other), send.data(), stride));
		events.push_back(comm.receive_payload(other, make_tag(other, comm.get_local_node_id()), receive.data(), stride));
	}

	while(!events.empty()) {
		const auto end_incomplete = std::remove_if(events.begin(), events.end(), [](const auto& evt) { return evt->is_complete(); });
		events.erase(end_incomplete, events.end());
	}

	auto received = receive_buffers.begin();
	for(node_id other = 0; other < comm.get_num_nodes(); ++other) {
		if(other == comm.get_local_node_id()) continue;

		std::vector<int> other_send(stride.allocation.size());
		std::iota(other_send.begin(), other_send.end(), make_tag(other, comm.get_local_node_id()));
		std::vector<int> expected(stride.allocation.size());
		experimental::for_each_item(stride.subrange.range, [&](const item<3>& item) {
			const auto id = stride.subrange.offset + item.get_id();
			const auto linear_index = get_linear_index(stride.allocation, id);
			expected[linear_index] = other_send[linear_index];
		});
		CHECK(*received == expected);
		++received;
	}
}
