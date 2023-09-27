#include "buffer_storage.h" // for memcpy_strided_host
#include "host_utils.h"
#include "recv_arbiter.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace celerity;
using namespace celerity::detail;

class mock_recv_communicator : public communicator {
  public:
	mock_recv_communicator() = default;
	mock_recv_communicator(mock_recv_communicator&&) = default;
	mock_recv_communicator& operator=(mock_recv_communicator&&) = default;
	mock_recv_communicator(const mock_recv_communicator&) = delete;
	mock_recv_communicator& operator=(const mock_recv_communicator&) = delete;
	~mock_recv_communicator() override { CHECK(m_pending_recvs.empty()); }

	size_t get_num_nodes() const override { utils::panic("unimplemented"); }
	node_id get_local_node_id() const override { utils::panic("unimplemented"); }
	void send_outbound_pilot(const outbound_pilot& /* pilot */) override { utils::panic("unimplemented"); }
	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override { utils::panic("unimplemented"); }

	[[nodiscard]] std::unique_ptr<communicator::event> send_payload(
	    const node_id /* to */, const int /* outbound_pilot_tag */, const void* const /* base */, const stride& /* stride */) override {
		utils::panic("unimplemented");
	}

	[[nodiscard]] std::unique_ptr<communicator::event> receive_payload(
	    const node_id from, const int inbound_pilot_tag, void* const base, const stride& stride) override {
		const auto key = std::pair(from, inbound_pilot_tag);
		REQUIRE(m_pending_recvs.count(key) == 0);
		completion_flag flag = std::make_shared<bool>(false);
		m_pending_recvs.emplace(key, std::tuple(base, stride, flag));
		return std::make_unique<event>(flag);
	}

	void complete_receiving_payload(const node_id from, const int inbound_pilot_tag, const void* const src, const range<3>& src_range) {
		const auto key = std::pair(from, inbound_pilot_tag);
		const auto [dest, stride, flag] = m_pending_recvs.at(key);
		REQUIRE(src_range == stride.subrange.range);
		memcpy_strided_host(src, dest, stride.element_size, src_range, zeros, stride.allocation, stride.subrange.offset, stride.subrange.range);
		*flag = true;
		m_pending_recvs.erase(key);
	}

  private:
	using completion_flag = std::shared_ptr<bool>;

	class event final : public communicator::event {
	  public:
		explicit event(const completion_flag& flag) : m_flag(flag) {}
		bool is_complete() const override { return *m_flag; }

	  private:
		completion_flag m_flag;
	};

	std::unordered_map<std::pair<node_id, int>, std::tuple<void*, stride, completion_flag>, utils::pair_hash> m_pending_recvs;
};

TEST_CASE("recv_arbiter correctly aggregates receives", "[recv_arbiter]") {
	const transfer_id trid = 1234567890;
	const buffer_id bid = 420;
	const range<3> buffer_range = {40, 10, 12};
	const box<3> alloc_box = {{2, 1, 0}, {39, 10, 10}};
	const box<3> recv_box = {{4, 2, 1}, {37, 9, 8}};
	const size_t elem_size = sizeof(int);

	const std::tuple<node_id, int, box<3>> fragments_meta[] = {
	    {node_id(1), 15, box<3>({4, 2, 1}, {22, 9, 4})},
	    {node_id(2), 14, box<3>({4, 2, 4}, {22, 9, 8})},
	    {node_id(3), 12, box<3>({22, 2, 1}, {37, 9, 4})},
	    {node_id(4), 13, box<3>({22, 2, 4}, {37, 9, 8})},
	};
	constexpr size_t num_fragments = std::size(fragments_meta);
	CAPTURE(num_fragments);

	std::vector<std::vector<int>> fragments;
	for(const auto& [from, tag, box] : fragments_meta) {
		fragments.emplace_back(box.get_range().size(), static_cast<int>(from));
	}
	std::vector<inbound_pilot> pilots;
	for(const auto& [from, tag, box] : fragments_meta) {
		pilots.push_back(inbound_pilot{from, pilot_message{tag, bid, trid, box}});
	}

	mock_recv_communicator comm;
	recv_arbiter ra(comm);

	const size_t num_pilots_pushed_before_recv = GENERATE(values<size_t>({0, num_fragments / 2, num_fragments}));
	CAPTURE(num_pilots_pushed_before_recv);
	for(size_t i = 0; i < num_pilots_pushed_before_recv; ++i) {
		ra.push_inbound_pilot(pilots[i]);
	}

	std::vector<int> allocation(alloc_box.get_range().size());
	const auto event = ra.begin_aggregated_recv(trid, bid, allocation.data(), alloc_box.get_range(), alloc_box.get_offset(),
	    recv_box.get_offset() - alloc_box.get_offset(), recv_box.get_range(), elem_size);

	for(size_t i = 0; i < num_pilots_pushed_before_recv; ++i) {
		const auto& [from, tag, box] = fragments_meta[i];
		comm.complete_receiving_payload(from, tag, fragments[i].data(), box.get_range());
	}

	for(size_t i = num_pilots_pushed_before_recv; i < num_fragments; ++i) {
		ra.push_inbound_pilot(pilots[i]);
	}

	for(size_t i = num_pilots_pushed_before_recv; i < num_fragments; ++i) {
		const auto& [from, tag, box] = fragments_meta[i];
		comm.complete_receiving_payload(from, tag, fragments[i].data(), box.get_range());
	}

	CHECK(event.is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(const auto& [from, tag, box] : fragments_meta) {
		experimental::for_each_item(box.get_range(), [&, from = from, &box = box](const item<3>& it) {
			const auto id_in_allocation = box.get_offset() - alloc_box.get_offset() + it.get_id();
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(from);
		});
	}
	CHECK(allocation == expected_allocation);
}
