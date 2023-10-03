#pragma once

#include "communicator.h"
#include "instruction_graph.h" // for pilot_message. TODO?
#include "utils.h"

#include <unordered_map>
#include <variant>

namespace celerity::detail {

class recv_arbiter {
  private:
	struct region_transfer;

  public:
	class event {
	  public:
		bool is_complete() const;

	  private:
		friend class recv_arbiter;

		const region_transfer* m_region_transfer;
		region<3> m_awaited_region;

		event(const region_transfer* region_transfer, const region<3>& awaited_region) : m_region_transfer(region_transfer), m_awaited_region(awaited_region) {}
	};

	explicit recv_arbiter(communicator& comm);
	recv_arbiter(const recv_arbiter&) = delete;
	recv_arbiter(recv_arbiter&&) = default;
	recv_arbiter& operator=(const recv_arbiter&) = delete;
	recv_arbiter& operator=(recv_arbiter&&) = default;
	~recv_arbiter();

	void begin_receive(transfer_id trid, buffer_id bid, void* allocation, const box<3>& allocated_box, size_t elem_size);
	[[nodiscard]] event await_receive(transfer_id trid, buffer_id bid, const region<3>& awaited_region);
	void end_receive(transfer_id trid, buffer_id bid);

	void poll_communicator();

  private:
	struct incoming_fragment {
		box<3> box;
		std::unique_ptr<communicator::event> done;
	};
	struct waiting_for_begin {
		std::vector<inbound_pilot> pilots;
	};
	struct region_transfer {
		void* allocation;
		box<3> allocated_bounding_box;
		region<3> complete_region;
		std::vector<incoming_fragment> incoming_fragments;

		region_transfer(void* const allocation, const box<3>& allocated_bounding_box)
		    : allocation(allocation), allocated_bounding_box(allocated_bounding_box) {}
	};
	struct buffer_transfer {
		std::optional<size_t> elem_size; // nullopt until the first `transfer` is inserted
		std::vector<inbound_pilot> unassigned_pilots;
		std::vector<std::unique_ptr<region_transfer>> regions; // unique_ptr for pointer stability (referenced by recv_arbiter::event)
	};

	communicator* m_comm;
	std::unordered_map<std::pair<transfer_id, buffer_id>, buffer_transfer, utils::pair_hash> m_transfers;

	void begin_receiving_fragment(region_transfer* region_tr, const inbound_pilot& pilot, size_t elem_size);
};

} // namespace celerity::detail
