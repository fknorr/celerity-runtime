#pragma once

#include "communicator.h"
#include "instruction_graph.h" // for pilot_message. TODO?

#include <unordered_map>

namespace celerity::detail {

class receive_arbiter {
  private:
	struct region_transfer;

  public:
	class event {
	  public:
		bool is_complete() const;

	  private:
		friend class receive_arbiter;

		const region_transfer* m_region_transfer;
		region<3> m_awaited_region;

		event(const region_transfer* region_transfer, const region<3>& awaited_region) : m_region_transfer(region_transfer), m_awaited_region(awaited_region) {}
	};

	explicit receive_arbiter(communicator& comm);
	receive_arbiter(const receive_arbiter&) = delete;
	receive_arbiter(receive_arbiter&&) = default;
	receive_arbiter& operator=(const receive_arbiter&) = delete;
	receive_arbiter& operator=(receive_arbiter&&) = default;
	~receive_arbiter();

	void begin_receive(const transfer_id& trid, void* allocation, const box<3>& instance_box, size_t elem_size);
	[[nodiscard]] event await_receive(const transfer_id& trid, const region<3>& awaited_region);
	void end_receive(const transfer_id& trid);

	void poll_communicator();

  private:
	struct incoming_fragment {
		box<3> box;
		std::unique_ptr<communicator::event> done;
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
		std::vector<std::unique_ptr<region_transfer>> regions; // unique_ptr for pointer stability (referenced by receive_arbiter::event)
	};

	communicator* m_comm;
	std::unordered_map<transfer_id, buffer_transfer> m_transfers;

	void begin_receiving_fragment(region_transfer* region_tr, const inbound_pilot& pilot, size_t elem_size);
};

} // namespace celerity::detail
