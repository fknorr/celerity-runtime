#pragma once

#include "communicator.h"
#include "instruction_graph.h" // for pilot_message. TODO?

#include <unordered_map>

namespace celerity::detail {

class receive_arbiter {
  private:
	struct region_request;
	struct gather_transfer;

  public:
	class event {
	  public:
		bool is_complete() const;

	  private:
		friend class receive_arbiter;

		struct complete_tag{} inline static constexpr complete;

		struct completed_state {};
		struct region_transfer_state {
			const std::weak_ptr<region_request> region_transfer;
			region<3> awaited_region;
		};
		struct gather_transfer_state {
			const std::weak_ptr<gather_transfer> gather_transfer;
		};
		using state = std::variant<completed_state, region_transfer_state, gather_transfer_state>;

		state m_state;

		explicit event(const complete_tag) : m_state(completed_state{}) {}
		explicit event(const std::shared_ptr<region_request>& region_transfer, const region<3>& awaited_region)
		    : m_state(region_transfer_state{region_transfer, awaited_region}) {}
		explicit event(const std::shared_ptr<gather_transfer>& gather_transfer) : m_state(gather_transfer_state{gather_transfer}) {}
	};

	explicit receive_arbiter(communicator& comm);
	receive_arbiter(const receive_arbiter&) = delete;
	receive_arbiter(receive_arbiter&&) = default;
	receive_arbiter& operator=(const receive_arbiter&) = delete;
	receive_arbiter& operator=(receive_arbiter&&) = default;
	~receive_arbiter();

	// TODO should take a vector of allocations, then we can omit end_receive!
	void begin_receive(const transfer_id& trid, region<3> request, void* allocation, const box<3>& allocated_box, size_t elem_size);
	[[nodiscard]] event await_receive(const transfer_id& trid, const region<3>& awaited_region);
	void end_receive(const transfer_id& trid);

	// This is a temporary solution until we implement inter-node reductions through MPI collectives.
	[[nodiscard]] event receive_gather(const transfer_id& trid, void* allocation, size_t node_chunk_size);

	void poll_communicator();

  private:
	struct incoming_region_fragment {
		box<3> box;
		std::unique_ptr<communicator::event> done;
	};

	struct region_request {
		void* allocation;
		box<3> allocation_box;
		region<3> incomplete_region;
		std::vector<incoming_region_fragment> incoming_fragments;

		region_request(region<3>&& requested_region, void* const allocation, const box<3>& allocated_bounding_box)
		    : allocation(allocation), allocation_box(allocated_bounding_box), incomplete_region(std::move(requested_region)) {}
		void commit_completed_fragments();
		bool is_complete() const;
	};

	// unique_ptr for pointer stability (referenced by receive_arbiter::event)
	using stable_region_request = std::shared_ptr<region_request>;

	struct multi_region_transfer {
		size_t elem_size;
		std::vector<inbound_pilot> unassigned_pilots;
		std::vector<stable_region_request> active_requests;
		explicit multi_region_transfer(const size_t elem_size) : elem_size(elem_size) {}
		explicit multi_region_transfer(const size_t elem_size, std::vector<inbound_pilot>&& pilots)
		    : elem_size(elem_size), unassigned_pilots(std::move(pilots)) {}
		void commit_completed_requests();
		bool is_complete() const;
	};
	using stable_multi_region_transfer = std::unique_ptr<multi_region_transfer>;

	struct incoming_gather_chunk {
		std::unique_ptr<communicator::event> done;
	};

	struct gather_transfer {
		void* allocation;
		size_t chunk_size;
		size_t num_incomplete_chunks;
		std::vector<incoming_gather_chunk> incoming_chunks;

		gather_transfer(void* const allocation, const size_t chunk_size, const size_t num_total_chunks)
		    : allocation(allocation), chunk_size(chunk_size), num_incomplete_chunks(num_total_chunks) {}
		void commit_completed_chunks();
		bool is_complete() const;
	};
	using stable_gather_transfer = std::shared_ptr<gather_transfer>;

	struct unassigned_transfer {
		std::vector<inbound_pilot> pilots;
	};

	using transfer = std::variant<unassigned_transfer, stable_multi_region_transfer, stable_gather_transfer>;

	communicator* m_comm;
	size_t m_num_nodes;
	std::unordered_map<transfer_id, transfer> m_transfers;

	void begin_receiving_region_fragment(region_request& rr, const inbound_pilot& pilot, size_t elem_size);
	void begin_receiving_gather_chunk(gather_transfer& gt, const inbound_pilot& pilot);
};

} // namespace celerity::detail
