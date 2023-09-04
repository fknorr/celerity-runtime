#pragma once

#include "communicator.h"
#include "instruction_graph.h" // for pilot_message. TODO?

#include <unordered_map>
#include <variant>

namespace celerity::detail {

class recv_arbiter {
  public:
	class event {
	  public:
		bool is_complete() const;

	  private:
		friend class recv_arbiter;
		event(recv_arbiter& arbiter, const transfer_id trid) : m_arbiter(&arbiter), m_trid(trid) {}
		recv_arbiter* m_arbiter;
		transfer_id m_trid;
	};

	explicit recv_arbiter(communicator& comm) : m_comm(&comm) {}
	event begin_aggregated_recv(transfer_id trid, void* allocation, const range<3>& allocation_range, const id<3>& allocation_offset_in_buffer,
	    const id<3>& recv_offset_in_allocation, const range<3>& recv_range, size_t elem_size);
	void push_pilot_message(const node_id source, const pilot_message& pilot);

  private:
	struct waiting_for_begin {
		std::vector<std::pair<node_id, pilot_message>> pilots;
	};
	struct waiting_for_communication {
		std::vector<std::unique_ptr<communicator::event>> active_individual_recvs;
		void* allocation_base;
		range<3> allocation_range;
		id<3> allocation_offset_in_buffer;
		id<3> recv_offset_in_allocation;
		region<3> pending_recv_region_in_allocation;
		size_t elem_size;
	};
	using active_recv = std::variant<waiting_for_begin, waiting_for_communication>;

	communicator* m_comm;
	std::unordered_map<transfer_id, active_recv> m_active;

	void begin_individual_recv(waiting_for_communication& state, node_id source, const pilot_message& pilot);
	bool forget_if_complete(const transfer_id trid);
};

} // namespace celerity::detail
