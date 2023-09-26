#include "recv_arbiter.h"
#include "device_queue.h"
#include "grid.h"
#include "instruction_graph.h"
#include "utils.h"
#include <memory>

namespace celerity::detail {

bool recv_arbiter::event::is_complete() const {
	// TODO dropping an event without calling is_complete() => true will keep stale entries around
	return m_arbiter->forget_if_complete(m_trid, m_bid);
}

recv_arbiter::event recv_arbiter::begin_aggregated_recv(const transfer_id trid, const buffer_id bid, void* const allocation_base,
    const range<3>& allocation_range, const id<3>& allocation_offset_in_buffer, const id<3>& recv_offset_in_allocation, const range<3>& recv_range,
    size_t elem_size) {
	const auto key = std::pair(trid, bid);
	auto new_state = waiting_for_communication{{}, allocation_base, allocation_range, allocation_offset_in_buffer, recv_offset_in_allocation,
	    subrange(recv_offset_in_allocation, recv_range), elem_size};
	if(const auto it = m_active.find(key); it != m_active.end()) {
		auto& [_, active_state] = *it;
		for(auto pilot : std::get<waiting_for_begin>(active_state).pilots) {
			begin_individual_recv(new_state, pilot);
		}
		active_state = std::move(new_state);
	} else {
		m_active.emplace(key, std::move(new_state));
		std::vector<std::unique_ptr<communicator::event>> active_individual_recvs;
	}
	return event(*this, trid, bid);
}

void recv_arbiter::push_inbound_pilot(const inbound_pilot& pilot) {
	const auto key = std::pair(pilot.message.transfer, pilot.message.buffer);
	if(const auto it = m_active.find(key); it != m_active.end()) {
		matchbox::match(
		    it->second,                                                                    //
		    [&](waiting_for_begin& state) { state.pilots.push_back(pilot); },              //
		    [&](waiting_for_communication& state) { begin_individual_recv(state, pilot); } //
		);
	} else {
		m_active.emplace(key, waiting_for_begin{{pilot}});
	}
}

void recv_arbiter::begin_individual_recv(waiting_for_communication& state, const inbound_pilot& pilot) {
	const auto [offset_in_buffer, payload_range] = pilot.message.box.get_subrange();
	assert(all_true(offset_in_buffer >= state.allocation_offset_in_buffer));
	const auto offset_in_allocation = offset_in_buffer - state.allocation_offset_in_buffer;
	assert(all_true(offset_in_allocation + payload_range <= state.allocation_range));
	const communicator::stride stride{
	    state.allocation_range,
	    subrange<3>{offset_in_allocation, payload_range},
	    state.elem_size,
	};
	state.active_individual_recvs.push_back(m_comm->receive_payload(pilot.from, pilot.message.tag, state.allocation_base, stride));
	state.pending_recv_region_in_allocation = region_difference(state.pending_recv_region_in_allocation, box(stride.subrange));
}

bool recv_arbiter::forget_if_complete(const transfer_id trid, const buffer_id bid) {
	const auto key = std::pair(trid, bid);
	const auto active = m_active.find(key);
	if(active == m_active.end()) return true;

	auto& state = std::get<waiting_for_communication>(active->second);
	if(!state.pending_recv_region_in_allocation.empty()) return false;

	const auto incomplete_end = std::remove_if(state.active_individual_recvs.begin(), state.active_individual_recvs.end(),
	    [](const std::unique_ptr<communicator::event>& evt) { return evt->is_complete(); });
	state.active_individual_recvs.erase(incomplete_end, state.active_individual_recvs.end());
	if(!state.active_individual_recvs.empty()) return false;

	m_active.erase(active);
	return true;
}

} // namespace celerity::detail
