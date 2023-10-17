#include "receive_arbiter.h"
#include "grid.h"
#include "instruction_graph.h"

#include <exception>
#include <memory>

namespace celerity::detail {

receive_arbiter::receive_arbiter(communicator& comm) : m_comm(&comm), m_num_nodes(comm.get_num_nodes()) {}

receive_arbiter::~receive_arbiter() { assert(std::uncaught_exceptions() > 0 || m_transfers.empty()); }

bool receive_arbiter::event::is_complete() const {
	return matchbox::match(
	    m_state,                                     //
	    [](const completed_state&) { return true; }, //
	    [](const region_transfer_state& rts) { return rts.request.expired(); },
	    [](const subregion_transfer_state& sts) {
		    const auto rr = sts.request.lock();
		    return rr == nullptr || region_intersection(rr->incomplete_region, sts.awaited_region).empty();
	    },
	    [](const gather_transfer_state& gts) { return gts.request.expired(); });
}

receive_arbiter::stable_region_request& receive_arbiter::begin_receive_region(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	assert(allocated_box.covers(bounding_box(request)));

	multi_region_transfer* mrt = nullptr;
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		matchbox::match(
		    entry->second, //
		    [&](unassigned_transfer& ut) { mrt = &utils::replace(entry->second, multi_region_transfer(elem_size, std::move(ut.pilots))); },
		    [&](multi_region_transfer& existing_mrt) { mrt = &existing_mrt; },
		    [&](gather_transfer& gt) { utils::panic("calling receive_arbiter::begin_receive on an active gather transfer"); });
	} else {
		mrt = &utils::replace(m_transfers[trid], multi_region_transfer(elem_size));
	}

	assert(std::all_of(mrt->active_requests.begin(), mrt->active_requests.end(),
	    [&](const stable_region_request& rr) { return box_intersection(rr->allocated_box, allocated_box).empty(); }));
	auto& rr = mrt->active_requests.emplace_back(std::make_shared<region_request>(request, allocation, allocated_box));

	for(auto& pilot : mrt->unassigned_pilots) {
		if(rr->allocated_box.covers(pilot.message.box)) {
			assert(region_intersection(rr->incomplete_region, pilot.message.box) == pilot.message.box);
			begin_receiving_region_fragment(*rr, pilot, elem_size);
		}
	}

	return rr;
}

void receive_arbiter::begin_receive(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	begin_receive_region(trid, request, allocation, allocated_box, elem_size);
}

receive_arbiter::event receive_arbiter::await_subregion_receive(const transfer_id& trid, const region<3>& awaited_region) {
	const auto transfer_it = m_transfers.find(trid);
	if(transfer_it == m_transfers.end()) { return event(event::complete); }

	auto& mrt = std::get<multi_region_transfer>(transfer_it->second);
	const auto awaited_bounds = bounding_box(awaited_region);
	assert(std::all_of(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
		// all boxes from the awaited region must be contained in a single allocation
		const auto overlap = box_intersection(rr->allocated_box, awaited_bounds);
		return overlap.empty() || overlap == awaited_bounds;
	}));

	const auto req_it = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(),
	    [&](const stable_region_request& rr) { return rr->allocated_box.covers(bounding_box(awaited_region)); });
	if(req_it == mrt.active_requests.end()) { return event(event::complete); }

	return event(*req_it, awaited_region);
}

receive_arbiter::event receive_arbiter::receive(
    const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size) {
	return event(begin_receive_region(trid, request, allocation, allocated_box, elem_size));
}

receive_arbiter::event receive_arbiter::receive_gather(const transfer_id& trid, void* allocation, size_t node_chunk_size) {
	auto gr = std::make_shared<gather_request>(allocation, node_chunk_size, m_num_nodes);
	auto event = receive_arbiter::event(gr);
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		auto& ut = std::get<unassigned_transfer>(entry->second);
		for(auto& pilot : ut.pilots) {
			begin_receiving_gather_chunk(*gr, pilot);
		}
		entry->second = gather_transfer{std::move(gr)};
	} else {
		m_transfers.emplace(trid, gather_transfer{std::move(gr)});
	}
	return event;
}

bool receive_arbiter::region_request::complete() {
	const auto incomplete_fragments_end = std::remove_if(incoming_fragments.begin(), incoming_fragments.end(), [&](const incoming_region_fragment& fragment) {
		const bool is_complete = fragment.done->is_complete();
		if(is_complete) { incomplete_region = region_difference(incomplete_region, fragment.box); }
		return is_complete;
	});
	incoming_fragments.erase(incomplete_fragments_end, incoming_fragments.end());
	return incomplete_region.empty();
}

bool receive_arbiter::multi_region_transfer::complete() {
	const auto incomplete_req_end = std::remove_if(active_requests.begin(), active_requests.end(), [&](stable_region_request& rr) { return rr->complete(); });
	active_requests.erase(incomplete_req_end, active_requests.end());
	return active_requests.empty();
}

bool receive_arbiter::gather_request::complete() {
	const auto incomplete_chunks_end = std::remove_if(incoming_chunks.begin(), incoming_chunks.end(), [&](const incoming_gather_chunk& chunk) {
		const bool is_complete = chunk.done->is_complete();
		if(is_complete) { num_incomplete_chunks -= 1; }
		return is_complete;
	});
	incoming_chunks.erase(incomplete_chunks_end, incoming_chunks.end());
	return num_incomplete_chunks == 0;
}

bool receive_arbiter::gather_transfer::complete() { return request->complete(); }

bool receive_arbiter::unassigned_transfer::complete() { // NOLINT(readability-convert-member-functions-to-static)
	// an unassigned_transfer inside receive_arbiter::m_transfers is never empty.
	return false;
}

void receive_arbiter::poll_communicator() {
	for(auto entry = m_transfers.begin(); entry != m_transfers.end();) {
		if(std::visit([](auto& transfer) { return transfer.complete(); }, entry->second)) {
			entry = m_transfers.erase(entry);
		} else {
			++entry;
		}
	}

	for(const auto& pilot : m_comm->poll_inbound_pilots()) {
		if(const auto entry = m_transfers.find(pilot.message.trid); entry != m_transfers.end()) {
			matchbox::match(
			    entry->second, //
			    [&](unassigned_transfer& ut) { ut.pilots.push_back(pilot); },
			    [&](multi_region_transfer& mrt) {
				    const auto rr = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(),
				        [&](const stable_region_request& rr) { return rr->allocated_box.covers(pilot.message.box); });
				    assert(rr != mrt.active_requests.end());
				    assert(region_intersection((*rr)->incomplete_region, pilot.message.box) == pilot.message.box);
				    begin_receiving_region_fragment(**rr, pilot, mrt.elem_size); // elem_size is set when transfer_region is inserted
			    },
			    [&](gather_transfer& gt) { begin_receiving_gather_chunk(*gt.request, pilot); });
		} else {
			m_transfers.emplace(pilot.message.trid, unassigned_transfer{{pilot}});
		}
	}
}

void receive_arbiter::begin_receiving_region_fragment(region_request& rr, const inbound_pilot& pilot, const size_t elem_size) {
	assert(rr.allocated_box.covers(pilot.message.box));
	const auto offset_in_allocation = pilot.message.box.get_offset() - rr.allocated_box.get_offset();
	const communicator::stride stride{
	    rr.allocated_box.get_range(),
	    subrange<3>{offset_in_allocation, pilot.message.box.get_range()},
	    elem_size,
	};
	auto event = m_comm->receive_payload(pilot.from, pilot.message.tag, rr.allocation, stride);
	rr.incoming_fragments.push_back({pilot.message.box, std::move(event)});
}

void receive_arbiter::begin_receiving_gather_chunk(gather_request& gr, const inbound_pilot& pilot) {
	const communicator::stride stride{range_cast<3>(range(m_num_nodes)), subrange(id_cast<3>(id(pilot.from)), range_cast<3>(range(1))), gr.chunk_size};
	auto event = m_comm->receive_payload(pilot.from, pilot.message.tag, gr.allocation, stride);
	gr.incoming_chunks.push_back(incoming_gather_chunk{std::move(event)});
}

} // namespace celerity::detail
