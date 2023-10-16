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
	    m_state,
	    [](const completed_state&) { //
		    return true;
	    },
	    [](const region_transfer_state& rts) {
		    const auto rr = rts.region_transfer.lock();
		    return rr == nullptr || region_intersection(rr->incomplete_region, rts.awaited_region).empty();
	    },
	    [](const gather_transfer_state& gts) { //
		    return gts.gather_transfer.expired();
	    });
}

void receive_arbiter::begin_receive(const transfer_id& trid, const region<3>& request, const std::vector<tile>& tiles, const size_t elem_size) {
	auto new_mrt = std::make_unique<multi_region_transfer>(elem_size);
	const auto mrt = new_mrt.get();
	std::vector<inbound_pilot> queued_pilots;
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		auto& ut = std::get<unassigned_transfer>(entry->second);
		queued_pilots = std::move(ut.pilots);
		entry->second = std::move(new_mrt);
	} else {
		m_transfers.emplace(trid, std::move(new_mrt));
	}

	mrt->active_requests.reserve(tiles.size());
	for(auto& tile : tiles) {
		mrt->active_requests.push_back(std::make_shared<region_request>(region_intersection(request, tile.allocated_box), tile.allocation, tile.allocated_box));
	}

	for(auto& pilot : queued_pilots) {
		const auto rr = std::find_if(mrt->active_requests.begin(), mrt->active_requests.end(),
		    [&](const stable_region_request& rr) { return rr->allocation_box.covers(pilot.message.box); });
		assert(rr != mrt->active_requests.end());
		assert(region_intersection((*rr)->incomplete_region, pilot.message.box) == pilot.message.box);
		begin_receiving_region_fragment(**rr, pilot, elem_size);
	}
}

receive_arbiter::event receive_arbiter::await_partial_receive(const transfer_id& trid, const region<3>& awaited_region) {
	const auto transfer_it = m_transfers.find(trid);
	if(transfer_it == m_transfers.end()) { return event(event::complete); }

	auto& mrt = *std::get<stable_multi_region_transfer>(transfer_it->second);
	const auto awaited_bounds = bounding_box(awaited_region);
	assert(std::all_of(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
		// all boxes from the awaited region must be contained in a single allocation
		const auto overlap = box_intersection(rr->allocation_box, awaited_bounds);
		return overlap.empty() || overlap == awaited_bounds;
	}));

	const auto req_it = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(),
	    [&](const stable_region_request& rr) { return rr->allocation_box.covers(bounding_box(awaited_region)); });
	if(req_it == mrt.active_requests.end()) { return event(event::complete); }

	return event(*req_it, awaited_region);
}

receive_arbiter::event receive_arbiter::receive_gather(const transfer_id& trid, void* allocation, size_t node_chunk_size) {
	auto gt = std::make_shared<gather_transfer>(allocation, node_chunk_size, m_num_nodes);
	auto event = receive_arbiter::event(gt);
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		auto& ut = std::get<unassigned_transfer>(entry->second);
		for(auto& pilot : ut.pilots) {
			begin_receiving_gather_chunk(*gt, pilot);
		}
		entry->second = std::move(gt);
	} else {
		m_transfers.emplace(trid, std::move(gt));
	}
	return event;
}

void receive_arbiter::region_request::commit_completed_fragments() {
	const auto incomplete_fragments_end = std::remove_if(incoming_fragments.begin(), incoming_fragments.end(), [&](const incoming_region_fragment& fragment) {
		const bool is_complete = fragment.done->is_complete();
		if(is_complete) { incomplete_region = region_difference(incomplete_region, fragment.box); }
		return is_complete;
	});
	incoming_fragments.erase(incomplete_fragments_end, incoming_fragments.end());
}

bool receive_arbiter::region_request::is_complete() const { return incomplete_region.empty(); }

void receive_arbiter::multi_region_transfer::commit_completed_requests() {
	const auto incomplete_requests_end = std::remove_if(active_requests.begin(), active_requests.end(), [&](const stable_region_request& rr) {
		rr->commit_completed_fragments();
		return rr->is_complete();
	});
	active_requests.erase(incomplete_requests_end, active_requests.end());
}

bool receive_arbiter::multi_region_transfer::is_complete() const { return active_requests.empty(); }

void receive_arbiter::gather_transfer::commit_completed_chunks() {
	const auto incomplete_chunks_end = std::remove_if(incoming_chunks.begin(), incoming_chunks.end(), [&](const incoming_gather_chunk& chunk) {
		const bool is_complete = chunk.done->is_complete();
		if(is_complete) { num_incomplete_chunks -= 1; }
		return is_complete;
	});
	incoming_chunks.erase(incomplete_chunks_end, incoming_chunks.end());
}

bool receive_arbiter::gather_transfer::is_complete() const { return num_incomplete_chunks == 0; }

void receive_arbiter::poll_communicator() {
	for(auto entry = m_transfers.begin(); entry != m_transfers.end();) {
		const auto complete = matchbox::match(
		    entry->second,
		    [&](const unassigned_transfer& /* ut */) { //
			    return false;
		    },
		    [&](stable_multi_region_transfer& mrt) {
			    mrt->commit_completed_requests();
			    return mrt->is_complete();
		    },
		    [&](stable_gather_transfer& gt) {
			    gt->commit_completed_chunks();
			    return gt->is_complete();
		    });
		if(complete) {
			entry = m_transfers.erase(entry);
		} else {
			++entry;
		}
	}

	for(const auto& pilot : m_comm->poll_inbound_pilots()) {
		if(const auto entry = m_transfers.find(pilot.message.trid); entry != m_transfers.end()) {
			matchbox::match(
			    entry->second,
			    [&](unassigned_transfer& ut) { //
				    ut.pilots.push_back(pilot);
			    },
			    [&](stable_multi_region_transfer& mrt) {
				    const auto rr = std::find_if(mrt->active_requests.begin(), mrt->active_requests.end(),
				        [&](const stable_region_request& rr) { return rr->allocation_box.covers(pilot.message.box); });
				    assert(rr != mrt->active_requests.end());
				    assert(region_intersection((*rr)->incomplete_region, pilot.message.box) == pilot.message.box);
				    begin_receiving_region_fragment(**rr, pilot, mrt->elem_size); // elem_size is set when transfer_region is inserted
			    },
			    [&](stable_gather_transfer& gt) { begin_receiving_gather_chunk(*gt, pilot); });
		} else {
			m_transfers.emplace(pilot.message.trid, unassigned_transfer{{pilot}});
		}
	}
}

void receive_arbiter::begin_receiving_region_fragment(region_request& rr, const inbound_pilot& pilot, const size_t elem_size) {
	assert(rr.allocation_box.covers(pilot.message.box));
	const auto offset_in_allocation = pilot.message.box.get_offset() - rr.allocation_box.get_offset();
	const communicator::stride stride{
	    rr.allocation_box.get_range(),
	    subrange<3>{offset_in_allocation, pilot.message.box.get_range()},
	    elem_size,
	};
	auto event = m_comm->receive_payload(pilot.from, pilot.message.tag, rr.allocation, stride);
	rr.incoming_fragments.push_back({pilot.message.box, std::move(event)});
}

void receive_arbiter::begin_receiving_gather_chunk(gather_transfer& gt, const inbound_pilot& pilot) {
	const communicator::stride stride{range_cast<3>(range(m_num_nodes)), subrange(id_cast<3>(id(pilot.from)), range_cast<3>(range(1))), gt.chunk_size};
	auto event = m_comm->receive_payload(pilot.from, pilot.message.tag, gt.allocation, stride);
	gt.incoming_chunks.push_back(incoming_gather_chunk{std::move(event)});
}

} // namespace celerity::detail
