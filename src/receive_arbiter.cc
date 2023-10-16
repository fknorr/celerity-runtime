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

void receive_arbiter::begin_receive(const transfer_id& trid, region<3> request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	multi_region_transfer* mrt = nullptr;
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		matchbox::match(
		    entry->second,
		    [&](unassigned_transfer& ut) {
			    entry->second = std::make_unique<multi_region_transfer>(elem_size, std::move(ut.pilots));
			    mrt = std::get<stable_multi_region_transfer>(entry->second).get();
		    },
		    [&](stable_multi_region_transfer& existing_mrt) { mrt = existing_mrt.get(); },
		    [](stable_gather_transfer&) { utils::panic("Calling begin_receive on an active gather transfer"); });
	} else {
		auto new_mrt = std::make_unique<multi_region_transfer>(elem_size);
		mrt = new_mrt.get();
		m_transfers.emplace(trid, std::move(new_mrt));
	}
	assert(mrt != nullptr);

	// There can be multiple begin_receives (i.e. disjoint allocations) for a single await_push, but allocations (currently) are not allowed to overlap.
	assert(std::all_of(mrt->active_requests.begin(), mrt->active_requests.end(),
	    [&](const stable_region_request& rr) { return box_intersection(rr->allocation_box, allocated_box).empty(); }));

	auto& rr = *mrt->active_requests.emplace_back(std::make_unique<region_request>(std::move(request), allocation, allocated_box));
	const auto remaining_unassigned_pilots_end = std::remove_if(mrt->unassigned_pilots.begin(), mrt->unassigned_pilots.end(), [&](const inbound_pilot& pilot) {
		assert(allocated_box.covers(pilot.message.box) == !box_intersection(allocated_box, pilot.message.box).empty());
		const auto now_assigned = allocated_box.covers(pilot.message.box);
		if(now_assigned) { begin_receiving_region_fragment(rr, pilot, elem_size); }
		return now_assigned;
	});
	mrt->unassigned_pilots.erase(remaining_unassigned_pilots_end, mrt->unassigned_pilots.end());
}

receive_arbiter::event receive_arbiter::await_receive(const transfer_id& trid, const region<3>& awaited_region) {
	auto& transfer = *std::get<stable_multi_region_transfer>(m_transfers.at(trid));
	const auto region_it = std::find_if(transfer.active_requests.begin(), transfer.active_requests.end(),
	    [&](const stable_region_request& rr) { return rr->allocation_box.covers(bounding_box(awaited_region)); });
	return region_it != transfer.active_requests.end() ? event(*region_it, awaited_region) : event(event::complete);
}

void receive_arbiter::end_receive(const transfer_id& trid) {
#if CELERITY_DETAIL_ENABLE_DEBUG
	matchbox::match(
	    m_transfers.at(trid), //
	    [](unassigned_transfer& ut) { assert(!"calling end_receive before begin_receive"); },
	    [](stable_multi_region_transfer& mrt) {
		    assert(std::all_of(mrt->active_requests.begin(), mrt->active_requests.end(), //
		        [](const stable_region_request& rr) { return rr->incoming_fragments.empty(); }));
	    },
	    [](stable_gather_transfer&) { utils::panic("calling end_receive on a gather transfer"); });
#endif

	m_transfers.erase(trid);
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

bool receive_arbiter::multi_region_transfer::is_complete() const {
	// just because there is no incomplete request and unassigned pilots, we do not know if any will arrive in the future - hence a multi_region_transfer must
	// be explicitly terminated by end_receive.
	return false;
}

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
				    if(rr != mrt->active_requests.end()) {
					    assert(region_intersection((*rr)->incomplete_region, pilot.message.box) == pilot.message.box);
					    begin_receiving_region_fragment(**rr, pilot, mrt->elem_size); // elem_size is set when transfer_region is inserted
				    } else {
					    mrt->unassigned_pilots.push_back(pilot);
				    }
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
