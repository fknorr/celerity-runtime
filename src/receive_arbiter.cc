#include "receive_arbiter.h"
#include "grid.h"
#include "instruction_graph.h"

#include <exception>
#include <memory>

namespace celerity::detail {

receive_arbiter::receive_arbiter(communicator& comm) : m_comm(&comm) {}

receive_arbiter::~receive_arbiter() { assert(std::uncaught_exceptions() > 0 || m_transfers.empty()); }

bool receive_arbiter::event::is_complete() const { return region_intersection(m_region_transfer->complete_region, m_awaited_region) == m_awaited_region; }

void receive_arbiter::begin_receive(const transfer_id trid, const buffer_id bid, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	auto& transfer = m_transfers[{trid, bid}]; // allow default-insert

	assert(!transfer.elem_size.has_value() || *transfer.elem_size == elem_size);
	transfer.elem_size = elem_size;

	// There can be multiple begin_receives (i.e. disjoint allocations) for a single await_push, but allocations (currently) are not allowed to overlap.
	assert(std::all_of(transfer.regions.begin(), transfer.regions.end(),
	    [&](const std::unique_ptr<region_transfer>& t) { return box_intersection(t->allocated_bounding_box, allocated_box).empty(); }));

	const auto region_tr = transfer.regions.emplace_back(std::make_unique<region_transfer>(allocation, allocated_box)).get();
	const auto remaining_unassigned_pilots_end =
	    std::remove_if(transfer.unassigned_pilots.begin(), transfer.unassigned_pilots.end(), [&](const inbound_pilot& pilot) {
		    assert(allocated_box.covers(pilot.message.box) == !box_intersection(allocated_box, pilot.message.box).empty());
		    const auto now_assigned = allocated_box.covers(pilot.message.box);
		    if(now_assigned) { begin_receiving_fragment(region_tr, pilot, elem_size); }
		    return now_assigned;
	    });
	transfer.unassigned_pilots.erase(remaining_unassigned_pilots_end, transfer.unassigned_pilots.end());
}

receive_arbiter::event receive_arbiter::await_receive(const transfer_id trid, const buffer_id bid, const region<3>& awaited_region) {
	auto& transfer = m_transfers.at({trid, bid});
	const auto region_it = std::find_if(transfer.regions.begin(), transfer.regions.end(),
	    [&](const std::unique_ptr<region_transfer>& rt) { return rt->allocated_bounding_box.covers(bounding_box(awaited_region)); });
	assert(region_it != transfer.regions.end());
	return event(region_it->get(), awaited_region);
}

void receive_arbiter::end_receive(const transfer_id trid, const buffer_id bid) {
#if CELERITY_DETAIL_ENABLE_DEBUG
	auto& transfer = m_transfers.at(std::pair{trid, bid}); // must exist
	assert(transfer.unassigned_pilots.empty());
	assert(std::all_of(transfer.regions.begin(), transfer.regions.end(), //
	    [](const std::unique_ptr<region_transfer>& rt) { return rt->incoming_fragments.empty(); }));
#endif

	m_transfers.erase(std::pair{trid, bid});
}

void receive_arbiter::poll_communicator() {
	for(auto& [id, transfer] : m_transfers) {
		for(auto& region_tr : transfer.regions) {
			const auto incomplete_fragments_end =
			    std::remove_if(region_tr->incoming_fragments.begin(), region_tr->incoming_fragments.end(), [&](const incoming_fragment& fragment) {
				    const bool is_complete = fragment.done->is_complete();
				    if(is_complete) { region_tr->complete_region = region_union(region_tr->complete_region, fragment.box); }
				    return is_complete;
			    });
			region_tr->incoming_fragments.erase(incomplete_fragments_end, region_tr->incoming_fragments.end());
		}
	}

	for(const auto& pilot : m_comm->poll_inbound_pilots()) {
		auto& transfer = m_transfers[{pilot.message.trid, pilot.message.bid}]; // allow default-insert
		const auto region_it = std::find_if(transfer.regions.begin(), transfer.regions.end(),
		    [&](const std::unique_ptr<region_transfer>& rt) { return rt->allocated_bounding_box.covers(pilot.message.box); });
		if(region_it != transfer.regions.end()) {
			assert(region_intersection((*region_it)->complete_region, pilot.message.box).empty()); // must not receive the same datum twice
			begin_receiving_fragment(region_it->get(), pilot, transfer.elem_size.value());         // elem_size is set when transfer_region is inserted
		} else {
			transfer.unassigned_pilots.push_back(pilot);
		}
	}
}

void receive_arbiter::begin_receiving_fragment(region_transfer* const region_tr, const inbound_pilot& pilot, const size_t elem_size) {
	assert(region_tr->allocated_bounding_box.covers(pilot.message.box));
	const auto offset_in_allocation = pilot.message.box.get_offset() - region_tr->allocated_bounding_box.get_offset();
	const communicator::stride stride{
	    region_tr->allocated_bounding_box.get_range(),
	    subrange<3>{offset_in_allocation, pilot.message.box.get_range()},
	    elem_size,
	};
	region_tr->incoming_fragments.push_back({pilot.message.box, m_comm->receive_payload(pilot.from, pilot.message.tag, region_tr->allocation, stride)});
}

} // namespace celerity::detail
