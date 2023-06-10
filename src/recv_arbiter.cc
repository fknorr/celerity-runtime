#include "recv_arbiter.h"

#include "grid.h"
#include "instruction_graph.h"
#include "instruction_queue.h"
#include "ranges.h"
#include "utils.h"

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <variant>

namespace celerity::detail {

class deferred_instruction_queue_event : public instruction_queue_event_impl {
  public:
	bool has_completed() const override {
		std::lock_guard lock(m_mutex);
		if(m_event == nullptr) return false;
		return m_event->has_completed();
	}

	void block_on() override {
		std::unique_lock lock(m_mutex);
		while(m_event == nullptr) {
			m_event_set.wait(lock);
		}
		m_event->block_on();
	}

	void set_event(instruction_queue_event event) {
		{
			std::lock_guard lock(m_mutex);
			m_event = std::move(event);
		}
		m_event_set.notify_all();
	}

  private:
	mutable std::mutex m_mutex;
	instruction_queue_event m_event;
	std::condition_variable m_event_set;
};

// TODO this is a draft implementation - needs significant rework.
// - there will usually be multiple pilot messages (at least one per source node) for each recv_instruction. We need to track the partial state of having 0 or 1
// recv_instruction and any number of pilot messages, until the merged region of pilot boxes matches the entire recv.
// - if we receive all pilots in advance, there can be a trade-off between minimizing the number of allocations and host-to-device copies, and overlapping
// receives and copies as much as possible.
// - the default should arguably be to receive every box into its own staging buffer, and doing RDMA recvs as the fast path where possible. The temporary
// allocation mechanism will probably bypass the allocation_manager since there won't be any benefit in keeping that indirection (other than testing?)
// - there should not really be a need to prepare_recv() before submit_recv() unless this allows us to perform some really expensive layout or schedule in
// advance. At least this is true as long as recvs do not overwrite data that is read by a (proximate) predecessor instruction, in which case we could
// reliably decide early on which recvs need staging buffers. This again needs to be handled with care to avoid pre-allocating (and potentially exhausting)
// memory too early.
struct recv_arbiter::impl {
	using deferred_event_ptr = std::shared_ptr<deferred_instruction_queue_event>;
	using await_push_id = std::pair<buffer_id, transfer_id>;

	struct dest_buffer_allocation {
		memory_id memory;
		allocation_id allocation;
		int dims;
		range<3> alloc_range;
		id<3> offset_in_alloc;
		id<3> offset_in_buffer;
		range<3> recv_range;
		size_t elem_size;

		friend bool operator==(const dest_buffer_allocation& lhs, const dest_buffer_allocation& rhs) {
			return lhs.memory == rhs.memory && lhs.allocation == rhs.allocation && lhs.dims == rhs.dims && lhs.alloc_range == rhs.alloc_range
			       && lhs.offset_in_alloc == rhs.offset_in_alloc && lhs.offset_in_buffer == rhs.offset_in_buffer && lhs.recv_range == rhs.recv_range
			       && lhs.elem_size == rhs.elem_size;
		}
		friend bool operator!=(const dest_buffer_allocation& lhs, const dest_buffer_allocation& rhs) { return !(lhs == rhs); }
	};

	struct metadata {
		GridBox<3> recv_box;
		node_id source;
		int tag;

		friend bool operator==(const metadata& lhs, const metadata& rhs) {
			return lhs.recv_box == rhs.recv_box && lhs.source == rhs.source && lhs.tag == rhs.tag;
		}
		friend bool operator!=(const metadata& lhs, const metadata& rhs) { return !(lhs == rhs); }
	};

	struct destination_known {
		dest_buffer_allocation dest;
		deferred_event_ptr ready_instruction_event; // null until instruction is ready
	};

	struct metadata_known {
		metadata metadata;
	};

	struct pending_immediate_recv {
		allocation_id allocation;
		size_t offset_bytes;
		size_t size_bytes;
		node_id source;
		int tag;

		friend bool operator==(const pending_immediate_recv& lhs, const pending_immediate_recv& rhs) {
			return lhs.allocation == rhs.allocation && lhs.offset_bytes == rhs.offset_bytes && lhs.size_bytes == rhs.size_bytes && lhs.source == rhs.source
			       && lhs.tag == rhs.tag;
		}
		friend bool operator!=(const pending_immediate_recv& lhs, const pending_immediate_recv& rhs) { return !(lhs == rhs); }
	};

	struct pending_staged_recv {
		dest_buffer_allocation dest;
		metadata metadata;
		allocation_id stage_allocation;

		friend bool operator==(const pending_staged_recv& lhs, const pending_staged_recv& rhs) {
			return lhs.dest == rhs.dest && lhs.metadata == rhs.metadata && lhs.stage_allocation == rhs.stage_allocation;
		}
		friend bool operator!=(const pending_staged_recv& lhs, const pending_staged_recv& rhs) { return !(lhs == rhs); }
	};

	using pending_recv = std::variant<pending_immediate_recv, pending_staged_recv>;
	using backlog_entry = std::variant<destination_known, metadata_known, pending_recv>;

	delegate* m_delegate;
	std::unordered_map<await_push_id, backlog_entry, utils::pair_hash> m_backlog;

	impl(delegate* delegate) : m_delegate(delegate) {}

	void push_recv_instruction(const recv_instruction& rinstr, const deferred_event_ptr& ready_instruction_event);
	void accept_pilot(const node_id source, const pilot_message& pilot);

	dest_buffer_allocation make_dest_buffer_allocation(const recv_instruction& rinstr) {
		return {rinstr.get_dest_memory_id(), rinstr.get_dest_allocation_id(), rinstr.get_dimensions(), rinstr.get_allocation_range(),
		    rinstr.get_offset_in_allocation(), rinstr.get_offset_in_buffer(), rinstr.get_recv_range(), rinstr.get_element_size()};
	}
	metadata make_metadata(const pilot_message& pilot, const node_id source) { return {pilot.box, source, pilot.tag}; }

	pending_recv create_pending_recv(const dest_buffer_allocation& dest, const metadata& meta);
	void dispatch_pending_recv(const pending_recv& recv, const deferred_event_ptr& deferred_event);
};

recv_arbiter::impl::pending_recv recv_arbiter::impl::create_pending_recv(const dest_buffer_allocation& dest, const metadata& meta) {
	const auto [recv_offset_in_buffer, recv_range] = grid_box_to_subrange(meta.recv_box);
	assert((recv_offset_in_buffer >= dest.offset_in_buffer) == id<3>(true, true, true));
	assert((recv_range <= dest.alloc_range) == id<3>(true, true, true));

	if((dest.memory == host_memory_id /* || TODO or MPI has DMA support for dest memory */)             //
	    && recv_offset_in_buffer[1] == dest.offset_in_buffer[1] && recv_range[1] == dest.alloc_range[1] //
	    && recv_offset_in_buffer[2] == dest.offset_in_buffer[2] && recv_range[2] == dest.alloc_range[2]) {
		const auto recv_offset_in_allocation = recv_offset_in_buffer - dest.offset_in_buffer + dest.offset_in_alloc;
		const auto recv_offset_bytes = recv_offset_in_allocation[0] * dest.elem_size;
		const auto recv_size_bytes = recv_range.size() * dest.elem_size;
		return pending_immediate_recv{dest.allocation, recv_offset_bytes, recv_size_bytes, meta.source, meta.tag};
	} else {
		// TODO this needs to either
		// a) generate a mini-IDAG for alloc -> recv -> copy -> free
		// b) rely on (not yet thought out) conditional nodes in the IDAG that are enabled when this branch is taken
		// Option b) would also let the IDAG generator assign the allocation ids.
		// Maybe this is best implemented by allowing branches / generic conditional nodes in the IDAG?
		//   - if we do that, we can signal the branch condition here to begin the allocation
		const auto stage = allocation_id(-1); // TODO
		return pending_staged_recv{dest, meta, stage};
	}
}

void recv_arbiter::impl::dispatch_pending_recv(const pending_recv& recv, const deferred_event_ptr& deferred_event) {
	assert(deferred_event != nullptr);
	auto event = utils::match(
	    recv, //
	    [&](const pending_immediate_recv& immediate) {
		    return m_delegate->begin_recv(immediate.allocation, immediate.offset_bytes, immediate.size_bytes, immediate.source, immediate.tag);
	    },
	    [&](const pending_staged_recv& staged) {
		    const auto recv_offset_bytes = 0;
		    const auto recv_size_bytes = staged.metadata.recv_box.area() * staged.dest.elem_size;
		    return m_delegate->begin_recv(staged.stage_allocation, recv_offset_bytes, recv_size_bytes, staged.metadata.source, staged.metadata.tag);
		    // TODO dispatch / unblock copy + free instructions
	    });
	deferred_event->set_event(std::move(event));
}

void recv_arbiter::impl::push_recv_instruction(const recv_instruction& rinstr, const deferred_event_ptr& ready_instruction_event) {
	const auto key = await_push_id{rinstr.get_buffer_id(), rinstr.get_transfer_id()};
	const auto it = m_backlog.find(key);
	if(it == m_backlog.end()) {
		m_backlog.emplace(key, destination_known{make_dest_buffer_allocation(rinstr), ready_instruction_event});
		return;
	}

	auto& entry = it->second;
	utils::match(
	    entry, //
	    [&](destination_known& dk) {
		    assert(dk.dest == make_dest_buffer_allocation(rinstr));
		    assert(dk.ready_instruction_event == nullptr || dk.ready_instruction_event == ready_instruction_event);
		    dk.ready_instruction_event = ready_instruction_event;
	    },
	    [&](const metadata_known& mk) {
		    const auto pending = create_pending_recv(make_dest_buffer_allocation(rinstr), mk.metadata);
		    if(ready_instruction_event) {
			    dispatch_pending_recv(pending, ready_instruction_event);
			    m_backlog.erase(it);
		    } else {
			    entry = pending;
		    }
	    },
	    [&](const pending_recv& pending) {
		    if(ready_instruction_event) {
			    dispatch_pending_recv(pending, ready_instruction_event);
			    m_backlog.erase(it);
		    }
	    });
}

void recv_arbiter::impl::accept_pilot(const node_id source, const pilot_message& pilot) {
	const auto key = await_push_id{pilot.buffer, pilot.transfer};
	const auto it = m_backlog.find(key);
	if(it == m_backlog.end()) {
		m_backlog.emplace(key, metadata_known{make_metadata(pilot, source)});
		return;
	}

	auto& entry = it->second;
	utils::match(
	    entry, //
	    [&](destination_known& dk) {
		    const auto pending = create_pending_recv(dk.dest, make_metadata(pilot, source));
		    if(dk.ready_instruction_event) {
			    dispatch_pending_recv(pending, dk.ready_instruction_event);
			    m_backlog.erase(it);
		    } else {
			    entry = pending;
		    }
	    },
	    [&](const metadata_known& mk) { //
		    assert(mk.metadata == make_metadata(pilot, source));
	    },
	    [](const pending_recv&) {
		    // we could assert that pilot data is consistent with whatever pending_recv variant we see here
	    });
}

recv_arbiter::recv_arbiter(delegate* delegate) : m_impl(std::make_unique<impl>(delegate)) {}

recv_arbiter::~recv_arbiter() = default;

void recv_arbiter::prepare_recv(const recv_instruction& rinstr) { m_impl->push_recv_instruction(rinstr, nullptr); }

instruction_queue_event recv_arbiter::submit_recv(const recv_instruction& rinstr) {
	auto event = std::make_shared<deferred_instruction_queue_event>();
	m_impl->push_recv_instruction(rinstr, event);
	return event;
}

void recv_arbiter::accept_pilot(const node_id source, const pilot_message& pilot) { m_impl->accept_pilot(source, pilot); }

} // namespace celerity::detail
