#include "mpi_instruction_queue.h"

#include "allocation_manager.h"
#include "instruction_graph.h"
#include "utils.h"
#include <mpi.h>

namespace celerity::detail {

class mpi_instruction_queue_event : public instruction_queue_event_impl {
  public:
	mpi_instruction_queue_event(const MPI_Request request) : m_request(request) {}

	bool has_completed() const override {
		std::lock_guard lock(m_mutex);
		int flag;
		MPI_Test(&m_request, &flag, MPI_STATUS_IGNORE);
		return flag != 0;
	}

	void block_on() override {
		std::lock_guard lock(m_mutex);
		MPI_Wait(&m_request, MPI_STATUS_IGNORE);
	}

  private:
	mutable std::mutex m_mutex;
	mutable MPI_Request m_request;
};

class deferred_mpi_instruction_queue_event : public instruction_queue_event_impl {
  public:
	bool has_completed() const override {
		std::lock_guard lock(m_mutex);
		if(!m_request.has_value()) return false;
		int flag;
		MPI_Test(&*m_request, &flag, MPI_STATUS_IGNORE);
		return flag != 0;
	}

	void block_on() override {
		std::unique_lock lock(m_mutex);
		while(!m_request.has_value()) {
			m_completed.wait(lock);
		}
		MPI_Wait(&*m_request, MPI_STATUS_IGNORE);
	}

	void set_request(const MPI_Request request) {
		{
			std::lock_guard lock(m_mutex);
			assert(!m_request.has_value());
			m_request = request;
		}
		m_completed.notify_all();
	}

  private:
	mutable std::mutex m_mutex;
	mutable std::optional<MPI_Request> m_request;
	std::condition_variable m_completed;
};

class recv_arbiter {
  public:
	using deferred_event_ptr = std::shared_ptr<deferred_mpi_instruction_queue_event>;

	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = delete;
		delegate& operator=(const delegate&) = delete;
		virtual ~delegate() = default;

		virtual MPI_Request begin_recv(allocation_id dest, size_t offset_bytes, size_t size_bytes, node_id source, int tag) = 0;
		// TODO how to begin staged recvs?
	};

	explicit recv_arbiter(delegate* delegate) : m_delegate(delegate) {}

	void prepare_recv(const recv_instruction& rinstr);
	[[nodiscard]] instruction_queue_event submit_recv(const recv_instruction& rinstr);
	void accept_pilot(const node_id source, const pilot_message& pilot);

  private:
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

	void push_recv_instruction(const recv_instruction& rinstr, const deferred_event_ptr& ready_instruction_event);

	await_push_id get_await_push_id(const recv_instruction& rinstr) { return {rinstr.get_buffer_id(), rinstr.get_transfer_id()}; }
	await_push_id get_await_push_id(const pilot_message& pilot) { return {pilot.buffer, pilot.transfer}; }

	dest_buffer_allocation make_dest_buffer_allocation(const recv_instruction& rinstr) {
		return {rinstr.get_dest_memory_id(), rinstr.get_dest_allocation_id(), rinstr.get_dimensions(), rinstr.get_allocation_range(),
		    rinstr.get_offset_in_allocation(), rinstr.get_offset_in_buffer(), rinstr.get_recv_range(), rinstr.get_element_size()};
	}
	metadata make_metadata(const pilot_message& pilot, const node_id source) { return {pilot.box, source, pilot.tag}; }

	pending_recv create_pending_recv(const dest_buffer_allocation& dest, const metadata& meta);
	void dispatch_pending_recv(const pending_recv& recv, const deferred_event_ptr& event);
};

void recv_arbiter::prepare_recv(const recv_instruction& rinstr) { push_recv_instruction(rinstr, nullptr); }

instruction_queue_event recv_arbiter::submit_recv(const recv_instruction& rinstr) {
	auto event = std::make_shared<deferred_mpi_instruction_queue_event>();
	push_recv_instruction(rinstr, event);
	return event;
}

recv_arbiter::pending_recv recv_arbiter::create_pending_recv(const dest_buffer_allocation& dest, const metadata& meta) {
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
		// b) rely on (not yet thougth out) conditional nodes in the IDAG that are enabled when this branch is taken
		// Option b) would also let the IDAG generator assign the allocation ids.
		// Maybe this is best implemented by allowing branches / generic conditional nodes in the IDAG?
		//   - if we do that, we can signal the branch condition here to begin the allocation
		const auto stage = allocation_id(-1); // TODO
		return pending_staged_recv{dest, meta, stage};
	}
}

void recv_arbiter::dispatch_pending_recv(const pending_recv& recv, const deferred_event_ptr& event) {
	assert(event != nullptr);
	const auto request = utils::match(
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
	event->set_request(request);
}

void recv_arbiter::push_recv_instruction(const recv_instruction& rinstr, const deferred_event_ptr& ready_instruction_event) {
	const auto key = get_await_push_id(rinstr);
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

void recv_arbiter::accept_pilot(const node_id source, const pilot_message& pilot) {
	const auto key = get_await_push_id(pilot);
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

struct mpi_instruction_queue::impl : recv_arbiter::delegate {
	MPI_Comm comm;
	const allocation_manager* alloc_mgr;

	std::mutex mutex;
	recv_arbiter arbiter;
	bool shutdown = false;

	impl(const MPI_Comm comm, const allocation_manager& am) : comm(comm), alloc_mgr(&am), arbiter(this) {}
	instruction_queue_event submit(std::unique_ptr<instruction> instr);

	MPI_Request begin_send(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id dest, int tag);
	MPI_Request begin_recv(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id source, int tag) override;
};

instruction_queue_event mpi_instruction_queue::impl::submit(std::unique_ptr<instruction> instr) {
	return utils::match(
	    *instr, //
	    [&](const send_instruction& sinstr) {
		    const auto req = begin_send(sinstr.get_allocation_id(), 0 /* offset_bytes */, sinstr.get_size_bytes(), sinstr.get_dest_node_id(), sinstr.get_tag());
		    return instruction_queue_event(std::make_shared<mpi_instruction_queue_event>(req));
	    },
	    [&](const recv_instruction& rinstr) { //
		    return arbiter.submit_recv(rinstr);
	    },
	    [&](const auto&) -> instruction_queue_event { //
		    panic("Invalid instruction type on mpi_instruction_queue");
	    });
}

MPI_Request mpi_instruction_queue::impl::begin_send(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id dest, int tag) {
	void* const pointer = static_cast<std::byte*>(alloc_mgr->get_pointer(aid)) + offset_bytes;
	MPI_Request request;
	MPI_Isend(pointer, size_bytes, MPI_BYTE, static_cast<int>(dest), tag, comm, &request);
	return request;
}

MPI_Request mpi_instruction_queue::impl::begin_recv(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id source, int tag) {
	void* const pointer = static_cast<std::byte*>(alloc_mgr->get_pointer(aid)) + offset_bytes;
	MPI_Request request;
	MPI_Irecv(pointer, size_bytes, MPI_BYTE, static_cast<int>(source), tag, comm, &request);
	return request;
}

mpi_instruction_queue::mpi_instruction_queue(const MPI_Comm comm, const allocation_manager& am) : m_impl(std::make_unique<impl>(comm, am)) {}

void mpi_instruction_queue::prepare(const recv_instruction& rinstr) { m_impl->arbiter.prepare_recv(rinstr); }

instruction_queue_event mpi_instruction_queue::submit(std::unique_ptr<instruction> instr) { return m_impl->submit(std::move(instr)); }

void mpi_instruction_queue::poll_incoming_messages() {
	constexpr int pilot_tag = 1;
	int flag;
	MPI_Status status;
	MPI_Iprobe(MPI_ANY_SOURCE, pilot_tag, m_impl->comm, &flag, &status);

	if(flag != 0) {
		pilot_message pilot;
		MPI_Recv(&pilot, sizeof(pilot), MPI_BYTE, status.MPI_SOURCE, pilot_tag, m_impl->comm, MPI_STATUS_IGNORE);
		m_impl->arbiter.accept_pilot(static_cast<node_id>(status.MPI_SOURCE), pilot);
	}
}

} // namespace celerity::detail
