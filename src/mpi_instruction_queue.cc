#include "mpi_instruction_queue.h"

#include "allocation_manager.h"
#include "instruction_graph.h"
#include "recv_arbiter.h"
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

struct mpi_instruction_queue::impl final : recv_arbiter::delegate {
	MPI_Comm comm;
	const allocation_manager* alloc_mgr;

	std::mutex mutex;
	recv_arbiter arbiter;
	bool shutdown = false;

	impl(const MPI_Comm comm, const allocation_manager& am) : comm(comm), alloc_mgr(&am), arbiter(this) {}
	instruction_queue_event submit(std::unique_ptr<instruction> instr);

	instruction_queue_event begin_send(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id dest, int tag);
	instruction_queue_event begin_recv(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id source, int tag) override;
};

instruction_queue_event mpi_instruction_queue::impl::submit(std::unique_ptr<instruction> instr) {
	return utils::match(
	    *instr, //
	    [&](const send_instruction& sinstr) {
		    return begin_send(sinstr.get_allocation_id(), 0 /* offset_bytes */, sinstr.get_size_bytes(), sinstr.get_dest_node_id(), sinstr.get_tag());
	    },
	    [&](const recv_instruction& rinstr) { //
		    return arbiter.submit_recv(rinstr);
	    },
	    [&](const auto&) -> instruction_queue_event { //
		    panic("Invalid instruction type on mpi_instruction_queue");
	    });
}

instruction_queue_event mpi_instruction_queue::impl::begin_send(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id dest, int tag) {
	void* const pointer = static_cast<std::byte*>(alloc_mgr->get_pointer(aid)) + offset_bytes;
	MPI_Request request;
	MPI_Isend(pointer, size_bytes, MPI_BYTE, static_cast<int>(dest), tag, comm, &request);
	return std::make_shared<mpi_instruction_queue_event>(request);
}

instruction_queue_event mpi_instruction_queue::impl::begin_recv(allocation_id aid, size_t offset_bytes, size_t size_bytes, node_id source, int tag) {
	void* const pointer = static_cast<std::byte*>(alloc_mgr->get_pointer(aid)) + offset_bytes;
	MPI_Request request;
	MPI_Irecv(pointer, size_bytes, MPI_BYTE, static_cast<int>(source), tag, comm, &request);
	return std::make_shared<mpi_instruction_queue_event>(request);
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
