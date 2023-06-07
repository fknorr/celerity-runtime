#include "mpi_instruction_queue.h"

#include "allocation_manager.h"
#include "instruction_graph.h"
#include "utils.h"
#include <mpi.h>

namespace celerity::detail {

class mpi_instruction_queue_event : public instruction_queue_event_impl {
  public:
	explicit mpi_instruction_queue_event(MPI_Request req) : m_request(req) {}

	bool has_completed() const override {
		int flag;
		MPI_Test(&m_request, &flag, MPI_STATUS_IGNORE);
		return flag != 0;
	}

	void block_on() override;

  private:
	mutable MPI_Request m_request;
};

struct mpi_instruction_queue::impl {
	MPI_Comm comm;
	const allocation_manager* alloc_mgr;

  std::mutex mutex;
  std::vector<pilot_message> pilot_backlog;
  std::vector<std::unique_ptr<recv_instruction>> recv_instruction_backlog;
  bool shutdown = false;

	impl(const MPI_Comm comm, const allocation_manager& am) : comm(comm), alloc_mgr(&am) {}
	instruction_queue_event submit(std::unique_ptr<instruction> instr);
	void thread_main();
};

instruction_queue_event mpi_instruction_queue::impl::submit(std::unique_ptr<instruction> instr) {
	MPI_Request req;
	utils::match(
	    *instr, //
	    [&](const send_instruction& sinstr) {
		    MPI_Isend(alloc_mgr->get_pointer(sinstr.get_allocation_id()), sinstr.get_size_bytes(), MPI_BYTE, static_cast<int>(sinstr.get_dest_node_id()),
		        sinstr.get_tag(), comm, &req);
	    },
	    [&](const recv_instruction& rinstr) {
		    // TODO if I have a pilot for this receive, do an MPI_Irecv in this thread, otherwise push the recv into the backlog
		    // TODO when I don't know the pilot, how can I generate the request for returning an event?
	    },
	    [&](const auto&) { panic("Invalid instruction type on mpi_instruction_queue"); });

	return std::make_shared<mpi_instruction_queue_event>(req);
}

void mpi_instruction_queue::impl::thread_main() {
  std::unique_lock lock(mutex);
	while (!shutdown) {
		// TODO MPI_Recv a pilot message
		// - if I know the corresponding recv_instruction, do an MPI_Irecv right here
		// - if there is no corresponding recv_instruction, push the pilot into the backlog
    // TODO the recv probably cannot be blocking, otherwise I would miss the thread shutdown flag
    // TODO should we suspend the thread when there is no recv_instruction_backlog? If we do that, MPI would buffer the incoming pilots for us.
	}
}

mpi_instruction_queue::mpi_instruction_queue(const MPI_Comm comm, const allocation_manager& am) : m_impl(std::make_unique<impl>(comm, am)) {}

instruction_queue_event mpi_instruction_queue::submit(std::unique_ptr<instruction> instr) { return m_impl->submit(std::move(instr)); }

} // namespace celerity::detail
