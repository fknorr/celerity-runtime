#pragma once

#include "instruction_queue.h"

#include <mpi.h>

namespace celerity::detail {

class recv_instruction;
struct pilot_message;

class mpi_instruction_queue final : public out_of_order_instruction_queue {
  public:
	explicit mpi_instruction_queue(MPI_Comm comm, const allocation_manager& am);

  // call as soon as a recv_instruction is generated, even before its dependencies have been fulfilled.
	void prepare(const recv_instruction& rinstr);

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override;

  // call from the executor loop.
  void poll_incoming_messages();

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
