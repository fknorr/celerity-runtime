#pragma once

#include "instruction_queue.h"

#include <mpi.h>

namespace celerity::detail {

class mpi_instruction_queue final : public out_of_order_instruction_queue {
  public:
	explicit mpi_instruction_queue(MPI_Comm comm, const allocation_manager &am);

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override;

  private:
  struct impl;
  std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
