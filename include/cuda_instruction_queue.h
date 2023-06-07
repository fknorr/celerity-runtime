#pragma once

#include "instruction_queue.h"

namespace celerity::detail {

class cuda_instruction_queue: public multiplex_instruction_queue {
  public:
	cuda_instruction_queue(int cuda_device_id, size_t num_streams, allocation_manager &am);
};

} // namespace celerity::detail
