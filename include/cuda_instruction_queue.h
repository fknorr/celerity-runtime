#pragma once

#include "instruction_queue.h"

namespace celerity::detail {

class cuda_instruction_queue final : public multiplex_instruction_queue {
  public:
	cuda_instruction_queue(int cuda_device_id, size_t num_streams, allocation_manager& am);

	// TODO consider moving CUDA async calls into a thread pool to reduce latency. Can this re-use the same thread pool that we use for the
	// host_instruction_queue, so that we don't spawn too many threads and overuse resources due to CPU affinity?
};

} // namespace celerity::detail
