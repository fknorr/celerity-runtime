#pragma once

#include "instruction_queue.h"
#include "workaround.h"

namespace celerity::detail {

class sycl_instruction_queue final : public graph_order_instruction_queue {
  public:
	explicit sycl_instruction_queue(sycl::queue q, allocation_manager& am);

	instruction_queue_event submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) override;

  private:
	sycl::queue m_queue;
	allocation_manager* m_allocation_mgr;
};

} // namespace celerity::detail
