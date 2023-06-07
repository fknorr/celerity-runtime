#pragma once

#include "instruction_queue.h"

#include <memory>

namespace celerity::detail {

class host_instruction_queue final : public out_of_order_instruction_queue {
  public:
	explicit host_instruction_queue(size_t num_threads);
	host_instruction_queue(host_instruction_queue&&) = default;
	host_instruction_queue& operator=(host_instruction_queue&&) = default;
	~host_instruction_queue();

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override;

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
