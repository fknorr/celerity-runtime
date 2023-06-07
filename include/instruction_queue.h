#pragma once

#include <memory>
#include <vector>

namespace celerity::detail {

class instruction;

class instruction_queue_event_impl {
  public:
	virtual ~instruction_queue_event_impl() = default;
	virtual bool has_completed() const = 0;
	virtual void block_on() = 0;
};

using instruction_queue_event = std::shared_ptr<instruction_queue_event_impl>;

class in_order_instruction_queue {
  public:
	virtual ~in_order_instruction_queue() = default;
	virtual instruction_queue_event submit(std::unique_ptr<instruction> instr) = 0;
	virtual void wait_on(const instruction_queue_event& evt) = 0;
};

class out_of_order_instruction_queue {
  public:
	virtual ~out_of_order_instruction_queue() = default;
	virtual instruction_queue_event submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) = 0;
};

} // namespace celerity::detail
