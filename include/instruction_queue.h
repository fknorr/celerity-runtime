#pragma once

#include <memory>
#include <vector>

namespace celerity::detail {

class allocation_manager;
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

// Turns multiple in_order_instruction_queues into an out_of_order_instruction_queue.
// Use cases: Host-task "thread pool", CUDA concurrency.
// TODO is this actually optimal for any of these applications?
//   - For CUDA we might want to have separate Kernel / D2H / H2D copy streams for maximum utilization
//   - Async submissions do not really do anything for host code, we should rather submit these just in time to avoid needlessly blocking inside host threads to
//     wait for events from their sibling queues
class multiplex_instruction_queue : public out_of_order_instruction_queue {
  public:
	explicit multiplex_instruction_queue(std::vector<std::unique_ptr<in_order_instruction_queue>> in_order_queues);

	instruction_queue_event submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) override;

  private:
	std::vector<std::unique_ptr<in_order_instruction_queue>> m_inorder_queues;
	size_t m_round_robin_inorder_queue_index = 0;
};

} // namespace celerity::detail
