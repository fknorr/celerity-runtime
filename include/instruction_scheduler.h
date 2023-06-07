#pragma once

#include "instruction_graph.h"
#include "instruction_queue.h"
#include "types.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace celerity::detail {

class instruction_scheduler {
  public:
	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate& operator=(const delegate&) = default;
		virtual ~delegate() = default;
		virtual instruction_queue_event submit_to_backend(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies);
	};

	enum class poll_action {
		poll_again,
		idle_until_next_submit,
	};

	explicit instruction_scheduler(delegate* delegate) : m_delegate(delegate) {}

	void submit(std::unique_ptr<instruction> instr);

	poll_action poll_events();

  private:
	struct pending_instruction_info {
		std::unique_ptr<instruction> instr;
		size_t num_inorder_dependencies;
		std::vector<instruction_id> inorder_submission_dependents;
		std::vector<instruction_id> inorder_completion_dependents;
	};
	struct active_instruction_info {
		instruction_backend backend;
		instruction_queue_event event;
		std::vector<instruction_id> inorder_completion_dependents;
	};
	struct finished_instruction_info {};
	using instruction_info = std::variant<pending_instruction_info, active_instruction_info, finished_instruction_info>;

	delegate* m_delegate;
	std::unordered_map<instruction_id, instruction_info> m_visible_instructions;      // TODO GC on effective epoch
	std::vector<std::pair<instruction_id, instruction_queue_event_impl*>> m_poll_set; // events are owned by m_visible_instructions

	void fulfill_dependency(const std::vector<instruction_id>& dependents, std::vector<instruction_info*>& out_ready_set);

	[[nodiscard]] instruction_queue_event submit_to_backend(std::unique_ptr<instruction> instr);
};

class async_instruction_scheduler {
  public:
	explicit async_instruction_scheduler(instruction_scheduler::delegate* delegate);
	async_instruction_scheduler(async_instruction_scheduler&&) = default;
	async_instruction_scheduler& operator=(async_instruction_scheduler&&) = default;
	~async_instruction_scheduler();

	void submit(std::unique_ptr<instruction> instr);

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
