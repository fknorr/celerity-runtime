#pragma once

#include "allocation_manager.h"
#include "instruction_queue.h"
#include "instruction_scheduler.h"
#include "utils.h"

namespace celerity::detail {

class instruction_executor final : private instruction_scheduler::delegate {
  public:
	using device_queue_map = std::unordered_map<std::pair<device_id, instruction_backend>, std::unique_ptr<out_of_order_instruction_queue>, utils::pair_hash>;

	instruction_executor(
	    std::unique_ptr<allocation_manager> alloc_manager, std::unique_ptr<out_of_order_instruction_queue> host_queue, device_queue_map device_queues);
	
	void submit(std::unique_ptr<instruction> instr);

  private:
	// only accessed by instruction scheduler thread
	std::unique_ptr<allocation_manager> m_alloc_manager;
	std::unique_ptr<out_of_order_instruction_queue> m_host_queue;
	device_queue_map m_device_queues;

	async_instruction_scheduler m_scheduler;

	out_of_order_instruction_queue* select_backend_queue(const instruction& isntr);
	out_of_order_instruction_queue* select_backend_queue(const instruction_backend backend, const std::initializer_list<memory_id>& mids);
	out_of_order_instruction_queue* select_backend_queue(const instruction_backend backend, const device_id did);

	instruction_queue_event submit_to_backend(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) override;
};

} // namespace celerity::detail
