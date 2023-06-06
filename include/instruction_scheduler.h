#pragma once

#include "instruction_graph.h"

namespace celerity::detail {

class allocation_manager {
  public:
	void begin_tracking(const allocation_id aid, void* const ptr) {
		std::lock_guard lock(m_mutex);
		const auto [_, inserted] = m_pointers.emplace(aid, ptr);
		assert(inserted);
	}

	void end_tracking(const allocation_id aid) {
		std::lock_guard lock(m_mutex);
		const auto erased = m_pointers.erase(aid);
		assert(erased == 1);
	}

	void* get_pointer(const allocation_id aid) const {
		std::lock_guard lock(m_mutex);
		return m_pointers.at(aid);
	}

  private:
	mutable std::mutex m_mutex;
	std::unordered_map<allocation_id, void*> m_pointers;
};

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

class instruction_scheduler {
  public:
	instruction_scheduler(std::unordered_map<device_id, int> cuda_device_ids, std::unordered_map<device_id, sycl::queue> sycl_queues);
	~instruction_scheduler();
	instruction_scheduler(instruction_scheduler&&) = default;
	instruction_scheduler& operator=(instruction_scheduler&&) = default;

	void submit(std::unique_ptr<instruction> instr);

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
