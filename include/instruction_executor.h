#pragma once

#include "backend/backend.h"
#include "instruction_graph.h"


namespace celerity::detail {

class instruction_executor {
  public:
	instruction_executor();
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	~instruction_executor();

	void submit(const instruction& instr);

  private:
    struct completed_synchronous{};
	using event = std::variant<sycl::event, std::unique_ptr<backend::queue::event>, completed_synchronous>;

	struct allocation {
		memory_id memory;
		void* pointer;
	};

    // accessed by by both threads
	std::mutex m_submission_mutex;
	std::vector<const instruction*> m_submission_queue;
	std::atomic<bool> m_submission_queue_nonempty;

    // accessed by executor thread only (unsynchronized)
    std::unordered_map<device_id, sycl::queue *> m_sycl_queues;
    std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<allocation_id, allocation> m_allocations;

	std::thread m_thread;

	void loop();

	event begin_executing(const instruction& instr);
};

} // namespace celerity::detail