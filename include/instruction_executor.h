#pragma once

#include "backend/backend.h"
#include "instruction_graph.h"

namespace celerity::detail {

class instruction_executor {
  public:
	explicit instruction_executor(std::unique_ptr<backend::queue> backend_queue);
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	~instruction_executor();

	void submit(const instruction& instr);

  private:
	struct completed_synchronous {};
	using event = std::variant<std::unique_ptr<backend::event>, completed_synchronous>;

	struct allocation {
		memory_id memory;
		void* pointer;
	};

	// accessed by by both threads
	std::mutex m_submission_mutex;
	std::vector<const instruction*> m_submission_queue;
	std::atomic<bool> m_submission_queue_nonempty;

	// accessed by executor thread only (unsynchronized)
	std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<allocation_id, allocation> m_allocations;

	std::thread m_thread;

	void loop();

	event begin_executing(const instruction& instr);

	std::pair<void*, event> malloc(memory_id where, size_t size, size_t alignment);
	event free(memory_id where, void* allocation);
};

} // namespace celerity::detail