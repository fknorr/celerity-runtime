#pragma once

#include "backend/backend.h"
#include "communicator.h"
#include "double_buffered_queue.h"
#include "instruction_graph.h"

namespace celerity::detail {

class instruction_executor final : private communicator::delegate {
  public:
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		virtual void instruction_checkpoint_reached(task_id tid) = 0;
	};

	instruction_executor(std::unique_ptr<backend::queue> backend_queue, const communicator_factory& comm_factory, delegate* dlg);
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor(instruction_executor&&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	instruction_executor& operator=(instruction_executor&&) = delete;
	~instruction_executor();

	void submit(const instruction& instr);

  private:
	struct completed_synchronous {};
	using event = std::variant<std::unique_ptr<backend::event>, std::unique_ptr<communicator::event>, completed_synchronous>;

	struct allocation {
		memory_id memory;
		void* pointer;
	};

	delegate* m_delegate;

	// accessed by by main and executor threads
	double_buffered_queue<const instruction*> m_submission_queue;

	// accessed by by main and communicator threads
	double_buffered_queue<pilot_message> m_pilot_queue;

	// accessed by executor thread only (unsynchronized)
	bool m_expecting_more_submissions = true;
	std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<allocation_id, allocation> m_allocations;
	std::unique_ptr<communicator> m_communicator;

	std::thread m_thread;

	void loop();

	event begin_executing(const instruction& instr);

	std::pair<void*, event> malloc(memory_id where, size_t size, size_t alignment);
	event free(memory_id where, void* allocation);

	void pilot_message_received(node_id from, const pilot_message& pilot) override;
};

} // namespace celerity::detail