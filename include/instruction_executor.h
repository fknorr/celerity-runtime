#pragma once

#include "backend/backend.h"
#include "communicator.h"
#include "double_buffered_queue.h"
#include "instruction_graph.h"
#include "recv_arbiter.h"

#include <unordered_map>

namespace celerity::detail {

struct host_object_instance;

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
		virtual void horizon_reached(task_id tid) = 0;
		virtual void epoch_reached(task_id tid) = 0;
	};

	instruction_executor(std::unique_ptr<backend::queue> backend_queue, const communicator_factory& comm_factory, delegate* dlg);
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor(instruction_executor&&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	instruction_executor& operator=(instruction_executor&&) = delete;
	~instruction_executor();

	void submit(const instruction& instr);

	void announce_buffer_user_pointer(buffer_id bid, const void* ptr);
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance);

  private:
	struct completed_synchronous {};
	using event = std::variant<std::unique_ptr<backend::event>, std::unique_ptr<communicator::event>, recv_arbiter::event,
	    std::future<host_queue::execution_info>, completed_synchronous>;

	struct allocation {
		memory_id memory;
		void* pointer;
	};

	using submission = std::variant<const instruction*, std::pair<buffer_id, const void*>>;

	delegate* m_delegate;

	// accessed by by main and executor threads
	double_buffered_queue<submission> m_submission_queue;

	// accessed by by main and communicator threads
	double_buffered_queue<std::pair<node_id, pilot_message>> m_pilot_queue;

	// accessed by executor thread only (unsynchronized)
	bool m_expecting_more_submissions = true;
	std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<buffer_id, const void*> m_buffer_user_pointers;
	std::unordered_map<allocation_id, allocation> m_allocations;
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> m_host_object_instances;
	std::unique_ptr<communicator> m_communicator;
	recv_arbiter m_recv_arbiter;
	host_queue m_host_queue;

	std::thread m_thread;

	void loop();

	[[nodiscard]] event begin_executing(const instruction& instr);

	void pilot_message_received(node_id from, const pilot_message& pilot) override;
};

} // namespace celerity::detail