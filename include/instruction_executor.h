#pragma once

#include "backend/backend.h"
#include "communicator.h"
#include "double_buffered_queue.h"
#include "instruction_graph.h"
#include "recv_arbiter.h"
#include "scheduler.h"

#include <unordered_map>

namespace celerity::detail {

struct host_object_instance;

class instruction_executor final : public abstract_scheduler::delegate {
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

	instruction_executor(std::unique_ptr<backend::queue> backend_queue, std::unique_ptr<communicator> comm, delegate* dlg);
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor(instruction_executor&&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	instruction_executor& operator=(instruction_executor&&) = delete;
	~instruction_executor();

	void submit_instruction(const instruction& instr) override;
	void submit_pilot(const outbound_pilot& pilot) override;

	void announce_buffer_user_pointer(buffer_id bid, const void* ptr);
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance);

  private:
	friend struct executor_testspy;

	struct completed_synchronous {};
	using event = std::variant<std::unique_ptr<backend::event>, std::unique_ptr<communicator::event>, recv_arbiter::event,
	    std::future<host_queue::execution_info>, completed_synchronous>;

	struct allocation {
		memory_id memory;
		void* pointer;
	};

	struct buffer_user_pointer_announcement {
		buffer_id bid;
		const void* ptr;
	};
	struct host_object_instance_announcement {
		host_object_id hoid;
		std::unique_ptr<host_object_instance> instance;
	};
	using submission = std::variant<const instruction*, outbound_pilot, buffer_user_pointer_announcement, host_object_instance_announcement>;

	// immutable
	delegate* m_delegate;
	std::unique_ptr<communicator> m_communicator;

	// accessed by by main and executor threads
	double_buffered_queue<submission> m_submission_queue;

	// accessed by executor thread only (unsynchronized)
	bool m_expecting_more_submissions = true;
	std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<buffer_id, const void*> m_buffer_user_pointers;
	std::unordered_map<allocation_id, allocation> m_allocations;
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> m_host_object_instances;
	recv_arbiter m_recv_arbiter;
	host_queue m_host_queue;

	std::thread m_thread;

	void loop();

	[[nodiscard]] event begin_executing(const instruction& instr);
};

} // namespace celerity::detail