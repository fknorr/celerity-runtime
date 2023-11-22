#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include "distributed_graph_generator.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"


namespace celerity {
namespace detail {

	class command_graph;
	class command_recorder;
	class instruction;
	class instruction_executor;
	class instruction_graph;
	class instruction_recorder;
	struct outbound_pilot;
	class task;

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
		friend struct scheduler_testspy;

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
			virtual void submit_instruction(const instruction& instr) = 0;
			virtual void submit_pilot(const outbound_pilot& pilot) = 0;
		};

		struct policy_set {
			distributed_graph_generator::policy_set command_graph_generator;
			instruction_graph_generator::policy_set instruction_graph_generator;
		};

		abstract_scheduler(size_t num_nodes, node_id local_node_id, instruction_graph_generator::system_info system_info, const task_manager& tm,
		    delegate* delegate, command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});
		abstract_scheduler(const abstract_scheduler&) = delete;
		abstract_scheduler(abstract_scheduler&&) = delete;
		abstract_scheduler& operator=(const abstract_scheduler&) = delete;
		abstract_scheduler& operator=(abstract_scheduler&&) = delete;

		virtual ~abstract_scheduler();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_created(
		    const buffer_id bid, const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const bool host_initialized) {
			notify(event_buffer_created{bid, dims, range, elem_size, elem_align, host_initialized});
		}

		void set_buffer_debug_name(const buffer_id bid, const std::string& name) { notify(event_set_buffer_debug_name{bid, name}); }

		void notify_buffer_destroyed(const buffer_id bid) { notify(event_buffer_destroyed{bid}); }

		void notify_host_object_created(const host_object_id hoid, const bool owns_instance) { notify(event_host_object_created{hoid, owns_instance}); }

		void notify_host_object_destroyed(const host_object_id hoid) { notify(event_host_object_destroyed{hoid}); }

		void notify_epoch_reached(const task_id tid) { notify(event_epoch_reached{tid}); }

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

	  private:
		struct event_task_available {
			const task* tsk;
		};
		struct event_buffer_created {
			buffer_id bid;
			int dims;
			celerity::range<3> range;
			size_t elem_size;
			size_t elem_align;
			bool host_initialized;
		};
		struct event_set_buffer_debug_name {
			buffer_id bid;
			std::string debug_name;
		};
		struct event_buffer_destroyed {
			buffer_id bid;
		};
		struct event_host_object_created {
			host_object_id hoid;
			bool owns_instance;
		};
		struct event_host_object_destroyed {
			host_object_id hoid;
		};
		struct event_epoch_reached {
			task_id tid;
		};
		struct test_event_signal_idle { // only used by scheduler_testspy
			std::atomic<bool>* idle;
		};
		using event = std::variant<event_task_available, event_buffer_created, event_set_buffer_debug_name, event_buffer_destroyed, event_host_object_created,
		    event_host_object_destroyed, event_epoch_reached, test_event_signal_idle>;

		std::unique_ptr<command_graph> m_cdag;
		command_recorder* m_crec;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		std::unique_ptr<instruction_graph> m_idag;
		instruction_recorder* m_irec;
		std::unique_ptr<instruction_graph_generator> m_iggen;

		delegate* m_delegate; // Pointer instead of reference so we can omit for tests / benchmarks

		std::queue<event> m_available_events;

		mutable std::mutex m_events_mutex;
		std::condition_variable m_events_cv;

		void notify(const event& evt);
	};

	class scheduler final : public abstract_scheduler {
		friend struct scheduler_testspy;

	  public:
		scheduler(size_t num_nodes, node_id local_node_id, instruction_graph_generator::system_info system_info, const task_manager& tm,
		    delegate* delegate, command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});

		scheduler(const scheduler&) = delete;
		scheduler(scheduler&&) = delete;
		scheduler& operator=(const scheduler&) = delete;
		scheduler& operator=(scheduler&&) = delete;

		~scheduler() override;

	  private:
		std::thread m_thread;

		void thread_main();
	};

} // namespace detail
} // namespace celerity
