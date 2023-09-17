#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include "ranges.h"
#include "types.h"

namespace celerity {
namespace detail {

	class distributed_graph_generator;
	class executor;
	class instruction_executor;
	class instruction_graph_generator;
	class task;

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
	  public:
		abstract_scheduler(const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen,
		    instruction_executor& exec);

		virtual ~abstract_scheduler();

		virtual void startup() = 0;

		virtual void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_created(
		    const buffer_id bid, const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const bool host_initialized) {
			notify(event_buffer_created{bid, dims, range, elem_size, elem_align, host_initialized});
		}

		void notify_buffer_destroyed(const buffer_id bid) { notify(event_buffer_destroyed{bid}); }

		void notify_host_object_created(const host_object_id hoid) { notify(event_host_object_created{hoid}); }

		void notify_host_object_destroyed(const host_object_id hoid) { notify(event_host_object_destroyed{hoid}); }

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

		// Constructor for tests that does not require an executor
		abstract_scheduler(const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen);

	  private:
		struct event_shutdown {};
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
		struct event_buffer_destroyed {
			buffer_id bid;
		};
		struct event_host_object_created {
			host_object_id hoid;
		};
		struct event_host_object_destroyed {
			host_object_id hoid;
		};
		using event = std::variant<event_shutdown, event_task_available, event_buffer_created, event_buffer_destroyed, event_host_object_created,
		    event_host_object_destroyed>;

		bool m_is_dry_run;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		std::unique_ptr<instruction_graph_generator> m_iggen;
		instruction_executor* m_exec; // Pointer instead of reference so we can omit for tests / benchmarks

		std::queue<event> m_available_events;
		std::queue<event> m_in_flight_events;

		mutable std::mutex m_events_mutex;
		std::condition_variable m_events_cv;

		void notify(const event& evt);
	};

	class scheduler final : public abstract_scheduler {
		friend struct scheduler_testspy;

	  public:
		using abstract_scheduler::abstract_scheduler;

		void startup() override;

		void shutdown() override;

	  private:
		std::thread m_worker_thread;
	};

} // namespace detail
} // namespace celerity
