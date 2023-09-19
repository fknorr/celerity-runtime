#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "instruction_graph_generator.h"
#include "named_threads.h"
#include "task.h"
#include "utils.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(
	    bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen, delegate* const delegate)
	    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_iggen(std::move(iggen)), m_delegate(delegate) {
		assert(m_dggen != nullptr);
		assert(m_iggen != nullptr);
	}

	abstract_scheduler::~abstract_scheduler() = default;

	void abstract_scheduler::schedule() {
		std::queue<event> in_flight_events;
		bool shutdown = false;
		while(!shutdown) {
			{
				std::unique_lock lk(m_events_mutex);
				m_events_cv.wait(lk, [this] { return !m_available_events.empty(); });
				std::swap(m_available_events, in_flight_events);
			}

			while(!in_flight_events.empty()) {
				const auto event = std::move(in_flight_events.front()); // NOLINT(performance-move-const-arg)
				in_flight_events.pop();

				utils::match(
				    event,
				    [&](const event_task_available& e) {
					    assert(e.tsk != nullptr);
					    const auto commands = m_dggen->build_task(*e.tsk);
					    for(const auto cmd : commands) {
						    const auto instructions = m_iggen->compile(*cmd);

						    if(m_delegate != nullptr) {
							    for(const auto instr : instructions) {
								    m_delegate->submit_instruction(*instr);
							    }
						    }

						    if(e.tsk->get_type() == task_type::epoch && e.tsk->get_epoch_action() == epoch_action::shutdown) {
							    assert(in_flight_events.empty());
							    shutdown = true;
							    // m_delegate must be considered dangling as soon as the instructions for the shutdown epoch have been emitted
						    }
					    }
				    },
				    [&](const event_buffer_created& e) {
					    m_dggen->create_buffer(e.bid, e.dims, e.range);
					    m_iggen->create_buffer(e.bid, e.dims, e.range, e.elem_size, e.elem_align, e.host_initialized);
				    },
				    [&](const event_set_buffer_debug_name& e) {
					    m_dggen->set_buffer_debug_name(e.bid, e.debug_name);
					    m_iggen->set_buffer_debug_name(e.bid, e.debug_name);
				    },
				    [&](const event_buffer_destroyed& e) {
					    m_dggen->destroy_buffer(e.bid);
					    m_iggen->destroy_buffer(e.bid);
				    },
				    [&](const event_host_object_created& e) {
					    m_dggen->create_host_object(e.hoid);
					    m_iggen->create_host_object(e.hoid, e.owns_instance);
				    },
				    [&](const event_host_object_destroyed& e) {
					    m_dggen->destroy_host_object(e.hoid);
					    m_iggen->destroy_host_object(e.hoid);
				    });
			}
		}
	}

	void abstract_scheduler::notify(const event& evt) {
		{
			std::lock_guard lk(m_events_mutex);
			m_available_events.push(evt);
		}
		m_events_cv.notify_one();
	}

	scheduler::scheduler(
	    const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen, delegate* const delegate)
	    : abstract_scheduler(is_dry_run, std::move(dggen), std::move(iggen), delegate), m_thread(&scheduler::schedule, this) {
		set_thread_name(m_thread.native_handle(), "cy-scheduler");
	}

	scheduler::~scheduler() {
		// schedule() will exit as soon as it has processed the shutdown epoch
		m_thread.join();
	}

} // namespace detail
} // namespace celerity
