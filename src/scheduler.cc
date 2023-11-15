#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "instruction_graph.h"
#include "instruction_graph_generator.h"
#include "named_threads.h"
#include "recorders.h"
#include "task.h"


namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(const size_t num_nodes, const node_id local_node_id,
	    std::vector<instruction_graph_generator::device_info> local_devices, const task_manager& tm, delegate* const delegate, command_recorder* const crec,
	    instruction_recorder* const irec)
	    : m_cdag(std::make_unique<command_graph>()), m_crec(crec),
	      m_dggen(std::make_unique<distributed_graph_generator>(num_nodes, local_node_id, *m_cdag, tm, crec)), m_idag(std::make_unique<instruction_graph>()),
	      m_irec(irec), m_iggen(std::make_unique<instruction_graph_generator>(tm, num_nodes, local_node_id, std::move(local_devices), *m_idag, irec)),
	      m_delegate(delegate) {}

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

				matchbox::match(
				    event,
				    [&](const event_task_available& e) {
					    assert(e.tsk != nullptr);
					    const auto commands = m_dggen->build_task(*e.tsk);
					    for(const auto cmd : commands) {
						    const auto [instructions, pilots] = m_iggen->compile(*cmd);

						    if(m_delegate != nullptr) {
							    for(const auto instr : instructions) {
								    m_delegate->submit_instruction(*instr);
							    }
							    for(const auto& p : pilots) {
								    m_delegate->submit_pilot(p);
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
					    m_dggen->create_buffer(e.bid, e.dims, e.range, e.host_initialized);
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
				    },
				    [&](const event_epoch_reached& e) { //
					    m_idag->prune_before_epoch(e.tid);
				    },
				    [&](const event_signal_idle& e) {
					    {
						    std::lock_guard lock(*e.mutex);
						    *e.idle = true;
					    }
					    e.cond->notify_one();
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

	scheduler::scheduler(const size_t num_nodes, const node_id local_node_id, std::vector<instruction_graph_generator::device_info> local_devices,
	    const task_manager& tm, delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec)
	    : abstract_scheduler(num_nodes, local_node_id, std::move(local_devices), tm, delegate, crec, irec), m_thread(&scheduler::thread_main, this) {
		set_thread_name(m_thread.native_handle(), "cy-scheduler");
	}

	scheduler::~scheduler() {
		// schedule() will exit as soon as it has processed the shutdown epoch
		m_thread.join();
	}

	void scheduler::thread_main() {
		try {
			schedule();
		} catch(const std::exception& e) {
			CELERITY_CRITICAL("[scheduler] {}", e.what());
			std::abort();
		}
	}

} // namespace detail
} // namespace celerity
