#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "instruction_executor.h"
#include "instruction_graph_generator.h"
#include "named_threads.h"
#include "utils.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(
	    bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen, instruction_executor& exec)
	    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_iggen(std::move(iggen)), m_exec(&exec) {
		assert(m_dggen != nullptr);
		assert(m_iggen != nullptr);
	}

	abstract_scheduler::abstract_scheduler(
	    const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, std::unique_ptr<instruction_graph_generator> iggen)
	    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_iggen(std::move(iggen)), m_exec(nullptr) {
		assert(m_dggen != nullptr);
		assert(m_iggen != nullptr);
	}

	abstract_scheduler::~abstract_scheduler() = default;

	void abstract_scheduler::shutdown() { notify(event_shutdown{}); }

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
					    for(const auto cmd : m_dggen->build_task(*e.tsk)) {
						    const auto instr_batch = m_iggen->compile(*cmd);
						    if(m_exec != nullptr) {
							    for(const auto instr : instr_batch) {
								    m_exec->submit(*instr);
							    }
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
					    // TODO switch dggen to explicit host object creation
					    m_iggen->create_host_object(e.hoid, e.owns_instance);
				    },
				    [&](const event_host_object_destroyed& e) {
					    // TODO clear tracking structures in dggen
					    m_iggen->destroy_host_object(e.hoid);
				    },
				    [&](const event_shutdown&) {
					    assert(in_flight_events.empty());
					    shutdown = true;
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

	void scheduler::startup() {
		m_worker_thread = std::thread(&scheduler::schedule, this);
		set_thread_name(m_worker_thread.native_handle(), "cy-scheduler");
	}

	void scheduler::shutdown() {
		abstract_scheduler::shutdown();
		if(m_worker_thread.joinable()) { m_worker_thread.join(); }
	}

} // namespace detail
} // namespace celerity
