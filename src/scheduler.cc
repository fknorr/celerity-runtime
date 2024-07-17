#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "instruction_graph_generator.h"
#include "log.h"
#include "named_threads.h"
#include "recorders.h"
#include "tracy.h"


namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(const size_t num_nodes, const node_id local_node_id, const system_info& system, const task_manager& tm,
	    delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
	    : m_cdag(std::make_unique<command_graph>()), m_crec(crec),
	      m_dggen(std::make_unique<distributed_graph_generator>(num_nodes, local_node_id, *m_cdag, tm, crec, policy.command_graph_generator)),
	      m_idag(std::make_unique<instruction_graph>()), m_irec(irec), //
	      m_iggen(std::make_unique<instruction_graph_generator>(
	          tm, num_nodes, local_node_id, system, *m_idag, delegate, irec, policy.instruction_graph_generator)) {}

	abstract_scheduler::~abstract_scheduler() = default;

	void abstract_scheduler::schedule() {
		bool shutdown = false;
		while(!shutdown) {
			// We can frequently suspend / resume the scheduler thread without adding latency as long as the executor queue remains filled
			m_event_queue.wait_while_empty();

			for(auto& event : m_event_queue.pop_all()) {
				matchbox::match(
				    event,
				    [&](const event_task_available& e) {
					    assert(!shutdown);
					    assert(e.tsk != nullptr);
					    auto& tsk = *e.tsk;

					    std::vector<abstract_command*> commands;
					    {
						    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::build_task", WebMaroon, "T{} build", tsk.get_id());
						    CELERITY_DETAIL_TRACY_ZONE_TEXT("{}", utils::make_task_debug_label(tsk.get_type(), tsk.get_id(), tsk.get_debug_name()));

						    commands = sort_topologically(m_dggen->build_task(tsk));
					    }

					    for(const auto cmd : commands) {
						    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::compile_command", MidnightBlue, "C{} compile", cmd->get_cid());
						    CELERITY_DETAIL_TRACY_ZONE_TEXT("{}", cmd->get_type());

						    m_iggen->compile(*cmd);

						    if(tsk.get_type() == task_type::epoch && tsk.get_epoch_action() == epoch_action::shutdown) {
							    shutdown = true;
							    // m_iggen.delegate must be considered dangling as soon as the instructions for the shutdown epoch have been emitted
						    }
					    }
				    },
				    [&](const event_buffer_created& e) {
					    assert(!shutdown);
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::create_buffer", DarkGreen, "B{} create", e.bid);
					    m_dggen->notify_buffer_created(e.bid, e.range, e.user_allocation_id != null_allocation_id);
					    m_iggen->notify_buffer_created(e.bid, e.range, e.elem_size, e.elem_align, e.user_allocation_id);
				    },
				    [&](const event_buffer_debug_name_changed& e) {
					    assert(!shutdown);
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::set_buffer_name", DarkGreen, "B{} set name", e.bid);
					    m_dggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
					    m_iggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
				    },
				    [&](const event_buffer_destroyed& e) {
					    assert(!shutdown);
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::destroy_buffer", DarkGreen, "B{} destroy", e.bid);
					    m_dggen->notify_buffer_destroyed(e.bid);
					    m_iggen->notify_buffer_destroyed(e.bid);
				    },
				    [&](const event_host_object_created& e) {
					    assert(!shutdown);
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::create_host_object", DarkGreen, "H{} create", e.hoid);
					    m_dggen->notify_host_object_created(e.hoid);
					    m_iggen->notify_host_object_created(e.hoid, e.owns_instance);
				    },
				    [&](const event_host_object_destroyed& e) {
					    assert(!shutdown);
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::destroy_host_object", DarkGreen, "H{} destroy", e.hoid);
					    m_dggen->notify_host_object_destroyed(e.hoid);
					    m_iggen->notify_host_object_destroyed(e.hoid);
				    },
				    [&](const event_epoch_reached& e) { //
					    // The dggen automatically prunes the CDAG on generation, which is safe because commands are not shared across threads.
					    // We might want to refactor this to match the IDAG behavior in the future.
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::prune_idag", Gray, "prune");
					    m_idag->prune_before_epoch(e.tid);
				    },
				    [&](const event_test_inspect& e) { //
					    e.inspect();
				    });
			}
		}
	}

	void abstract_scheduler::notify(event&& event) { m_event_queue.push(std::move(event)); }

	scheduler::scheduler(const size_t num_nodes, const node_id local_node_id, const system_info& system, const task_manager& tm, delegate* const delegate,
	    command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
	    : abstract_scheduler(num_nodes, local_node_id, system, tm, delegate, crec, irec, policy), m_thread(&scheduler::thread_main, this) {
		set_thread_name(m_thread.native_handle(), "cy-scheduler");
	}

	scheduler::~scheduler() {
		// schedule() will exit as soon as it has processed the shutdown epoch
		m_thread.join();
	}

	void scheduler::thread_main() {
		CELERITY_DETAIL_TRACY_SET_THREAD_NAME("cy-scheduler")
		try {
			schedule();
		}
		// LCOV_EXCL_START
		catch(const std::exception& e) {
			CELERITY_CRITICAL("[scheduler] {}", e.what());
			std::abort();
		}
		// LCOV_EXCL_STOP
	}

} // namespace detail
} // namespace celerity
