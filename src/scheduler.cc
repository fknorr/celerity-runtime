#include "scheduler.h"

#include "command_graph_generator.h"
#include "instruction_graph_generator.h"
#include "log.h"
#include "named_threads.h"
#include "recorders.h"
#include "tracy.h"

#include <matchbox.hh>


namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(const size_t num_nodes, const node_id local_node_id, const system_info& system, const task_manager& tm,
	    delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
	    : m_cdag(std::make_unique<command_graph>()), m_crec(crec),
	      m_cggen(std::make_unique<command_graph_generator>(num_nodes, local_node_id, *m_cdag, tm, crec, policy.command_graph_generator)),
	      m_idag(std::make_unique<instruction_graph>()), m_irec(irec), //
	      m_iggen(std::make_unique<instruction_graph_generator>(
	          tm, num_nodes, local_node_id, system, *m_idag, delegate, irec, policy.instruction_graph_generator)) {}

	abstract_scheduler::~abstract_scheduler() = default;

	std::vector<const abstract_command*> abstract_scheduler::build_task(const task& tsk) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::build_task", WebMaroon, "T{} build", tsk.get_id());
		CELERITY_DETAIL_TRACY_ZONE_TEXT(utils::make_task_debug_label(tsk.get_type(), tsk.get_id(), tsk.get_debug_name()));
		return m_cggen->build_task(tsk);
	}

	bool abstract_scheduler::should_compile_commands(const task& tsk) const {
		switch(m_lookahead) {
		case experimental::lookahead::none:
		case experimental::lookahead::automatic: return true;
		case experimental::lookahead::infinite: return false;
		default: utils::unreachable();
		}
	}

	void abstract_scheduler::compile_command(const abstract_command& cmd) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::compile_command", MidnightBlue, "C{} compile", cmd.get_cid());
		CELERITY_DETAIL_TRACY_ZONE_TEXT("{}", cmd.get_type());
		m_iggen->compile(cmd);
	}

	// TODO split CDAG / IDAG schedulers to simplify queueing?
	// - IDAG scheduler will always need a branch "perform now or later?" so we can compile the first instruction ASAP
	// - We must not opportunistically anticipate() things later in the queue, otherwise we get non-deterministic IDAGs

	void abstract_scheduler::schedule() {
		std::optional<task_id> shutdown_epoch_built = std::nullopt;
		bool shutdown_epoch_reached = false;

		std::queue<task_queue_event> local_task_queue;

		std::queue<command_queue_event> command_queue;
		int num_queued_epoch_cmds = 0;
		int num_queued_horizon_cmds = 0;
		int num_queued_fence_cmds = 0;

		while(!shutdown_epoch_reached) {
			if(local_task_queue.empty()) {
				// We can frequently suspend / resume the scheduler thread without adding latency as long as the executor queue remains non-empty
				m_task_queue.wait_while_empty();
				for(const auto& event : m_task_queue.pop_all()) {
					local_task_queue.emplace(event);
				}
				assert(!local_task_queue.empty());
			}

			matchbox::match(
			    local_task_queue.front(),
			    [&](const event_task_available& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    assert(e.tsk != nullptr);
				    auto& tsk = *e.tsk;

				    auto commands = build_task(tsk);

				    for(const auto cmd : commands) {
					    // If there are multiple commands, the shutdown epoch must come last. m_iggen.delegate must be considered dangling after receiving
					    // the corresponding instruction, as runtime will begin destroying the executor after it has observed the epoch to be reached.
					    assert(!shutdown_epoch_built);

					    command_queue.emplace(event_command_available{cmd});
					    switch(cmd->get_type()) {
					    case command_type::epoch: num_queued_epoch_cmds += 1; break;
					    case command_type::horizon: num_queued_horizon_cmds += 1; break;
					    case command_type::fence: num_queued_fence_cmds += 1; break;
					    default: break;
					    }

					    if(tsk.get_type() == task_type::epoch && tsk.get_epoch_action() == epoch_action::shutdown) { shutdown_epoch_built = tsk.get_id(); }
				    }
			    },
			    [&](const event_buffer_created& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_created", DarkGreen, "B{} create", e.bid);
				    m_cggen->notify_buffer_created(e.bid, e.range, e.user_allocation_id != null_allocation_id);
				    command_queue.push(e);
			    },
			    [&](const event_buffer_debug_name_changed& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_name_changed", DarkGreen, "B{} set name", e.bid);
				    m_cggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
				    command_queue.push(e);
			    },
			    [&](const event_buffer_destroyed& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_destroyed", DarkGreen, "B{} destroy", e.bid);
				    m_cggen->notify_buffer_destroyed(e.bid);
				    command_queue.push(e);
			    },
			    [&](const event_host_object_created& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_created", DarkGreen, "H{} create", e.hoid);
				    m_cggen->notify_host_object_created(e.hoid);
				    command_queue.push(e);
			    },
			    [&](const event_host_object_destroyed& e) {
				    assert(!shutdown_epoch_built && !shutdown_epoch_reached);
				    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_destroyed", DarkGreen, "H{} destroy", e.hoid);
				    m_cggen->notify_host_object_destroyed(e.hoid);
				    command_queue.push(e);
			    },
			    [&](const event_epoch_reached& e) { //
				    assert(!shutdown_epoch_reached);
				    {
					    // The cggen automatically prunes the CDAG on generation, which is safe because commands are not shared across threads.
					    // We might want to refactor this to match the IDAG behavior in the future.
					    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::prune_idag", Gray);
					    m_idag->prune_before_epoch(e.tid);
				    }

				    // The scheduler will receive the shutdown-epoch completion event via the runtime even if executor destruction has already begun.
				    if(shutdown_epoch_built && e.tid == *shutdown_epoch_built) { shutdown_epoch_reached = true; }
			    },
			    [&](const event_set_lookahead& e) { //
				    command_queue.push(e);
			    },
			    [&](const event_test_inspect& e) { //
				    e.inspect();
			    });

			local_task_queue.pop();

			while(!command_queue.empty()) {
				if(num_queued_epoch_cmds == 0 && num_queued_fence_cmds == 0
				    && (m_lookahead == experimental::lookahead::infinite || num_queued_horizon_cmds > 1)) {
					break;
				}

				matchbox::match(
				    command_queue.front(),
				    [&](const event_command_available& e) {
					    compile_command(*e.cmd);

					    switch(e.cmd->get_type()) {
					    case command_type::epoch:
						    assert(num_queued_epoch_cmds > 0);
						    num_queued_epoch_cmds -= 1;
						    break;
					    case command_type::horizon:
						    assert(num_queued_horizon_cmds > 0);
						    num_queued_horizon_cmds -= 1;
						    break;
					    case command_type::fence:
						    assert(num_queued_fence_cmds > 0);
						    num_queued_fence_cmds -= 1;
						    break;
					    default: break;
					    }
				    },
				    [&](const event_buffer_created& e) { m_iggen->notify_buffer_created(e.bid, e.range, e.elem_size, e.elem_align, e.user_allocation_id); },
				    [&](const event_buffer_debug_name_changed& e) { m_iggen->notify_buffer_debug_name_changed(e.bid, e.debug_name); },
				    [&](const event_buffer_destroyed& e) { m_iggen->notify_buffer_destroyed(e.bid); },
				    [&](const event_host_object_created& e) { m_iggen->notify_host_object_created(e.hoid, e.owns_instance); },
				    [&](const event_host_object_destroyed& e) { m_iggen->notify_host_object_destroyed(e.hoid); },
				    [&](const event_set_lookahead& e) { m_lookahead = e.lookahead; });

				command_queue.pop();
			}
		}

		assert(local_task_queue.empty());
		assert(command_queue.empty());
	}

	void abstract_scheduler::task_notify(task_queue_event&& evt) { m_task_queue.push(std::move(evt)); }

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
		CELERITY_DETAIL_TRACY_SET_THREAD_NAME_AND_ORDER("cy-scheduler", tracy_detail::thread_order::scheduler)
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
