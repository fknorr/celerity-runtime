#include "graph_serializer.h"

#include <cassert>

#include "command.h"
#include "command_graph.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	bool is_virtual_dependency(const abstract_command* const cmd) {
		// The initial epoch command is not flushed, so including it in dependencies is not useful
		// TODO we might want to generate and flush init tasks explicitly to avoid this kind of special casing
		const auto ecmd = dynamic_cast<const epoch_command*>(cmd);
		return ecmd && ecmd->get_tid() == task_manager::initial_epoch_task;
	}

	void graph_serializer::flush(const std::unordered_set<abstract_command*>& cmds) {
		[[maybe_unused]] task_id check_tid = task_id(-1);

		// Separate push commands from task commands. We flush pushes first to avoid deadlocking the executor.
		// This is always safe as no other unflushed command within a single task can precede a push.
		std::vector<abstract_command*> push_cmds;
		push_cmds.reserve(cmds.size());
		std::vector<abstract_command*> collective_cmds;
		collective_cmds.reserve(cmds.size());
		std::vector<abstract_command*> task_cmds;
		task_cmds.reserve(cmds.size());

		for(auto& cmd : cmds) {
			if(isa<push_command>(cmd)) {
				push_cmds.push_back(cmd);
			} else if(isa<gather_command>(cmd) || isa<broadcast_command>(cmd) || isa<scatter_command>(cmd)) {
				collective_cmds.push_back(cmd);
			} else if(isa<task_command>(cmd)) {
				task_cmds.push_back(cmd);
			}
		}

		// Flush a command and all of its unflushed predecessors, recursively. Usually this will only require one level of recursion.
		// One notable exception are reductions, which generate a tree of await push commands and reduction commands as successors.
		[[maybe_unused]] size_t flush_count = 0;
		const auto flush_recursive = [this, &check_tid, &flush_count](abstract_command* cmd, auto recurse) -> void {
			if(cmd->is_flushed()) return;

			(void)check_tid;
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			if(isa<task_command>(cmd)) {
				// Verify that all commands belong to the same task
				assert(check_tid == task_id(-1) || check_tid == static_cast<task_command*>(cmd)->get_tid());
				check_tid = static_cast<task_command*>(cmd)->get_tid();
			}
#endif
			std::vector<command_id> deps;
			for(auto dep : cmd->get_dependencies()) {
				recurse(dep.node, recurse);
				if(!is_virtual_dependency(dep.node)) { deps.push_back(dep.node->get_cid()); }
			}
			serialize_and_flush(cmd, deps);
			flush_count++;
		};

		for(auto& cmds_ptr : {&push_cmds, &collective_cmds, &task_cmds}) {
			for(auto& cmd : *cmds_ptr) {
				flush_recursive(cmd, flush_recursive);
			}
		}

		assert(flush_count == cmds.size());
	}

	void graph_serializer::serialize_and_flush(abstract_command* cmd, std::vector<command_id> dependencies) const {
		assert(!cmd->is_flushed() && "Command has already been flushed.");

		command_pkg pkg;

		pkg.cid = cmd->get_cid();
		if(const auto* ecmd = dynamic_cast<epoch_command*>(cmd)) {
			pkg.data = epoch_data{ecmd->get_tid(), ecmd->get_epoch_action()};
		} else if(const auto* xcmd = dynamic_cast<execution_command*>(cmd)) {
			pkg.data = execution_data{xcmd->get_tid(), xcmd->get_execution_range(), xcmd->is_reduction_initializer(), xcmd->get_device_id()};
		} else if(const auto* pcmd = dynamic_cast<push_command*>(cmd)) {
			pkg.data = push_data{pcmd->get_bid(), pcmd->get_rid(), pcmd->get_target(), pcmd->get_transfer_id(), pcmd->get_range()};
		} else if(const auto* apcmd = dynamic_cast<await_push_command*>(cmd)) {
			subrange<3> region[await_push_data::max_subranges] = {};
			size_t i = 0;
			apcmd->get_region().scanByBoxes([&](const GridBox<3>& box) {
				if(i >= await_push_data::max_subranges) throw std::runtime_error("NOPE");
				region[i++] = grid_box_to_subrange(box);
			});
			auto apd = await_push_data{apcmd->get_bid(), 0 /* FIXME */, apcmd->get_transfer_id(), 0, {}};
			apd.num_subranges = i;
			std::memcpy(&apd.region[0], &region[0], sizeof(region));
			pkg.data = std::move(apd);
		} else if(const auto* drcmd = dynamic_cast<data_request_command*>(cmd)) {
			pkg.data = data_request_data{drcmd->get_bid(), drcmd->get_source(), drcmd->get_range()};
		} else if(const auto* rcmd = dynamic_cast<reduction_command*>(cmd)) {
			pkg.data = reduction_data{rcmd->get_reduction_info().rid};
		} else if(const auto* hcmd = dynamic_cast<horizon_command*>(cmd)) {
			pkg.data = horizon_data{hcmd->get_tid()};
		} else if(const auto* gcmd = dynamic_cast<gather_command*>(cmd)) {
			pkg.data = gather_data{gcmd->get_bid(), gcmd->get_source_regions(), gcmd->get_dest_region(), gcmd->get_single_destination()};
		} else if(const auto* bcmd = dynamic_cast<broadcast_command*>(cmd)) {
			pkg.data = broadcast_data{bcmd->get_bid(), bcmd->get_region(), bcmd->get_source()};
		} else if(const auto* scmd = dynamic_cast<scatter_command*>(cmd)) {
			pkg.data = scatter_data{scmd->get_bid(), scmd->get_source_nid(), scmd->get_source_region(), scmd->get_dest_regions()};
		} else {
			assert(false && "Unknown command");
		}

		pkg.dependencies = std::move(dependencies);

		m_flush_cb(cmd->get_nid(), std::move(pkg));
		cmd->mark_as_flushed();
	}

} // namespace detail
} // namespace celerity
