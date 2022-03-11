#include "print_graph.h"

#include <regex>
#include <sstream>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "task_manager.h"
#include "reduction_manager.h"

namespace celerity {
namespace detail {

	const char* dependency_style(dependency_kind kind) {
		switch(kind) {
		case dependency_kind::ORDER_DEP: return "color=blue"; break;
		case dependency_kind::ANTI_DEP: return "color=limegreen"; break;
		default: return "";
		}
	}

	std::string get_task_label(const task* tsk) {
		switch(tsk->get_type()) {
		case task_type::NOP: return fmt::format("Task {} (nop)", tsk->get_id());
		case task_type::HOST_COMPUTE: return fmt::format("Task {} (host-compute)", tsk->get_id());
		case task_type::DEVICE_COMPUTE: return fmt::format("Task {} ({})", tsk->get_id(), tsk->get_debug_name());
		case task_type::COLLECTIVE: return fmt::format("Task {} (collective #{})", tsk->get_id(), static_cast<size_t>(tsk->get_collective_group_id()));
		case task_type::MASTER_NODE: return fmt::format("Task {} (master-node)", tsk->get_id());
		case task_type::HORIZON: return fmt::format("Task {} (horizon)", tsk->get_id());
		default: assert(false); return fmt::format("Task {} (unknown)", tsk->get_id());
		}
	}

	std::string print_graph(const std::unordered_map<task_id, std::unique_ptr<task>>& tdag) {
		std::ostringstream ss;
		ss << "digraph G { label=\"Task Graph\" ";

		for(auto& it : tdag) {
			if(it.first == 0) continue; // Skip INIT task
			const auto tsk = it.second.get();

			std::unordered_map<std::string, std::string> props;
			props["label"] = "\"" + get_task_label(tsk) + "\"";

			ss << tsk->get_id();
			ss << "[";
			for(const auto& it : props) {
				ss << " " << it.first << "=" << it.second;
			}
			ss << "];";

			for(auto d : tsk->get_dependencies()) {
				if(d.node->get_id() == 0) continue; // Skip INIT task
				ss << fmt::format("{} -> {} [{}];", d.node->get_id(), tsk->get_id(), dependency_style(d.kind));
			}
		}

		ss << "}";
		return ss.str();
	}

	std::string make_requirements_label(const execution_command* const cmd, const task_manager& tm, const reduction_manager& rm) {
		const auto tsk = tm.get_task(cmd->get_tid());

		auto access_map = tsk->get_buffer_access_map();
		std::unordered_map<buffer_id, reduction_id> reduction_map;
		for(auto rid : tsk->get_reductions()) {
			const auto reduction = rm.get_reduction(rid);
			const auto bid = reduction.output_buffer_id;
			const auto rmode = cmd->is_reduction_initializer() && reduction.initialize_from_buffer ? access_mode::read_write : access_mode::discard_write;
			access_map.add_access(bid, std::make_unique<range_mapper<1, celerity::access::all<>>>(celerity::access::all{}, rmode, range<1>{1}));
			reduction_map.emplace(bid, rid);
		}

		const auto sr = cmd->get_execution_range();
		const auto global_size = tsk->get_global_size();

		std::string debug_label;
		for(const buffer_id bid : access_map.get_accessed_buffers()) {
			std::unordered_map<cl::sycl::access::mode, GridRegion<3>> required_modes;
			for(const auto mode : detail::access::all_modes) {
				if(const auto req = access_map.get_requirements_for_access(bid, mode, tsk->get_dimensions(), sr, global_size); !req.empty()) {
					if(const auto rit = reduction_map.find(bid); rit != reduction_map.end()) { debug_label += fmt::format("(R{}) ", rit->second); }
					debug_label += fmt::format("{} {} {}\\n", detail::access::mode_traits::name(mode), bid, req);
				}
			}
		}

		for(const auto& side_effect : tsk->get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			debug_label += fmt::format("affect host-object {}\\n", hoid);
		}

		return debug_label;
	}

	std::string get_command_label(const abstract_command* cmd, const task_manager& tm, const reduction_manager& rm) {
		std::string label = fmt::format("[{}] Node {}:\\n", cmd->get_cid(), cmd->get_nid());
		if(const auto ecmd = dynamic_cast<const execution_command*>(cmd)) {
			label += fmt::format("EXECUTION {}\\n{}", subrange_to_grid_box(ecmd->get_execution_range()), make_requirements_label(ecmd, tm, rm));
		} else if(const auto pcmd = dynamic_cast<const push_command*>(cmd)) {
			if(pcmd->get_rid()) { label += fmt::format("(R{}) ", pcmd->get_rid()); }
			label += fmt::format("PUSH {} to {}\\n {}", pcmd->get_bid(), pcmd->get_target(), subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(cmd)) {
			if(apcmd->get_source()->get_rid()) { label += fmt::format("(R{}) ", apcmd->get_source()->get_rid()); }
			label += fmt::format("AWAIT PUSH {} from {}\n {}", apcmd->get_source()->get_bid(), apcmd->get_source()->get_nid(),
			    subrange_to_grid_box(apcmd->get_source()->get_range()));
		} else if(const auto rrcmd = dynamic_cast<const reduction_command*>(cmd)) {
			label += fmt::format("REDUCTION {}", rrcmd->get_rid());
		} else if(const auto hcmd = dynamic_cast<const horizon_command*>(cmd)) {
			label += "HORIZON";
		} else {
			label += "UNKNOWN";
		}
		return label;
	}

	std::string print_graph(const command_graph& cdag, const task_manager& tm, const reduction_manager& rm) {
		std::ostringstream main_ss;
		std::unordered_map<task_id, std::ostringstream> task_subgraph_ss;

		const auto write_vertex = [&](std::ostream& out, abstract_command* cmd) {
			const char* colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			std::unordered_map<std::string, std::string> props;
			props["label"] = "\"" + get_command_label(cmd, tm, rm) + "\"";
			props["fontcolor"] = colors[cmd->get_nid() % (sizeof(colors) / sizeof(char*))];
			if(isa<task_command>(cmd)) { props["shape"] = "box"; }

			out << cmd->get_cid();
			out << "[";
			for(const auto& it : props) {
				out << " " << it.first << "=" << it.second;
			}
			out << "];";
		};

		const auto write_command = [&](auto* cmd) {
			if(isa<nop_command>(cmd)) return;

			if(const auto tcmd = dynamic_cast<task_command*>(cmd)) {
				// Add to subgraph as well
				if(task_subgraph_ss.find(tcmd->get_tid()) == task_subgraph_ss.end()) {
					// TODO: Can we print the task debug label here as well?
					task_subgraph_ss[tcmd->get_tid()] << fmt::format("subgraph cluster_{} {{ label=\"Task {}\"; color=gray;", tcmd->get_tid(), tcmd->get_tid());
				}
				write_vertex(task_subgraph_ss[tcmd->get_tid()], cmd);
			} else {
				write_vertex(main_ss, cmd);
			}

			for(auto d : cmd->get_dependencies()) {
				if(isa<nop_command>(d.node)) continue;
				main_ss << fmt::format("{} -> {} [{}];", d.node->get_cid(), cmd->get_cid(), dependency_style(d.kind));
			}

			// Add a dashed line to the corresponding PUSH
			if(isa<await_push_command>(cmd)) {
				auto await_push = static_cast<await_push_command*>(cmd);
				main_ss << fmt::format("{} -> {} [style=dashed color=gray40];", await_push->get_source()->get_cid(), cmd->get_cid());
			}
		};

		for(auto cmd : cdag.all_commands()) {
			write_command(cmd);
		}

		// Close all subgraphs
		for(auto& sg : task_subgraph_ss) {
			sg.second << "}";
		}

		std::ostringstream result_ss;
		result_ss << "digraph G { label=\"Command Graph\" ";
		for(auto& sg : task_subgraph_ss) {
			result_ss << sg.second.str();
		}
		result_ss << main_ss.str();
		result_ss << "}";
		return result_ss.str();
	}

} // namespace detail
} // namespace celerity
