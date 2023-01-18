#include "print_graph.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "instruction_graph.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	template <typename Dependency>
	const char* dependency_style(const Dependency& dep) {
		if(dep.kind == dependency_kind::anti_dep) return "color=limegreen";
		switch(dep.origin) {
		case dependency_origin::collective_group_serialization: return "color=blue";
		case dependency_origin::execution_front: return "color=orange";
		case dependency_origin::last_epoch: return "color=orchid";
		default: return "";
		}
	}

	const char* task_type_string(const task_type tt) {
		switch(tt) {
		case task_type::epoch: return "epoch";
		case task_type::host_compute: return "host-compute";
		case task_type::device_compute: return "device-compute";
		case task_type::collective: return "collective host";
		case task_type::master_node: return "master-node host";
		case task_type::horizon: return "horizon";
		default: return "unknown";
		}
	}

	const char* command_type_string(const command_type ct) {
		switch(ct) {
		// default: return "unknown";
		case command_type::epoch: return "epoch";
		case command_type::horizon: return "horizon";
		case command_type::execution: return "execution";
		case command_type::data_request: return "data request";
		case command_type::push: return "push";
		case command_type::await_push: return "await push";
		case command_type::reduction: return "reduction";
		default: return "unknown";
		}
	}

	std::string get_buffer_label(const buffer_manager* bm, const buffer_id bid) {
		// if there is no buffer manager or no name defined, the name will be the buffer id.
		// if there is a name we want "id name"
		std::string name;
		if(bm != nullptr) { name = bm->get_debug_name(bid); }
		return !name.empty() ? fmt::format("B{} \"{}\"", bid, name) : fmt::format("B{}", bid);
	}

	void format_requirements(
	    std::string& label, const task& tsk, subrange<3> execution_range, access_mode reduction_init_mode, const buffer_manager* const bm) {
		for(const auto& reduction : tsk.get_reductions()) {
			auto rmode = cl::sycl::access::mode::discard_write;
			if(reduction.init_from_buffer) { rmode = reduction_init_mode; }

			const auto req = GridRegion<3>{{1, 1, 1}};
			const std::string bl = get_buffer_label(bm, reduction.bid);
			fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", reduction.rid, detail::access::mode_traits::name(rmode), bl, req);
		}

		const auto& bam = tsk.get_buffer_access_map();
		for(const auto bid : bam.get_accessed_buffers()) {
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), execution_range, tsk.get_global_size());
				const std::string bl = get_buffer_label(bm, bid);
				// While uncommon, we do support chunks that don't require access to a particular buffer at all.
				if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", detail::access::mode_traits::name(mode), bl, req); }
			}
		}

		for(const auto& [hoid, order] : tsk.get_side_effect_map()) {
			fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
		}
	}

	std::string get_task_label(const task& tsk, const buffer_manager* const bm) {
		std::string label;
		fmt::format_to(std::back_inserter(label), "T{}", tsk.get_id());
		if(!tsk.get_debug_name().empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", tsk.get_debug_name()); }

		const auto execution_range = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};

		fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.get_type()));
		if(tsk.get_type() == task_type::host_compute || tsk.get_type() == task_type::device_compute) {
			fmt::format_to(std::back_inserter(label), " {}", execution_range);
		} else if(tsk.get_type() == task_type::collective) {
			fmt::format_to(std::back_inserter(label), " in CG{}", tsk.get_collective_group_id());
		}

		format_requirements(label, tsk, execution_range, access_mode::read_write, bm);

		return label;
	}

	std::string print_task_graph(const task_ring_buffer& tdag, const buffer_manager* const bm) {
		std::string dot = "digraph G {label=\"Task Graph\" ";

		for(auto tsk : tdag) {
			const auto shape = tsk->get_type() == task_type::epoch || tsk->get_type() == task_type::horizon ? "ellipse" : "box style=rounded";
			fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk->get_id(), shape, get_task_label(*tsk, bm));
			for(auto d : tsk->get_dependencies()) {
				fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node->get_id(), tsk->get_id(), dependency_style(d));
			}
		}

		dot += "}";
		return dot;
	}

	std::string get_command_label(const abstract_command& cmd, const task_manager& tm, const buffer_manager* const bm) {
		const command_id cid = cmd.get_cid();
		const node_id nid = cmd.get_nid();

		std::string label = fmt::format("C{} on N{}<br/>", cid, nid);

		if(const auto ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
			label += "<b>epoch</b>";
			if(ecmd->get_epoch_action() == epoch_action::barrier) { label += " (barrier)"; }
			if(ecmd->get_epoch_action() == epoch_action::shutdown) { label += " (shutdown)"; }
		} else if(const auto xcmd = dynamic_cast<const execution_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", subrange_to_grid_box(xcmd->get_execution_range()));
		} else if(const auto pcmd = dynamic_cast<const push_command*>(&cmd)) {
			if(pcmd->get_rid()) { fmt::format_to(std::back_inserter(label), "(R{}) ", pcmd->get_rid()); }
			const std::string bl = get_buffer_label(bm, pcmd->get_bid());
			fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>B{} {}", pcmd->get_transfer_id(), pcmd->get_target(), bl,
			    subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
			// if(apcmd->get_source()->get_rid()) { label += fmt::format("(R{}) ", apcmd->get_source()->get_rid()); }
			const std::string bl = get_buffer_label(bm, apcmd->get_bid());
			fmt::format_to(std::back_inserter(label), "<b>await push</b> transfer {} <br/>B{} {}", apcmd->get_transfer_id(), bl, apcmd->get_region());
		} else if(const auto drcmd = dynamic_cast<const data_request_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>request data</b> from N{}<br/>B{} {}", drcmd->get_source(), drcmd->get_bid(),
			    subrange_to_grid_box(drcmd->get_range()));
		} else if(const auto rrcmd = dynamic_cast<const reduction_command*>(&cmd)) {
			const auto& reduction = rrcmd->get_reduction_info();
			const auto req = GridRegion<3>{{1, 1, 1}};
			const auto bl = get_buffer_label(bm, reduction.bid);
			fmt::format_to(std::back_inserter(label), "<b>reduction</b> R{}<br/> {} {}", reduction.rid, bl, req);
		} else if(const auto hcmd = dynamic_cast<const horizon_command*>(&cmd)) {
			label += "<b>horizon</b>";
		} else {
			assert(!"Unkown command");
			label += "<b>unknown</b>";
		}

		if(const auto tcmd = dynamic_cast<const task_command*>(&cmd)) {
			if(!tm.has_task(tcmd->get_tid())) return label; // NOCOMMIT This is only needed while we do TDAG pruning but not CDAG pruning
			assert(tm.has_task(tcmd->get_tid()));

			const auto& tsk = *tm.get_task(tcmd->get_tid());

			auto reduction_init_mode = access_mode::discard_write;
			auto execution_range = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
			if(const auto ecmd = dynamic_cast<const execution_command*>(&cmd)) {
				if(ecmd->is_reduction_initializer()) { reduction_init_mode = cl::sycl::access::mode::read_write; }
				execution_range = ecmd->get_execution_range();
			}

			format_requirements(label, tsk, execution_range, reduction_init_mode, bm);
		}

		return label;
	}

	std::string print_task_reference_label(const task_id tid, const task_manager& tm) {
		std::string task_label;
		fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
		if(const auto tsk = tm.find_task(tid)) {
			if(!tsk->get_debug_name().empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", tsk->get_debug_name()); }
			task_label += "(";
			task_label += task_type_string(tsk->get_type());
			if(tsk->get_type() == task_type::collective) { fmt::format_to(std::back_inserter(task_label), " on CG{}", tsk->get_collective_group_id()); }
			task_label += ")";
		} else {
			task_label += "(deleted)";
		}
		return task_label;
	}

	std::string print_command_graph(const node_id local_nid, const command_graph& cdag, const task_manager& tm, const buffer_manager* const bm) {
		std::string main_dot;
		std::unordered_map<task_id, std::string> task_subgraph_dot;

		const auto local_to_global_id = [local_nid](uint64_t id) {
			// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
			return fmt::format("id_{}_{}", local_nid, id);
		};

		const auto print_vertex = [&](const abstract_command& cmd) {
			static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			const auto id = local_to_global_id(cmd.get_cid());
			const auto label = get_command_label(cmd, tm, bm);
			const auto fontcolor = colors[cmd.get_nid() % (sizeof(colors) / sizeof(char*))];
			const auto shape = isa<task_command>(&cmd) ? "box" : "ellipse";
			return fmt::format("{}[label=<{}> fontcolor={} shape={}];", id, label, fontcolor, shape);
		};

		for(const auto cmd : cdag.all_commands()) {
			if(const auto tcmd = dynamic_cast<const task_command*>(cmd)) {
				const auto tid = tcmd->get_tid();
				// Add to subgraph as well
				if(task_subgraph_dot.count(tid) == 0) {
					task_subgraph_dot.emplace(tid, fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;",
					                                   local_to_global_id(tid), print_task_reference_label(tid, tm)));
				}
				task_subgraph_dot[tid] += print_vertex(*cmd);
			} else {
				main_dot += print_vertex(*cmd);
			}

			for(const auto& d : cmd->get_dependencies()) {
				fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(d.node->get_cid()), local_to_global_id(cmd->get_cid()),
				    dependency_style(d));
			}

			// Add a dashed line to the corresponding push
			// if(const auto apcmd = dynamic_cast<const await_push_command*>(cmd)) {
			// 	fmt::format_to(std::back_inserter(main_dot), "{}->{}[style=dashed color=gray40];", local_to_global_id(apcmd->get_source()->get_cid()),
			// 	    local_to_global_id(cmd->get_cid()));
			// }
		};

		std::string result_dot = "digraph G{label=\"Command Graph\" "; // If this changes, also change in combine_command_graphs
		for(auto& [sg_tid, sg_dot] : task_subgraph_dot) {
			result_dot += sg_dot;
			result_dot += "}";
		}
		result_dot += main_dot;
		result_dot += "}";
		return result_dot;
	}

	std::string combine_command_graphs(const std::vector<std::string>& graphs) {
		const std::string preamble = "digraph G{label=\"Command Graph\" ";
		std::string result_dot = preamble;
		for(auto& g : graphs) {
			result_dot += g.substr(preamble.size(), g.size() - preamble.size() - 1);
		}
		result_dot += "}";
		return result_dot;
	}

	std::string print_command_reference_label(const command_id cid, const command_graph& cdag, const task_manager& tm) {
		std::string cmd_label;
		fmt::format_to(std::back_inserter(cmd_label), "C{} ", cid);
		if(cdag.has(cid)) {
			const auto cmd = cdag.get(cid);
			fmt::format_to(std::back_inserter(cmd_label), "({})", command_type_string(cmd->get_type()));
			if(const auto tcmd = dynamic_cast<task_command*>(cmd)) { cmd_label += "<br/>from " + print_task_reference_label(tcmd->get_tid(), tm); }
		} else {
			cmd_label += "(deleted)";
		}
		return cmd_label;
	}

	class instruction_graph_node_printer : public const_instruction_graph_visitor {
	  public:
		instruction_graph_node_printer(std::string& dot, const command_graph& cdag, const task_manager& tm) : m_dot(dot), m_cdag(cdag), m_tm(tm) {}

		void visit_alloc(const alloc_instruction& ainsn) override { print_node(ainsn, "<b>alloc</b> B{} {}", ainsn.get_buffer_id(), ainsn.get_region()); }

		void visit_copy(const copy_instruction& cinsn) override {
			const bool source_host = cinsn.get_source_memory() == host_memory_id;
			const bool dest_host = cinsn.get_dest_memory() == host_memory_id;
			const auto direction = source_host && dest_host ? "h2h" : source_host && !dest_host ? "h2d" : !source_host && dest_host ? "d2h" : "d2d";
			const auto side = cinsn.get_side() == copy_instruction::side::source ? "to" : "from";
			print_node(
			    cinsn, "<b>{}</b> {} M{}<br/>B{} {}", direction, side, cinsn.get_counterpart().get_memory_id(), cinsn.get_buffer_id(), cinsn.get_region());
		}

		void visit_device_kernel(const device_kernel_instruction& dkinsn) override {
			const auto back = std::back_inserter(m_dot);
			begin_node(dkinsn);
			fmt::format_to(back, "<b>kernel</b> on D{} {}", dkinsn.get_device_id(), dkinsn.get_execution_range());
			print_buffer_read_write_map(dkinsn.get_buffer_read_write_map());
			end_node();
		}

		void visit_host_kernel(const host_kernel_instruction& hkinsn) override {
			const auto back = std::back_inserter(m_dot);
			begin_node(hkinsn);
			fmt::format_to(back, "<b>host kernel</b> {}", hkinsn.get_execution_range());
			if(hkinsn.get_collective_group_id()) { fmt::format_to(back, " (collective CG{})", hkinsn.get_collective_group_id()); }
			print_buffer_read_write_map(hkinsn.get_buffer_read_write_map());
			print_side_effect_map(hkinsn.get_side_effect_map());
			end_node();
		}

		void visit_send(const send_instruction& sinsn) override {
			print_node(sinsn, "<b>send</b> to N{}<br/>B{} {}", sinsn.get_dest_node_id(), sinsn.get_buffer_id(), sinsn.get_region());
		}

		void visit_recv(const recv_instruction& rinsn) override {
			print_node(rinsn, "<b>recv</b> transfer {}<br/>B{} {}", rinsn.get_transfer_id(), rinsn.get_buffer_id(), rinsn.get_region());
		}

		void visit_horizon(const horizon_instruction& hinsn) override { print_node(hinsn, "<b>horizon</b>"); }

		void visit_epoch(const epoch_instruction& einsn) override { print_node(einsn, "<b>epoch</b>"); }

		void visit(const instruction& insn) override { print_node(insn, "<b>unknown</b>"); }

	  private:
		std::string& m_dot;
		const command_graph& m_cdag;
		const task_manager& m_tm;

		void begin_node(const instruction& insn) {
			const auto shape = insn.get_command_id().has_value() ? "box" : "ellipse";
			const auto back = std::back_inserter(m_dot);
			fmt::format_to(back, "I{0}[shape={1} label=<", insn.get_id(), shape);
			if(insn.get_command_id().has_value()) {
				fmt::format_to(back, "<font color=\"#606060\" point-size=\"14\">from {}<br/><br/></font>",
				    print_command_reference_label(*insn.get_command_id(), m_cdag, m_tm));
			}
			fmt::format_to(back, "I{0} on M{1}<br/>", insn.get_id(), insn.get_memory_id());
		}

		void end_node() { fmt::format_to(std::back_inserter(m_dot), ">];"); }

		template <typename... FmtArgs>
		void print_node(const instruction& insn, FmtArgs&&... fmt_args) {
			begin_node(insn);
			fmt::format_to(std::back_inserter(m_dot), std::forward<FmtArgs>(fmt_args)...);
			end_node();
		}

		void print_buffer_read_write_map(const buffer_read_write_map& rw_map) {
			const auto back = std::back_inserter(m_dot);
			for(const auto& [bid, rw] : rw_map) {
				if(const auto read_only = GridRegion<3>::difference(rw.reads, rw.writes); !read_only.empty()) {
					fmt::format_to(back, "<br/><i>read</i> B{} {}", bid, read_only);
				}
			}
			for(const auto& [bid, rw] : rw_map) {
				if(const auto read_write = GridRegion<3>::intersect(rw.reads, rw.writes); !read_write.empty()) {
					fmt::format_to(back, "<br/><i>read-write</i> B{} {}", bid, read_write);
				}
			}
			for(const auto& [bid, rw] : rw_map) {
				if(const auto write_only = GridRegion<3>::difference(rw.writes, rw.reads); !write_only.empty()) {
					fmt::format_to(back, "<br/><i>write</i> B{} {}", bid, write_only);
				}
			}
		}

		void print_side_effect_map(const side_effect_map& se_map) {
			const auto back = std::back_inserter(m_dot);
			for(const auto& [hoid, order] : se_map) {
				fmt::format_to(back, "<br/><i>affect</i> H{}", hoid);
			}
		}
	};

	std::string print_instruction_graph(const instruction_graph& idag, const command_graph& cdag, const task_manager& tm) {
		std::string dot = "digraph G{label=\"Instruction Graph\";";

		idag.visit(instruction_graph_node_printer(dot, cdag, tm));

		class edge_printer : public const_instruction_graph_visitor {
		  public:
			explicit edge_printer(std::string& dot) : m_dot(dot) {}

			void visit_copy(const copy_instruction& cinsn) override {
				visit(cinsn);
				if(cinsn.get_side() == copy_instruction::side::source) {
					print_edge(cinsn, cinsn.get_counterpart(), "color=gray,style=dashed,dir=both,constraint=false");
				}
			}

			void visit(const instruction& insn) override {
				for(const auto& dep : insn.get_dependencies()) {
					print_edge(*dep.node, insn, dependency_style(dep));
				}
			}

		  private:
			std::string& m_dot;

			void print_edge(const instruction& from, const instruction& to, std::string_view style) {
				fmt::format_to(std::back_inserter(m_dot), "I{}->I{}[{}];", from.get_id(), to.get_id(), style);
			}
		};
		idag.visit(edge_printer(dot));

		dot += "}";
		return dot;
	}

} // namespace detail
} // namespace celerity
