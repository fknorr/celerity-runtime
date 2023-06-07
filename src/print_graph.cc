#include "print_graph.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "fmt_internals.h"
#include "grid.h"
#include "instruction_graph.h"
#include "spdlog/fmt/bundled/core.h"
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
		case task_type::host_collective: return "host_collective host";
		case task_type::master_node: return "master-node host";
		case task_type::horizon: return "horizon";
		case task_type::forward: return "forward";
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
	    std::string& label, const command_group_task& tsk, subrange<3> execution_range, access_mode reduction_init_mode, const buffer_manager* const bm) {
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

		if(const auto cgtsk = dynamic_cast<const command_group_task*>(&tsk)) {
			if(!cgtsk->get_debug_name().empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", cgtsk->get_debug_name()); }
		}

		fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.get_type()));

		if(const auto cgtsk = dynamic_cast<const command_group_task*>(&tsk)) {
			const auto execution_range = subrange<3>{cgtsk->get_global_offset(), cgtsk->get_global_size()};
			if(cgtsk->get_type() == task_type::host_compute || cgtsk->get_type() == task_type::device_compute) {
				fmt::format_to(std::back_inserter(label), " {}", execution_range);
			} else if(cgtsk->get_type() == task_type::host_collective) {
				fmt::format_to(std::back_inserter(label), " in CG{}", cgtsk->get_collective_group_id());
			}

			format_requirements(label, *cgtsk, execution_range, access_mode::read_write, bm);
		} else if(const auto* ftsk = dynamic_cast<const forward_task*>(&tsk)) {
			fmt::format_to(std::back_inserter(label), "<br/>B{} {}", ftsk->get_bid(), ftsk->get_region());
		}
		return label;
	}

	std::string print_task_graph(const task_ring_buffer& tdag, const buffer_manager* const bm) {
		std::string dot = "digraph G {label=\"Task Graph\" ";

		for(auto tsk : tdag) {
			const auto shape = isa<command_group_task>(tsk) ? "box style=rounded" : "ellipse";
			fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk->get_id(), shape, get_task_label(*tsk, bm));
			for(auto d : tsk->get_dependencies()) {
				std::vector<std::string> attrs;
				if(const auto d_style = dependency_style(d); *d_style != 0) { attrs.push_back(d_style); }
				fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node->get_id(), tsk->get_id(), fmt::join(attrs, ","));
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
			const auto xbox = subrange_to_grid_box(xcmd->get_execution_range());
			if(const auto tsk = tm.find_task(xcmd->get_tid()); tsk && tsk->get_execution_target() == execution_target::device) {
				fmt::format_to(std::back_inserter(label), "<b>execution</b> on D{} {}", xcmd->get_device_id(), xbox);
			} else {
				fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", xbox);
			}
		} else if(const auto pcmd = dynamic_cast<const push_command*>(&cmd)) {
			if(pcmd->get_rid()) { fmt::format_to(std::back_inserter(label), "(R{}) ", pcmd->get_rid()); }
			const std::string bl = get_buffer_label(bm, pcmd->get_bid());
			fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>{} {}", pcmd->get_transfer_id(), pcmd->get_target(), bl,
			    subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
			// if(apcmd->get_source()->get_rid()) { label += fmt::format("(R{}) ", apcmd->get_root()->get_rid()); }
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
		} else if(const auto gcmd = dynamic_cast<const gather_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>gather</b> to N{}", gcmd->get_root());
			if(const auto& source_region = gcmd->get_source_regions()[gcmd->get_nid()]; !source_region.empty()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>read</i> B{} {}", gcmd->get_bid(), source_region);
			}
			if(gcmd->get_root() == gcmd->get_nid()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>write</i> B{} {}", gcmd->get_bid(), merge_regions(gcmd->get_source_regions()));
			}
			if(const auto& coherence = gcmd->get_local_coherence_region(); !coherence.empty()) {
				fmt::format_to(std::back_inserter(label), "<br/>D0 <i>make coherent</i> {}", coherence);
			}
		} else if(const auto agcmd = dynamic_cast<const allgather_command*>(&cmd)) {
			label += "<b>all-gather</b>";
			if(const auto& source_region = agcmd->get_source_regions()[agcmd->get_nid()]; !source_region.empty()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>read</i> B{} {}", agcmd->get_bid(), source_region);
			}
			fmt::format_to(std::back_inserter(label), "<br/><i>write</i> B{} {}", agcmd->get_bid(), merge_regions(agcmd->get_source_regions()));
			if(const auto& coherence = agcmd->get_local_coherence_region(); !coherence.empty()) {
				fmt::format_to(std::back_inserter(label), "<br/>D* <i>make coherent</i> {}", coherence);
			}
		} else if(const auto bcmd = dynamic_cast<const broadcast_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>broadcast</b> from N{}", bcmd->get_root());
			if(bcmd->get_root() == bcmd->get_nid()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>read</i> B{} {}", bcmd->get_bid(), bcmd->get_region());
			}
			fmt::format_to(std::back_inserter(label), "<br/><i>write</i> B{} {}", bcmd->get_bid(), bcmd->get_region());
		} else if(const auto scmd = dynamic_cast<const scatter_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>scatter</b> from N{}", scmd->get_root());
			if(scmd->get_root() == scmd->get_nid()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>read</i> B{} {}", scmd->get_bid(), merge_regions(scmd->get_dest_regions()));
			} else if(const auto& dest_region = scmd->get_dest_regions()[scmd->get_nid()]; !dest_region.empty()) {
				fmt::format_to(std::back_inserter(label), "<br/><i>write</i> B{} {}", scmd->get_bid(), dest_region);
			}
			for(device_id did = 0; did < scmd->get_local_device_coherence_regions().size(); ++did) {
				const auto& coherence = scmd->get_local_device_coherence_regions()[did];
				if(!coherence.empty()) { fmt::format_to(std::back_inserter(label), "<br/>D{} <i>make coherent</i> {}", did, coherence); }
			}
		} else if(const auto a2acmd = dynamic_cast<const alltoall_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>all-to-all</b> B{}", a2acmd->get_bid());
			for(node_id from_nid = 0; from_nid < a2acmd->get_send_regions().size(); ++from_nid) {
				const auto& sends = a2acmd->get_send_regions()[from_nid];
				if(!sends.empty()) { fmt::format_to(std::back_inserter(label), "<br/>N{} &lt;- <i>read</i> {}", from_nid, sends); }
			}
			for(node_id to_nid = 0; to_nid < a2acmd->get_recv_regions().size(); ++to_nid) {
				const auto& recvs = a2acmd->get_recv_regions()[to_nid];
				if(!recvs.empty()) { fmt::format_to(std::back_inserter(label), "<br/>N{} -&gt; <i>write</i> {}", to_nid, recvs); }
			}
			for(device_id did = 0; did < a2acmd->get_local_device_coherence_regions().size(); ++did) {
				const auto& coherence = a2acmd->get_local_device_coherence_regions()[did];
				if(!coherence.empty()) { fmt::format_to(std::back_inserter(label), "<br/>D{} <i>make coherent</i> {}", did, coherence); }
			}
		} else {
			assert(!"Unkown command");
			label += "<b>unknown</b>";
		}

		if(const auto tcmd = dynamic_cast<const task_command*>(&cmd)) {
			if(!tm.has_task(tcmd->get_tid())) return label; // NOCOMMIT This is only needed while we do TDAG pruning but not CDAG pruning
			assert(tm.has_task(tcmd->get_tid()));

			const auto tsk = tm.get_task(tcmd->get_tid());
			if(const auto* cgtsk = dynamic_cast<const command_group_task*>(tsk)) {
				auto reduction_init_mode = access_mode::discard_write;
				auto execution_range = subrange<3>{cgtsk->get_global_offset(), cgtsk->get_global_size()};
				if(const auto ecmd = dynamic_cast<const execution_command*>(&cmd)) {
					if(ecmd->is_reduction_initializer()) { reduction_init_mode = cl::sycl::access::mode::read_write; }
					execution_range = ecmd->get_execution_range();
				}
				format_requirements(label, *cgtsk, execution_range, reduction_init_mode, bm);
			}
		}

		return label;
	}

	std::string print_task_reference_label(const task_id tid, const task_manager& tm) {
		std::string task_label;
		fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
		if(const auto tsk = tm.find_task(tid)) {
			const auto cgtsk = dynamic_cast<const command_group_task*>(tsk);
			if(cgtsk && !cgtsk->get_debug_name().empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", cgtsk->get_debug_name()); }
			task_label += "(";
			task_label += task_type_string(tsk->get_type());
			if(cgtsk && cgtsk->get_type() == task_type::host_collective) {
				fmt::format_to(std::back_inserter(task_label), " on CG{}", cgtsk->get_collective_group_id());
			}
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
			// 	fmt::format_to(std::back_inserter(main_dot), "{}->{}[style=dashed color=gray40];", local_to_global_id(apcmd->get_root()->get_cid()),
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

	std::string print_instruction_graph(const instruction_graph& idag, const std::vector<pilot_message>& pilots) {
		std::string dot = "digraph G{label=\"Instruction Graph\";";
		const auto back = std::back_inserter(dot);

		const auto begin_node = [&](const instruction& instr, const std::string_view& shape) {
			fmt::format_to(back, "I{}[", instr.get_id());
			switch(instr.get_backend()) {
			case instruction_backend::host: break;
			case instruction_backend::mpi: dot += "color=deeppink2,"; break;
			case instruction_backend::sycl: dot += "color=darkorange2,"; break;
			case instruction_backend::cuda: dot += "color=green3,"; break;
			}
			fmt::format_to(back, "shape={},label=<{} ", shape, instr.get_backend());
		};

		const auto end_node = [&] { fmt::format_to(back, ">];"); };

		const auto print_buffer_range = [&](const buffer_id bid, const std::string& debug_name, const GridBox<3>& box) {
			if(debug_name.empty()) {
				fmt::format_to(back, "B{} {}", bid, box);
			} else {
				fmt::format_to(back, "{} (B{}) {}", debug_name /* TODO escape? */, bid, box);
			}
		};

		const auto print_buffer_allocation = [&](const buffer_allocation_info& ba) { print_buffer_range(ba.bid, ba.debug_name, ba.box); };

		std::unordered_map<int, instruction_id> send_instructions_by_tag; // for connecting pilot messages to send instructions
		idag.for_each([&](const instruction& instr) {
			utils::match(
			    instr,
			    [&](const alloc_instruction& ainstr) {
				    begin_node(ainstr, "ellipse");
				    fmt::format_to(back, "I{}<br/>", ainstr.get_id());
				    const auto debug_info = ainstr.get_debug_info();
				    if(debug_info) {
					    switch(debug_info->origin) {
					    case alloc_instruction_debug_info::alloc_origin::send: dot += "send "; break;
					    case alloc_instruction_debug_info::alloc_origin::buffer: dot += "buffer "; break;
					    }
				    }
				    fmt::format_to(back, "<b>alloc</b> A{} on M{}", ainstr.get_allocation_id(), ainstr.get_memory_id());
				    if(debug_info && debug_info->buffer_allocation) {
					    dot += "<br/>for ";
					    print_buffer_allocation(*debug_info->buffer_allocation);
				    }
				    fmt::format_to(back, "<br/>{}%{} bytes", ainstr.get_size(), ainstr.get_alignment());
				    end_node();
			    },
			    [&](const free_instruction& finstr) {
				    begin_node(finstr, "ellipse");
				    fmt::format_to(back, "I{}<br/>", finstr.get_id());
				    fmt::format_to(back, "<b>free</b> A{}", finstr.get_allocation_id());
				    if(const auto debug_info = finstr.get_debug_info()) {
					    fmt::format_to(back, " on M{}", debug_info->mid);
					    if(debug_info->buffer_allocation) {
						    dot += "<br/>";
						    print_buffer_allocation(*debug_info->buffer_allocation);
					    }
					    fmt::format_to(back, " <br/>{}%{} bytes", debug_info->size, debug_info->alignment);
				    }
				    end_node();
			    },
			    [&](const copy_instruction& cinstr) {
				    begin_node(cinstr, "ellipse");
				    fmt::format_to(back, "I{}<br/>", cinstr.get_id());
				    const auto debug_info = cinstr.get_debug_info();
				    fmt::format_to(back, "{}D ", cinstr.get_dimensions());
				    if(debug_info) {
					    switch(debug_info->origin) {
					    case copy_instruction_debug_info::copy_origin::linearize: dot += "linearizing "; break;
					    case copy_instruction_debug_info::copy_origin::resize: dot += "resize "; break;
					    case copy_instruction_debug_info::copy_origin::coherence: dot += "coherence "; break;
					    }
				    }
				    dot += "<b>copy</b>";
				    if(debug_info) {
					    dot += "<br/>on ";
					    print_buffer_range(debug_info->buffer, debug_info->buffer_debug_name, debug_info->box);
				    }
				    fmt::format_to(back, "<br/>from M{}.A{} ([{},{},{}]) +[{},{},{}]<br/>to M{}.A{} ([{},{},{}]) +[{},{},{}]<br/>[{},{},{}]x{} bytes",
				        cinstr.get_source_memory(), cinstr.get_source_allocation(), cinstr.get_source_range()[0], cinstr.get_source_range()[1],
				        cinstr.get_source_range()[2], cinstr.get_offset_in_source()[0], cinstr.get_offset_in_source()[1], cinstr.get_offset_in_source()[2],
				        cinstr.get_dest_memory(), cinstr.get_dest_allocation(), cinstr.get_dest_range()[0], cinstr.get_dest_range()[1],
				        cinstr.get_dest_range()[2], cinstr.get_offset_in_dest()[0], cinstr.get_offset_in_dest()[1], cinstr.get_offset_in_dest()[2],
				        cinstr.get_copy_range()[0], cinstr.get_copy_range()[1], cinstr.get_copy_range()[2], cinstr.get_element_size());
				    end_node();
			    },
			    [&](const kernel_instruction& kinstr) {
				    begin_node(kinstr, "box,margin=0.1");
				    fmt::format_to(back, "I{}", kinstr.get_id());
				    const auto debug_info = kinstr.get_debug_info();
				    if(debug_info) {
					    // TODO does not correctly label master-node / collective host tasks
					    fmt::format_to(back, " ({}-compute T{}, exectuion C{})", isa<sycl_kernel_instruction>(&kinstr) ? "device" : "host", debug_info->cg_tid,
					        debug_info->execution_cid);
				    }
				    dot += "<br/><b>kernel</b>";
				    if(debug_info && !debug_info->kernel_debug_name.empty()) {
					    dot += " ";
					    dot += debug_info->kernel_debug_name; // TODO escape?
				    }
				    if(const auto dkinstr = dynamic_cast<const sycl_kernel_instruction*>(&kinstr)) {
					    fmt::format_to(back, " on D{}", dkinstr->get_device_id());
				    }
				    fmt::format_to(back, " {}", kinstr.get_execution_range());

				    const auto& amap = kinstr.get_allocation_map();
				    for(size_t i = 0; i < amap.size(); ++i) {
					    dot += "<br/>";
					    if(debug_info) {
						    assert(amap.size() == debug_info->allocation_buffer_map.size());
						    print_buffer_allocation(debug_info->allocation_buffer_map[i]);
						    dot += " via ";
					    }
					    fmt::format_to(back, "A{} ([{},{},{}]) +[{},{},{}]", amap[i].aid, amap[i].allocation_range[0], amap[i].allocation_range[1],
					        amap[i].allocation_range[2], amap[i].offset_in_allocation[0], amap[i].offset_in_allocation[1], amap[i].offset_in_allocation[2]);
				    }
				    end_node();
			    },
			    [&](const send_instruction& sinstr) {
				    begin_node(sinstr, "box");
				    fmt::format_to(back, "I{}", sinstr.get_id());
				    if(const auto debug_info = sinstr.get_debug_info()) { fmt::format_to(back, " (push C{})", debug_info->push_cid); }
				    fmt::format_to(back, "<br/><b>send</b> to N{} tag {}", sinstr.get_dest_node_id(), sinstr.get_tag());
				    if(const auto debug_info = sinstr.get_debug_info()) {
					    fmt::format_to(back, " (C{})", debug_info->push_cid);
					    dot += "<br/>";
					    print_buffer_range(debug_info->buffer, debug_info->buffer_debug_name, debug_info->box);
				    }
				    fmt::format_to(back, "<br/>A{}, {} bytes", sinstr.get_allocation_id(), sinstr.get_size_bytes());
				    send_instructions_by_tag.emplace(sinstr.get_tag(), sinstr.get_id());
				    end_node();
			    },
			    [&](const recv_instruction& rinstr) {
				    begin_node(rinstr, "box");
				    fmt::format_to(back, "I{}", rinstr.get_id());
				    const auto debug_info = rinstr.get_debug_info();
				    if(debug_info) { fmt::format_to(back, " (await-push C{})", debug_info->await_push_cid); }
				    fmt::format_to(back, "<br/><b>recv</b> transfer {}", rinstr.get_transfer_id());
				    dot += "<br/>";
				    if(debug_info) {
					    print_buffer_range(debug_info->buffer, debug_info->buffer_debug_name,
					        subrange_to_grid_box(subrange(rinstr.get_offset_in_buffer(), rinstr.get_recv_range())));
				    } else {
					    fmt::format_to(back, "B? {}", subrange(rinstr.get_offset_in_buffer(), rinstr.get_recv_range()));
				    }
				    fmt::format_to(back, "<br/>to A{} ([{},{},{}]) +[{},{},{}], {}D [{},{},{}]x{} bytes", rinstr.get_dest_allocation_id(),
				        rinstr.get_allocation_range()[0], rinstr.get_allocation_range()[1], rinstr.get_allocation_range()[2],
				        rinstr.get_offset_in_allocation()[0], rinstr.get_offset_in_allocation()[1], rinstr.get_offset_in_allocation()[2],
				        rinstr.get_dimensions(), rinstr.get_recv_range()[0], rinstr.get_recv_range()[1], rinstr.get_recv_range()[2], rinstr.get_element_size());
				    end_node();
			    },
			    [&](const horizon_instruction& hinstr) {
				    begin_node(hinstr, "box,margin=0.1");
				    fmt::format_to(back, "I{} (T{}", hinstr.get_id(), hinstr.get_horizon_task_id());
				    if(const auto debug_info = hinstr.get_debug_info()) { fmt::format_to(back, " , C{}", debug_info->horizon_cid); }
				    dot += ")<br/><b>horizon</b>";
				    end_node();
			    },
			    [&](const epoch_instruction& einstr) {
				    begin_node(einstr, "box,margin=0.1");
				    fmt::format_to(back, "I{} (T{}", einstr.get_id(), einstr.get_epoch_task_id());
				    if(const auto debug_info = einstr.get_debug_info()) { fmt::format_to(back, " , C{}", debug_info->epoch_cid); }
				    dot += ")<br/><b>epoch</b>";
				    end_node();
			    });
		});

		idag.for_each([&](const instruction& instr) {
			for(const auto& dep : instr.get_dependencies()) {
				fmt::format_to(back, "I{}->I{}[{}];", dep.node->get_id(), instr.get_id(), dependency_style(dep));
			}
		});

		for(const auto& pilot : pilots) {
			fmt::format_to(back, "P{}[margin=0.2,shape=cds,color=\"#606060\",label=<<font color=\"#606060\"><b>pilot</b> tag {}<br/>for B{} {}</font>>];",
			    pilot.tag, pilot.tag, pilot.buffer, pilot.box);
			if(auto it = send_instructions_by_tag.find(pilot.tag); it != send_instructions_by_tag.end()) {
				fmt::format_to(back, "P{}->I{}[dir=none,style=dashed,color=\"#606060\"];", pilot.tag, it->second);
			}
		}

		dot += "}";
		return dot;
	}

} // namespace detail
} // namespace celerity
