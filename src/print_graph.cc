#include "print_graph.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "instruction_graph.h"
#include "recorders.h"
#include "task.h"
#include "task_manager.h"

namespace celerity::detail {

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
	case task_type::fence: return "fence";
	default: return "unknown";
	}
}

const char* command_type_string(const command_type ct) {
	switch(ct) {
	// default: return "unknown";
	case command_type::epoch: return "epoch";
	case command_type::horizon: return "horizon";
	case command_type::execution: return "execution";
	case command_type::push: return "push";
	case command_type::await_push: return "await push";
	case command_type::reduction: return "reduction";
	default: return "unknown";
	}
}

std::string get_buffer_label(const buffer_id bid, const std::string& name = "") {
	// if there is no name defined, the name will be the buffer id.
	// if there is a name we want "id name"
	return !name.empty() ? fmt::format("B{} \"{}\"", bid, name) : fmt::format("B{}", bid);
}

void format_requirements(std::string& label, const reduction_list& reductions, const access_list& accesses, const side_effect_map& side_effects,
    const access_mode reduction_init_mode) {
	for(const auto& [rid, bid, buffer_name, init_from_buffer] : reductions) {
		auto rmode = init_from_buffer ? reduction_init_mode : cl::sycl::access::mode::discard_write;
		const region scalar_region(box<3>({0, 0, 0}, {1, 1, 1}));
		const std::string bl = get_buffer_label(bid, buffer_name);
		fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", rid, detail::access::mode_traits::name(rmode), bl, scalar_region);
	}

	for(const auto& [bid, buffer_name, mode, req] : accesses) {
		const std::string bl = get_buffer_label(bid, buffer_name);
		// While uncommon, we do support chunks that don't require access to a particular buffer at all.
		if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", detail::access::mode_traits::name(mode), bl, req); }
	}

	for(const auto& [hoid, order] : side_effects) {
		fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
	}
}

std::string get_task_label(const task_record& tsk) {
	std::string label;
	fmt::format_to(std::back_inserter(label), "T{}", tsk.tid);
	if(!tsk.debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", utils::escape_for_dot_label(tsk.debug_name)); }

	fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.type));
	if(tsk.type == task_type::host_compute || tsk.type == task_type::device_compute) {
		fmt::format_to(std::back_inserter(label), " {}", subrange<3>{tsk.geometry.global_offset, tsk.geometry.global_size});
	} else if(tsk.type == task_type::collective) {
		fmt::format_to(std::back_inserter(label), " in CG{}", tsk.cgid);
	}

	format_requirements(label, tsk.reductions, tsk.accesses, tsk.side_effect_map, access_mode::read_write);

	return label;
}

std::string make_graph_preamble(const std::string& title) { return fmt::format("digraph G{{label=\"{}\" ", title); }

std::string print_task_graph(const task_recorder& recorder, const std::string& title) {
	std::string dot = make_graph_preamble(title);

	CELERITY_DEBUG("print_task_graph, {} entries", recorder.get_tasks().size());

	for(const auto& tsk : recorder.get_tasks()) {
		const char* shape = tsk.type == task_type::epoch || tsk.type == task_type::horizon ? "ellipse" : "box style=rounded";
		fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk.tid, shape, get_task_label(tsk));
		for(auto d : tsk.dependencies) {
			fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node, tsk.tid, dependency_style(d));
		}
	}

	dot += "}";
	return dot;
}

const char* get_epoch_label(epoch_action action) {
	switch(action) {
	case epoch_action::none: return "<b>epoch</b>";
	case epoch_action::barrier: return "<b>epoch</b> (barrier)";
	case epoch_action::shutdown: return "<b>epoch</b> (shutdown)";
	}
}

std::string get_command_label(const node_id local_nid, const command_record& cmd) {
	const command_id cid = cmd.cid;

	std::string label = fmt::format("C{} on N{}<br/>", cid, local_nid);

	auto add_reduction_id_if_reduction = [&]() {
		if(cmd.reduction_id.has_value() && cmd.reduction_id != 0) { fmt::format_to(std::back_inserter(label), "(R{}) ", cmd.reduction_id.value()); }
	};
	const std::string buffer_label = cmd.buffer_id.has_value() ? get_buffer_label(cmd.buffer_id.value(), cmd.buffer_name) : "";

	switch(cmd.type) {
	case command_type::epoch: {
		label += get_epoch_label(cmd.epoch_action.value());
	} break;
	case command_type::execution: {
		fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", cmd.execution_range.value());
	} break;
	case command_type::push: {
		add_reduction_id_if_reduction();
		fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>B{} {}", cmd.transfer_id.value(), cmd.target.value(), buffer_label,
		    cmd.push_range.value());
	} break;
	case command_type::await_push: {
		add_reduction_id_if_reduction();
		fmt::format_to(std::back_inserter(label), "<b>await push</b> transfer {} <br/>{} {}", //
		    cmd.transfer_id.value(), buffer_label, cmd.await_region.value());
	} break;
	case command_type::reduction: {
		const region scalar_region(box<3>({0, 0, 0}, {1, 1, 1}));
		fmt::format_to(std::back_inserter(label), "<b>reduction</b> R{}<br/> {} {}", cmd.reduction_id.value(), buffer_label, scalar_region);
	} break;
	case command_type::horizon: {
		label += "<b>horizon</b>";
	} break;
	case command_type::fence: {
		label += "<b>fence</b>";
	} break;
	default: assert(!"Unkown command"); label += "<b>unknown</b>";
	}

	if(cmd.task_id.has_value() && cmd.task_geometry.has_value()) {
		auto reduction_init_mode = cmd.is_reduction_initializer ? cl::sycl::access::mode::read_write : access_mode::discard_write;

		format_requirements(label, cmd.reductions.value_or(reduction_list{}), cmd.accesses.value_or(access_list{}),
		    cmd.side_effects.value_or(side_effect_map{}), reduction_init_mode);
	}

	return label;
}

std::string print_command_graph(const node_id local_nid, const command_recorder& recorder, const std::string& title) {
	std::string main_dot;
	std::map<task_id, std::string> task_subgraph_dot; // this map must be ordered!

	const auto local_to_global_id = [local_nid](uint64_t id) {
		// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
		return fmt::format("id_{}_{}", local_nid, id);
	};

	const auto print_vertex = [&](const command_record& cmd) {
		static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

		const auto id = local_to_global_id(cmd.cid);
		const auto label = get_command_label(local_nid, cmd);
		const auto* const fontcolor = colors[local_nid % (sizeof(colors) / sizeof(char*))];
		const auto* const shape = cmd.task_id.has_value() ? "box" : "ellipse";
		return fmt::format("{}[label=<{}> fontcolor={} shape={}];", id, label, fontcolor, shape);
	};

	// we want to iterate over our command records in a sorted order, without moving everything around, and we aren't in C++20 (yet)
	std::vector<const command_record*> sorted_cmd_pointers;
	for(const auto& cmd : recorder.get_commands()) {
		sorted_cmd_pointers.push_back(&cmd);
	}
	std::sort(sorted_cmd_pointers.begin(), sorted_cmd_pointers.end(), [](const auto* a, const auto* b) { return a->cid < b->cid; });

	for(const auto& cmd : sorted_cmd_pointers) {
		if(cmd->task_id.has_value()) {
			const auto tid = cmd->task_id.value();
			// Add to subgraph as well
			if(task_subgraph_dot.count(tid) == 0) {
				std::string task_label;
				fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
				if(!cmd->task_name.empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", utils::escape_for_dot_label(cmd->task_name)); }
				task_label += "(";
				task_label += task_type_string(cmd->task_type.value());
				if(cmd->task_type == task_type::collective) { fmt::format_to(std::back_inserter(task_label), " on CG{}", cmd->collective_group_id.value()); }
				task_label += ")";

				task_subgraph_dot.emplace(
				    tid, fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;", local_to_global_id(tid), task_label));
			}
			task_subgraph_dot[tid] += print_vertex(*cmd);
		} else {
			main_dot += print_vertex(*cmd);
		}

		for(const auto& d : cmd->dependencies) {
			fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(d.node), local_to_global_id(cmd->cid), dependency_style(d));
		}
	};

	std::string result_dot = make_graph_preamble(title);
	for(auto& [_, sg_dot] : task_subgraph_dot) {
		result_dot += sg_dot;
		result_dot += "}";
	}
	result_dot += main_dot;
	result_dot += "}";
	return result_dot;
}

std::string combine_command_graphs(const std::vector<std::string>& graphs, const std::string& title) {
	const auto preamble = make_graph_preamble(title);
	std::string result_dot = make_graph_preamble(title);
	for(const auto& g : graphs) {
		result_dot += g.substr(preamble.size(), g.size() - preamble.size() - 1);
	}
	result_dot += "}";
	return result_dot;
}

std::string print_task_reference_label(const task_record& task) {
	std::string task_label;
	fmt::format_to(std::back_inserter(task_label), "T{} ", task.tid);
	if(!task.debug_name.empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", task.debug_name); }
	task_label += "(";
	task_label += task_type_string(task.type);
	if(task.type == task_type::collective) { fmt::format_to(std::back_inserter(task_label), " on CG{}", task.cgid); }
	task_label += ")";
	return task_label;
}

std::string print_command_reference_label(const command_record cmd, const task_recorder& trec) {
	std::string cmd_label;
	fmt::format_to(std::back_inserter(cmd_label), "C{} ", cmd.cid);
	fmt::format_to(std::back_inserter(cmd_label), "({})", command_type_string(cmd.type));
	if(cmd.task_id.has_value()) { cmd_label += "<br/>from " + print_task_reference_label(trec.get_task(*cmd.task_id)); }
	return cmd_label;
}

std::string print_instruction_graph(const instruction_recorder& irec, const command_recorder& crec, const task_recorder& trec, const std::string& title) {
	std::string dot = make_graph_preamble(title);
	const auto back = std::back_inserter(dot);

	const auto begin_node = [&](const instruction_record_base& instr, const std::string_view& shape, const std::string_view& color) {
		// TODO consider moving the task-reference / command-reference printing here.
		fmt::format_to(back, "I{}[color={},shape={},label=<", instr.id, color, shape);
	};

	const auto end_node = [&] { fmt::format_to(back, ">];"); };

	const auto print_buffer_range = [&](const buffer_id bid, const box<3>& box) {
		const auto& debug_name = irec.get_buffer_debug_name(bid);
		if(debug_name.empty()) {
			fmt::format_to(back, "B{} {}", bid, box);
		} else {
			fmt::format_to(back, "{} (B{}) {}", debug_name /* TODO escape? */, bid, box);
		}
	};

	const auto print_buffer_allocation = [&](const buffer_allocation_record& ba) { print_buffer_range(ba.buffer_id, ba.box); };

	std::unordered_map<int, instruction_id> send_instructions_by_tag; // for connecting pilot messages to send instructions
	for(const auto& instr : irec.get_instructions()) {
		matchbox::match(
		    instr,
		    [&](const alloc_instruction_record& ainstr) {
			    begin_node(ainstr, "ellipse", "cyan3");
			    fmt::format_to(back, "I{}<br/>", ainstr.id);
			    switch(ainstr.origin) {
			    case alloc_instruction_record::alloc_origin::send: dot += "send "; break;
			    case alloc_instruction_record::alloc_origin::buffer: dot += "buffer "; break;
			    }
			    fmt::format_to(back, "<b>alloc</b> A{} on M{}", ainstr.allocation_id, ainstr.memory_id);
			    if(ainstr.buffer_allocation.has_value()) {
				    dot += "<br/>for ";
				    print_buffer_allocation(*ainstr.buffer_allocation);
			    }
			    fmt::format_to(back, "<br/>{}%{} bytes", ainstr.size, ainstr.alignment);
			    end_node();
		    },
		    [&](const free_instruction_record& finstr) {
			    begin_node(finstr, "ellipse", "cyan3");
			    fmt::format_to(back, "I{}<br/>", finstr.id);
			    fmt::format_to(back, "<b>free</b> A{} on M{}", finstr.allocation_id, finstr.memory_id);
			    if(finstr.buffer_allocation.has_value()) {
				    dot += "<br/>";
				    print_buffer_allocation(*finstr.buffer_allocation);
			    }
			    fmt::format_to(back, " <br/>{}%{} bytes", finstr.size, finstr.alignment);
			    end_node();
		    },
		    [&](const init_buffer_instruction_record& ibinstr) {
			    begin_node(ibinstr, "ellipse", "green3");
			    fmt::format_to(back, "I{}<br/>", ibinstr.id);
			    fmt::format_to(back, "<b>init buffer</b> B{}<br/>via M0.A{}, {} bytes", ibinstr.buffer_id, ibinstr.host_allocation_id, ibinstr.size);
			    end_node();
		    },
		    [&](const export_instruction_record& einstr) {
			    begin_node(einstr, "ellipse", "green3");
			    fmt::format_to(back, "I{}<br/>", einstr.id);
			    fmt::format_to(back, "<b>export</b> from M0.A{} ({})+{}, {}D {}x{} bytes", einstr.host_allocation_id, einstr.allocation_range,
			        einstr.offset_in_allocation, einstr.dimensions, einstr.copy_range, einstr.element_size);
			    end_node();
		    },
		    [&](const copy_instruction_record& cinstr) {
			    begin_node(cinstr, "ellipse", "green3");
			    fmt::format_to(back, "I{}<br/>", cinstr.id);
			    fmt::format_to(back, "{}D ", cinstr.dimensions);
			    switch(cinstr.origin) {
			    case copy_instruction_record::copy_origin::linearize: dot += "linearizing "; break;
			    case copy_instruction_record::copy_origin::resize: dot += "resize "; break;
			    case copy_instruction_record::copy_origin::coherence: dot += "coherence "; break;
			    }
			    dot += "<b>copy</b><br/>on ";
			    print_buffer_range(cinstr.buffer, cinstr.box);
			    fmt::format_to(back, "<br/>from M{}.A{} ({}) +{}<br/>to M{}.A{} ({}) +{}<br/>{}x{} bytes", cinstr.source_memory, cinstr.source_allocation,
			        cinstr.source_range, cinstr.offset_in_source, cinstr.dest_memory, cinstr.dest_allocation, cinstr.dest_range, cinstr.offset_in_dest,
			        cinstr.copy_range, cinstr.element_size);
			    end_node();
		    },
		    [&](const kernel_instruction_record& kinstr) {
			    begin_node(kinstr, "box,margin=0.1", "darkorange2");
			    fmt::format_to(back, "I{}", kinstr.id);
			    // TODO does not correctly label master-node / collective host tasks
			    fmt::format_to(back, " ({}-compute T{}, exectuion C{})", kinstr.target == execution_target::device ? "device" : "host",
			        kinstr.command_group_task_id, kinstr.execution_command_id);
			    fmt::format_to(back, "<br/><b>{} kernel</b>", kinstr.target == execution_target::device ? "device" : "host");
			    if(!kinstr.kernel_debug_name.empty()) { fmt::format_to(back, " {}", kinstr.kernel_debug_name /* TODO escape? */); }
			    if(kinstr.device_id.has_value()) { fmt::format_to(back, " on D{}", *kinstr.device_id); }
			    fmt::format_to(back, " {}", kinstr.execution_range);

			    const auto& amap = kinstr.allocation_map;
			    for(size_t i = 0; i < amap.size(); ++i) {
				    dot += "<br/>";
				    assert(amap.size() == kinstr.allocation_buffer_map.size()); // TODO why separate structs?
				    print_buffer_allocation(kinstr.allocation_buffer_map[i]);
				    dot += " via ";
				    const auto accessed_box_in_allocation = box(amap[i].accessed_box_in_buffer.get_min() - amap[i].allocated_box_in_buffer.get_min(),
				        amap[i].accessed_box_in_buffer.get_max() - amap[i].allocated_box_in_buffer.get_max());
				    fmt::format_to(back, "A{} {}", amap[i].aid, accessed_box_in_allocation);
			    }
			    end_node();
		    },
		    [&](const send_instruction_record& sinstr) {
			    begin_node(sinstr, "box", "deeppink2");
			    fmt::format_to(back, "I{} (push C{})", sinstr.id, sinstr.push_cid);
			    fmt::format_to(back, "<br/><b>send</b> transfer {} to N{} tag {}<br/>", sinstr.transfer_id, sinstr.dest_node_id, sinstr.tag);
			    print_buffer_range(sinstr.buffer, subrange(sinstr.offset_in_buffer, sinstr.send_range));
			    fmt::format_to(back, "<br/>from A{} ({}) + {}, {}x{} bytes", sinstr.source_allocation_id, sinstr.allocation_range, sinstr.offset_in_allocation,
			        sinstr.send_range, sinstr.element_size);
			    send_instructions_by_tag.emplace(sinstr.tag, sinstr.id);
			    end_node();
		    },
		    [&](const recv_instruction_record& rinstr) {
			    begin_node(rinstr, "box", "deeppink2");
			    fmt::format_to(back, "I{} (await-push C{})", rinstr.id, irec.get_await_push_command_id(rinstr.transfer_id));
			    fmt::format_to(back, "<br/><b>recv</b> transfer {}<br/>", rinstr.transfer_id);
			    print_buffer_range(rinstr.buffer_id, subrange(rinstr.offset_in_buffer, rinstr.recv_range));
			    fmt::format_to(back, "<br/>to A{} ({}) +{}, {}x{} bytes", rinstr.dest_allocation_id, rinstr.allocation_range, rinstr.offset_in_allocation,
			        rinstr.recv_range, rinstr.element_size);
			    end_node();
		    },
		    [&](const fence_instruction_record& finstr) {
			    begin_node(finstr, "box,margin=0.1", "darkorange");
			    fmt::format_to(back, "I{} (T{}, C{})<br/><b>fence</b><br/>", finstr.id, finstr.tid, finstr.cid);
			    matchbox::match(
			        finstr.variant, //
			        [&](const fence_instruction_record::buffer_variant& buffer) { print_buffer_range(buffer.bid, buffer.box); },
			        [&](const fence_instruction_record::host_object_variant& obj) { fmt::format_to(back, "<br/>H{}", obj.hoid); });
			    end_node();
		    },
		    [&](const destroy_host_object_instruction_record& dhoinstr) {
			    begin_node(dhoinstr, "ellipse", "black");
			    fmt::format_to(back, "I{}<br/><b>destroy</b> H{}", dhoinstr.id, dhoinstr.host_object_id);
			    end_node();
		    },
		    [&](const horizon_instruction_record& hinstr) {
			    begin_node(hinstr, "box,margin=0.1", "black");
			    fmt::format_to(back, "I{} (T{}, C{})<br/><b>horizon</b>", hinstr.id, hinstr.horizon_task_id, hinstr.horizon_command_id);
			    end_node();
		    },
		    [&](const epoch_instruction_record& einstr) {
			    begin_node(einstr, "box,margin=0.1", "black");
			    fmt::format_to(back, "I{} (T{}, C{})<br/>{}", einstr.id, einstr.epoch_task_id, einstr.epoch_command_id, get_epoch_label(einstr.epoch_action));
			    end_node();
		    });
	}

	for(const auto& instr : irec.get_instructions()) {
		// since all instruction_records inherit from instruction_record_base, this *should* just compile to a pointer adjustment
		const auto& instr_base = matchbox::match(instr, [](const auto& i) -> const instruction_record_base& { return i; });

		for(const auto& dep : instr_base.dependencies) {
			fmt::format_to(back, "I{}->I{}[{}];", dep.node, instr_base.id, dependency_style(dep));
		}
	}

	for(const auto& pilot : irec.get_outbound_pilots()) {
		fmt::format_to(back,
		    "P{}[margin=0.2,shape=cds,color=\"#606060\",label=<<font color=\"#606060\"><b>pilot</b> to N{} tag {}<br/>transfer {}<br/>for B{} {}</font>>];",
		    pilot.message.tag, pilot.to, pilot.message.tag, pilot.message.transfer, pilot.message.buffer, pilot.message.box);
		if(auto it = send_instructions_by_tag.find(pilot.message.tag); it != send_instructions_by_tag.end()) {
			fmt::format_to(back, "P{}->I{}[dir=none,style=dashed,color=\"#606060\"];", pilot.message.tag, it->second);
		}
	}

	dot += "}";
	return dot;
}


} // namespace celerity::detail
