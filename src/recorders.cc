#include "recorders.h"
#include "command.h"
#include "task_manager.h"
#include "utils.h"

#include <regex>

namespace celerity::detail {

// Naming

std::string get_buffer_name(const buffer_id bid, const buffer_name_map& accessed_buffer_names) {
	if(const auto it = accessed_buffer_names.find(bid); it != accessed_buffer_names.end()) { return it->second; }
	return {};
}

// Tasks

access_list build_access_list(const task& tsk, const buffer_name_map& accessed_buffer_names, const std::optional<subrange<3>> execution_range = {}) {
	access_list ret;
	const auto exec_range = execution_range.value_or(subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(const auto mode : bam.get_access_modes(bid)) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), exec_range, tsk.get_global_size());
			ret.push_back({bid, get_buffer_name(bid, accessed_buffer_names), mode, req});
		}
	}
	return ret;
}

reduction_list build_reduction_list(const task& tsk, const buffer_name_map& accessed_buffer_names) {
	reduction_list ret;
	for(const auto& reduction : tsk.get_reductions()) {
		ret.push_back({reduction.rid, reduction.bid, get_buffer_name(reduction.bid, accessed_buffer_names), reduction.init_from_buffer});
	}
	return ret;
}

task_dependency_list build_task_dependency_list(const task& tsk) {
	task_dependency_list ret;
	for(const auto& dep : tsk.get_dependencies()) {
		ret.push_back({dep.node->get_id(), dep.kind, dep.origin});
	}
	return ret;
}

task_record::task_record(const task& from, const buffer_name_map& accessed_buffer_names)
    : tid(from.get_id()), debug_name(utils::simplify_task_name(from.get_debug_name())), cgid(from.get_collective_group_id()), type(from.get_type()),
      geometry(from.get_geometry()), reductions(build_reduction_list(from, accessed_buffer_names)), accesses(build_access_list(from, accessed_buffer_names)),
      side_effect_map(from.get_side_effect_map()), dependencies(build_task_dependency_list(from)) {}

// Commands

std::optional<epoch_action> get_epoch_action(const abstract_command& cmd) {
	const auto* epoch_cmd = dynamic_cast<const epoch_command*>(&cmd);
	return epoch_cmd != nullptr ? epoch_cmd->get_epoch_action() : std::optional<epoch_action>{};
}

std::optional<subrange<3>> get_execution_range(const abstract_command& cmd) {
	const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd);
	return execution_cmd != nullptr ? execution_cmd->get_execution_range() : std::optional<subrange<3>>{};
}

std::optional<reduction_id> get_reduction_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id().rid;
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id().rid;
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().rid;
	return {};
}

std::optional<buffer_id> get_buffer_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id().bid;
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id().bid;
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().bid;
	return {};
}

std::string get_cmd_buffer_name(const std::optional<buffer_id>& bid, const buffer_name_map& accessed_buffer_names) {
	if(bid.has_value()) return get_buffer_name(*bid, accessed_buffer_names);
	return {};
}

std::optional<node_id> get_target(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_target();
	return {};
}

std::optional<region<3>> get_await_region(const abstract_command& cmd) {
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_region();
	return {};
}

std::optional<subrange<3>> get_push_range(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_range();
	return {};
}

std::optional<transfer_id> get_transfer_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id();
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id();
	return {};
}

std::optional<task_id> get_task_id(const abstract_command& cmd) {
	if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) return task_cmd->get_tid();
	return {};
}

const task* get_task_for(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) {
		if(task_mngr != nullptr) {
			assert(task_mngr->has_task(task_cmd->get_tid()));
			return task_mngr->get_task(task_cmd->get_tid());
		}
	}
	return nullptr;
}

bool get_is_reduction_initializer(const abstract_command& cmd) {
	if(const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd)) return execution_cmd->is_reduction_initializer();
	return false;
}

access_list build_cmd_access_list(const abstract_command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names) {
	const auto execution_range_a = get_execution_range(cmd);
	const auto execution_range_b = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
	const auto execution_range = execution_range_a.value_or(execution_range_b);
	return build_access_list(tsk, accessed_buffer_names, execution_range);
}

command_dependency_list build_command_dependency_list(const abstract_command& cmd) {
	command_dependency_list ret;
	for(const auto& dep : cmd.get_dependencies()) {
		ret.push_back({dep.node->get_cid(), dep.kind, dep.origin});
	}
	return ret;
}

std::string get_task_name(const task& tsk) { return utils::simplify_task_name(tsk.get_debug_name()); }

std::optional<task_type> get_task_type(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_type();
	return {};
}

std::optional<collective_group_id> get_collective_group_id(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_collective_group_id();
	return {};
}

command_record::command_record(const abstract_command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names)
    : cid(cmd.get_cid()), type(cmd.get_type()), epoch_action(get_epoch_action(cmd)), execution_range(get_execution_range(cmd)),
      reduction_id(get_reduction_id(cmd)), buffer_id(get_buffer_id(cmd)), buffer_name(get_cmd_buffer_name(buffer_id, accessed_buffer_names)),
      target(get_target(cmd)), await_region(get_await_region(cmd)), push_range(get_push_range(cmd)), transfer_id(get_transfer_id(cmd)),
      task_id(get_task_id(cmd)), task_geometry(tsk.get_geometry()), is_reduction_initializer(get_is_reduction_initializer(cmd)),
      accesses(build_cmd_access_list(cmd, tsk, accessed_buffer_names)), reductions(build_reduction_list(tsk, accessed_buffer_names)),
      side_effects(tsk.get_side_effect_map()), dependencies(build_command_dependency_list(cmd)), task_name(get_task_name(tsk)), task_type(tsk.get_type()),
      collective_group_id(tsk.get_collective_group_id()) {}

// Instructions

instruction_record::instruction_record(const instruction& instr) : id(instr.get_id()) {}

clone_collective_group_instruction_record::clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr)
    : acceptor_base(ccginstr), origin_collective_group_id(ccginstr.get_origin_collective_group_id()),
      new_collective_group_id(ccginstr.get_new_collective_group_id()) {}

alloc_instruction_record::alloc_instruction_record(
    const alloc_instruction& ainstr, const alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation, std::optional<size_t> num_chunks)
    : acceptor_base(ainstr), allocation_id(ainstr.get_allocation_id()), memory_id(ainstr.get_memory_id()), size(ainstr.get_size()),
      alignment(ainstr.get_alignment()), origin(origin), buffer_allocation(buffer_allocation), num_chunks(num_chunks) {}

free_instruction_record::free_instruction_record(
    const free_instruction& finstr, const size_t size, const std::optional<buffer_allocation_record>& buffer_allocation)
    : acceptor_base(finstr), memory_id(finstr.get_memory_id()), allocation_id(finstr.get_allocation_id()), size(size), buffer_allocation(buffer_allocation) {}

init_buffer_instruction_record::init_buffer_instruction_record(const init_buffer_instruction& ibinstr)
    : acceptor_base(ibinstr), buffer_id(ibinstr.get_buffer_id()), host_allocation_id(ibinstr.get_host_allocation_id()), size(ibinstr.get_size()) {}

export_instruction_record::export_instruction_record(const export_instruction& einstr, const buffer_id buffer, const celerity::id<3>& offset_in_buffer)
    : acceptor_base(einstr), buffer(buffer), offset_in_buffer(offset_in_buffer), host_allocation_id(einstr.get_host_allocation_id()),
      dimensions(einstr.get_dimensions()), allocation_range(einstr.get_allocation_range()), offset_in_allocation(einstr.get_offset_in_allocation()),
      copy_range(einstr.get_copy_range()), element_size(einstr.get_element_size()) {}

copy_instruction_record::copy_instruction_record(const copy_instruction& cinstr, const copy_origin origin, const buffer_id buffer, const detail::box<3>& box)
    : acceptor_base(cinstr), source_memory(cinstr.get_source_memory()), source_allocation(cinstr.get_source_allocation()),
      dest_memory(cinstr.get_dest_memory()), dest_allocation(cinstr.get_dest_allocation()), dimensions(cinstr.get_dimensions()),
      source_range(cinstr.get_source_range()), dest_range(cinstr.get_dest_range()), offset_in_source(cinstr.get_offset_in_source()),
      offset_in_dest(cinstr.get_offset_in_dest()), copy_range(cinstr.get_copy_range()), element_size(cinstr.get_element_size()), origin(origin), buffer(buffer),
      box(box) {}

launch_instruction_record::launch_instruction_record(const launch_instruction& linstr, const task_id cg_tid, const command_id execution_cid,
    const std::string& debug_name, const std::vector<buffer_memory_allocation_record>& buffer_memory_allocation_map,
    const std::vector<buffer_memory_reduction_record>& buffer_memory_reduction_map)
    : acceptor_base(linstr), target(utils::isa<host_task_instruction>(&linstr) ? execution_target::host : execution_target::device),
      device_id(matchbox::match<std::optional<detail::device_id>>(
          linstr, [](const sycl_kernel_instruction& skinstr) { return skinstr.get_device_id(); }, [](const auto&) { return std::nullopt; })),
      collective_group_id(matchbox::match<std::optional<detail::collective_group_id>>(
          linstr, [](const host_task_instruction& htinstr) { return htinstr.get_collective_group_id(); }, [](const auto&) { return std::nullopt; })),
      execution_range(linstr.get_execution_range()), command_group_task_id(cg_tid), execution_command_id(execution_cid),
      debug_name(utils::simplify_task_name(debug_name)) //
{
	assert(linstr.get_access_allocations().size() == buffer_memory_allocation_map.size());
	access_map.reserve(linstr.get_access_allocations().size());
	for(size_t i = 0; i < linstr.get_access_allocations().size(); ++i) {
		access_map.emplace_back(linstr.get_access_allocations()[i], buffer_memory_allocation_map[i]);
	}
	if(const auto skinstr = dynamic_cast<const sycl_kernel_instruction*>(&linstr)) {
		assert(skinstr->get_reduction_allocations().size() == buffer_memory_reduction_map.size());
		reduction_map.reserve(skinstr->get_reduction_allocations().size());
		for(size_t i = 0; i < skinstr->get_reduction_allocations().size(); ++i) {
			reduction_map.emplace_back(skinstr->get_reduction_allocations()[i], buffer_memory_reduction_map[i]);
		}
	}
}

send_instruction_record::send_instruction_record(
    const send_instruction& sinstr, const command_id push_cid, const detail::transfer_id& trid, const celerity::id<3>& offset_in_buffer)
    : acceptor_base(sinstr), dest_node_id(sinstr.get_dest_node_id()), tag(sinstr.get_tag()), source_memory_id(sinstr.get_source_memory_id()),
      source_allocation_id(sinstr.get_source_allocation_id()), allocation_range(sinstr.get_allocation_range()),
      offset_in_allocation(sinstr.get_offset_in_allocation()), send_range(sinstr.get_send_range()), element_size(sinstr.get_element_size()), push_cid(push_cid),
      transfer_id(trid), offset_in_buffer(offset_in_buffer) {}

receive_instruction_record_impl::receive_instruction_record_impl(const receive_instruction_impl& rinstr)
    : transfer_id(rinstr.get_transfer_id()), requested_region(rinstr.get_requested_region()), dest_memory(rinstr.get_dest_memory()),
      dest_allocation(rinstr.get_dest_allocation()), allocated_box(rinstr.get_allocated_box()), element_size(rinstr.get_element_size()) {}

receive_instruction_record::receive_instruction_record(const receive_instruction& rinstr) : acceptor_base(rinstr), receive_instruction_record_impl(rinstr) {}

split_receive_instruction_record::split_receive_instruction_record(const split_receive_instruction& srinstr)
    : acceptor_base(srinstr), receive_instruction_record_impl(srinstr) {}

await_receive_instruction_record::await_receive_instruction_record(const await_receive_instruction& arinstr)
    : acceptor_base(arinstr), transfer_id(arinstr.get_transfer_id()), received_region(arinstr.get_received_region()) {}

gather_receive_instruction_record::gather_receive_instruction_record(const gather_receive_instruction& grinstr, const box<3>& gather_box, size_t num_nodes)
    : acceptor_base(grinstr), transfer_id(grinstr.get_transfer_id()), memory_id(grinstr.get_memory_id()), allocation_id(grinstr.get_allocation_id()),
      node_chunk_size(grinstr.get_node_chunk_size()), gather_box(gather_box), num_nodes(num_nodes) {}

fill_identity_instruction_record::fill_identity_instruction_record(const fill_identity_instruction& fiinstr)
    : acceptor_base(fiinstr), reduction_id(fiinstr.get_reduction_id()), memory_id(fiinstr.get_memory_id()), allocation_id(fiinstr.get_allocation_id()),
      num_values(fiinstr.get_num_values()) {}

reduce_instruction_record::reduce_instruction_record(const reduce_instruction& rinstr, const std::optional<detail::command_id> reduction_cid,
    const detail::buffer_id bid, const detail::box<3>& box, const reduction_scope scope)
    : acceptor_base(rinstr), reduction_id(rinstr.get_reduction_id()), memory_id(rinstr.get_memory_id()),
      source_allocation_id(rinstr.get_source_allocation_id()), num_source_values(rinstr.get_num_source_values()),
      dest_allocation_id(rinstr.get_dest_allocation_id()), reduction_command_id(reduction_cid), buffer_id(bid), box(box), scope(scope) {}

fence_instruction_record::fence_instruction_record(const fence_instruction& finstr, task_id tid, const command_id cid, const buffer_id bid, const box<3>& box)
    : acceptor_base(finstr), tid(tid), cid(cid), variant(buffer_variant{bid, box}) {}

fence_instruction_record::fence_instruction_record(const fence_instruction& finstr, task_id tid, const command_id cid, const host_object_id hoid)
    : acceptor_base(finstr), tid(tid), cid(cid), variant(host_object_variant{hoid}) {}

destroy_host_object_instruction_record::destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr)
    : acceptor_base(dhoinstr), host_object_id(dhoinstr.get_host_object_id()) {}

horizon_instruction_record::horizon_instruction_record(const horizon_instruction& hinstr, const command_id horizon_cid)
    : acceptor_base(hinstr), horizon_task_id(hinstr.get_horizon_task_id()), horizon_command_id(horizon_cid) {}

epoch_instruction_record::epoch_instruction_record(const epoch_instruction& einstr, const command_id epoch_cid)
    : acceptor_base(einstr), epoch_task_id(einstr.get_epoch_task_id()), epoch_command_id(epoch_cid), epoch_action(einstr.get_epoch_action()) {}

command_id instruction_recorder::get_await_push_command_id(const transfer_id& trid) const { return m_await_push_cids.at(trid); }

const std::string& instruction_recorder::get_buffer_debug_name(const buffer_id bid) const {
	if(const auto it = m_buffer_debug_names.find(bid); it != m_buffer_debug_names.end()) { return it->second; }
	static const std::string m_empty_debug_name;
	return m_empty_debug_name;
}

void instruction_recorder::record_dependencies(const instruction& instr) {
	auto& record = const_cast<instruction_record&>(get_instruction(instr.get_id()));
	for(auto& d : instr.get_dependencies()) {
		record.dependencies.push_back(dependency_record<instruction_id>{d.node->get_id(), d.kind, d.origin});
	}
}

} // namespace celerity::detail
