#pragma once

#include "command.h"
#include "instruction_graph.h"
#include "task.h"

namespace celerity::detail {

class task_manager;

// General recording

struct access_record {
	const buffer_id bid;
	const std::string buffer_name;
	const access_mode mode;
	const region<3> req;
};
using access_list = std::vector<access_record>;
using buffer_name_map = std::unordered_map<buffer_id, std::string>;

struct reduction_record {
	const reduction_id rid;
	const buffer_id bid;
	const std::string buffer_name;
	const bool init_from_buffer;
};
using reduction_list = std::vector<reduction_record>;

template <typename IdType>
struct dependency_record {
	const IdType node;
	const dependency_kind kind;
	const dependency_origin origin;
};

// Task recording

using task_dependency_list = std::vector<dependency_record<task_id>>;

struct task_record {
	task_record(const task& tsk, const buffer_name_map& accessed_buffer_names);

	task_id tid;
	std::string debug_name;
	collective_group_id cgid;
	task_type type;
	task_geometry geometry;
	reduction_list reductions;
	access_list accesses;
	side_effect_map side_effect_map;
	task_dependency_list dependencies;
};

class task_recorder {
  public:
	using task_records = std::vector<task_record>;

	void record_task(task_record&& record) { m_recorded_tasks.push_back(std::move(record)); }

	friend task_recorder& operator<<(task_recorder& recorder, task_record&& record) {
		recorder.record_task(std::move(record));
		return recorder;
	}

	const task_records& get_tasks() const { return m_recorded_tasks; }

	const task_record& get_task(const task_id tid) const {
		const auto it = std::find_if(m_recorded_tasks.begin(), m_recorded_tasks.end(), [tid](const task_record& rec) { return rec.tid == tid; });
		assert(it != m_recorded_tasks.end());
		return *it;
	}

  private:
	task_records m_recorded_tasks;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	command_id cid;
	command_type type;

	std::optional<epoch_action> epoch_action;
	std::optional<subrange<3>> execution_range;
	std::optional<reduction_id> reduction_id;
	std::optional<buffer_id> buffer_id;
	std::string buffer_name;
	std::optional<node_id> target;
	std::optional<region<3>> await_region;
	std::optional<subrange<3>> push_range;
	std::optional<transfer_id> transfer_id;
	std::optional<task_id> task_id;
	std::optional<task_geometry> task_geometry;
	bool is_reduction_initializer;
	std::optional<access_list> accesses;
	std::optional<reduction_list> reductions;
	std::optional<side_effect_map> side_effects;
	command_dependency_list dependencies;
	std::string task_name;
	std::optional<task_type> task_type;
	std::optional<collective_group_id> collective_group_id;
	std::vector<detail::reduction_id> completed_reductions;

	command_record(const abstract_command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names);
};

class command_recorder {
  public:
	using command_records = std::vector<command_record>;

	void record_command(command_record&& record) { m_recorded_commands.push_back(std::move(record)); }

	friend command_recorder& operator<<(command_recorder& recorder, command_record&& record) {
		recorder.record_command(std::move(record));
		return recorder;
	}

	const command_records& get_commands() const { return m_recorded_commands; }

	const command_record& get_command(const command_id cid) const {
		const auto it = std::find_if(m_recorded_commands.begin(), m_recorded_commands.end(), [cid](const command_record& rec) { return rec.cid == cid; });
		assert(it != m_recorded_commands.end());
		return *it;
	}

  private:
	command_records m_recorded_commands;
};

struct buffer_allocation_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	box<3> box;

	friend bool operator==(const buffer_allocation_record& lhs, const buffer_allocation_record& rhs) {
		return lhs.buffer_id == rhs.buffer_id && lhs.buffer_name == rhs.buffer_name && lhs.box == rhs.box;
	}
	friend bool operator!=(const buffer_allocation_record& lhs, const buffer_allocation_record& rhs) { return !(lhs == rhs); }
};

using instruction_dependency_list = std::vector<dependency_record<instruction_id>>;

struct instruction_record
    : matchbox::acceptor<struct clone_collective_group_instruction_record, struct alloc_instruction_record, struct free_instruction_record,
          struct init_buffer_instruction_record, struct export_instruction_record, struct copy_instruction_record, struct device_kernel_instruction_record,
          struct host_task_instruction_record, struct send_instruction_record, struct receive_instruction_record, struct split_receive_instruction_record,
          struct await_receive_instruction_record, struct gather_receive_instruction_record, struct fill_identity_instruction_record,
          struct reduce_instruction_record, struct fence_instruction_record, struct destroy_host_object_instruction_record, struct horizon_instruction_record,
          struct epoch_instruction_record> //
{
	instruction_id id;
	instruction_dependency_list dependencies;

	explicit instruction_record(const instruction& instr);
};

struct clone_collective_group_instruction_record : matchbox::implement_acceptor<instruction_record, clone_collective_group_instruction_record> {
	collective_group_id original_collective_group_id;
	collective_group_id new_collective_group_id;

	explicit clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr);
};

struct alloc_instruction_record : matchbox::implement_acceptor<instruction_record, alloc_instruction_record> {
	enum class alloc_origin {
		buffer,
		gather,
	};

	detail::allocation_id allocation_id;
	size_t size_bytes;
	size_t alignment_bytes;
	alloc_origin origin;
	std::optional<buffer_allocation_record> buffer_allocation;
	std::optional<size_t> num_chunks;

	alloc_instruction_record(
	    const alloc_instruction& ainstr, alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation, std::optional<size_t> num_chunks);
};

struct free_instruction_record : matchbox::implement_acceptor<instruction_record, free_instruction_record> {
	detail::allocation_id allocation_id;
	size_t size;
	std::optional<buffer_allocation_record> buffer_allocation;

	free_instruction_record(const free_instruction& finstr, size_t size, std::optional<buffer_allocation_record> buffer_allocation);
};

struct init_buffer_instruction_record : matchbox::implement_acceptor<instruction_record, init_buffer_instruction_record> {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::allocation_id host_allocation_id;
	size_t size_bytes;

	explicit init_buffer_instruction_record(const init_buffer_instruction& ibinstr, std::string buffer_name);
};

struct export_instruction_record : matchbox::implement_acceptor<instruction_record, export_instruction_record> {
	buffer_id buffer_id;
	std::string buffer_name;
	celerity::id<3> offset_in_buffer;
	allocation_id host_allocation_id;
	int dimensions;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	range<3> copy_range;
	size_t element_size;

	explicit export_instruction_record(
	    const export_instruction& einstr, detail::buffer_id buffer_id, std::string buffer_name, const celerity::id<3>& offset_in_buffer);
};

struct copy_instruction_record : matchbox::implement_acceptor<instruction_record, copy_instruction_record> {
	enum class copy_origin {
		linearize,
		resize,
		coherence,
		gather,
	};

	allocation_id source_allocation_id;
	allocation_id dest_allocation_id;
	int dimensions;
	range<3> source_range;
	range<3> dest_range;
	celerity::id<3> offset_in_source;
	celerity::id<3> offset_in_dest;
	range<3> copy_range;
	size_t element_size;
	copy_origin origin;
	buffer_id buffer_id;
	std::string buffer_name;
	detail::box<3> box; // NOMERGE what does this do?

	copy_instruction_record(
	    const copy_instruction& cinstr, copy_origin origin, detail::buffer_id buffer_id, std::string buffer_name, const detail::box<3>& box);
};

struct buffer_memory_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
};

struct buffer_reduction_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::reduction_id reduction_id;
};

struct buffer_access_allocation_record : buffer_access_allocation, buffer_memory_record {
	buffer_access_allocation_record(const buffer_access_allocation& aa, buffer_memory_record mr)
	    : buffer_access_allocation(aa), buffer_memory_record(std::move(mr)) {}
};

struct buffer_reduction_allocation_record : buffer_access_allocation, buffer_reduction_record {
	buffer_reduction_allocation_record(const buffer_access_allocation& aa, buffer_reduction_record mrr)
	    : buffer_access_allocation(aa), buffer_reduction_record(std::move(mrr)) {}
};

struct device_kernel_instruction_record : matchbox::implement_acceptor<instruction_record, device_kernel_instruction_record> {
	detail::device_id device_id;
	subrange<3> execution_range;
	std::vector<buffer_access_allocation_record> access_map;
	std::vector<buffer_reduction_allocation_record> reduction_map;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string debug_name;

	device_kernel_instruction_record(const device_kernel_instruction& dkinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const std::vector<buffer_memory_record>& buffer_memory_allocation_map, const std::vector<buffer_reduction_record>& buffer_memory_reduction_map);
};

struct host_task_instruction_record : matchbox::implement_acceptor<instruction_record, host_task_instruction_record> {
	detail::collective_group_id collective_group_id;
	subrange<3> execution_range;
	std::vector<buffer_access_allocation_record> access_map;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string debug_name;

	host_task_instruction_record(const host_task_instruction& htinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const std::vector<buffer_memory_record>& buffer_memory_allocation_map);
};

struct send_instruction_record : matchbox::implement_acceptor<instruction_record, send_instruction_record> {
	node_id dest_node_id;
	int tag;
	allocation_id source_allocation_id;
	range<3> source_allocation_range;
	celerity::id<3> offset_in_source_allocation;
	range<3> send_range;
	size_t element_size;
	command_id push_cid;
	detail::transfer_id transfer_id;
	std::string buffer_name;
	celerity::id<3> offset_in_buffer;

	send_instruction_record(
	    const send_instruction& sinstr, command_id push_cid, const detail::transfer_id& trid, std::string buffer_name, const celerity::id<3>& offset_in_buffer);
};

struct receive_instruction_record_impl {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	region<3> requested_region;
	allocation_id dest_allocation_id;
	box<3> allocated_box;
	size_t element_size;

	receive_instruction_record_impl(const receive_instruction_impl& rinstr, std::string buffer_name);
};

struct receive_instruction_record : matchbox::implement_acceptor<instruction_record, receive_instruction_record>, receive_instruction_record_impl {
	receive_instruction_record(const receive_instruction& rinstr, std::string buffer_name);
};

struct split_receive_instruction_record : matchbox::implement_acceptor<instruction_record, split_receive_instruction_record>, receive_instruction_record_impl {
	split_receive_instruction_record(const split_receive_instruction& srinstr, std::string buffer_name);
};

struct await_receive_instruction_record : matchbox::implement_acceptor<instruction_record, await_receive_instruction_record> {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	region<3> received_region;

	await_receive_instruction_record(const await_receive_instruction& arinstr, std::string buffer_name);
};

struct gather_receive_instruction_record : matchbox::implement_acceptor<instruction_record, gather_receive_instruction_record> {
	transfer_id transfer_id;
	std::string buffer_name;
	allocation_id allocation_id;
	size_t node_chunk_size;
	box<3> gather_box;
	size_t num_nodes;

	gather_receive_instruction_record(const gather_receive_instruction& grinstr, std::string buffer_name, const box<3>& gather_box, size_t num_nodes);
};

struct fill_identity_instruction_record : matchbox::implement_acceptor<instruction_record, fill_identity_instruction_record> {
	reduction_id reduction_id;
	allocation_id allocation_id;
	size_t num_values;

	fill_identity_instruction_record(const fill_identity_instruction& fiinstr);
};

struct reduce_instruction_record : matchbox::implement_acceptor<instruction_record, reduce_instruction_record> {
	enum class reduction_scope {
		global,
		local,
	};

	reduction_id reduction_id;
	allocation_id source_allocation_id;
	size_t num_source_values;
	allocation_id dest_allocation_id;
	std::optional<command_id> reduction_command_id;
	buffer_id buffer_id;
	std::string buffer_name;
	box<3> box;
	reduction_scope scope;

	reduce_instruction_record(const reduce_instruction& rinstr, std::optional<detail::command_id> reduction_cid, detail::buffer_id bid, std::string buffer_name,
	    const detail::box<3>& box, reduction_scope scope);
};

struct fence_instruction_record : matchbox::implement_acceptor<instruction_record, fence_instruction_record> {
	struct buffer_variant {
		buffer_id bid;
		std::string name;
		box<3> box;
	};
	struct host_object_variant {
		host_object_id hoid;
	};

	task_id tid;
	command_id cid;
	std::variant<buffer_variant, host_object_variant> variant;

	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, buffer_id bid, std::string buffer_name, const box<3>& box);
	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, host_object_id hoid);
};

struct destroy_host_object_instruction_record : matchbox::implement_acceptor<instruction_record, destroy_host_object_instruction_record> {
	detail::host_object_id host_object_id;

	explicit destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr);
};

struct horizon_instruction_record : matchbox::implement_acceptor<instruction_record, horizon_instruction_record> {
	task_id horizon_task_id;
	command_id horizon_command_id;
	std::vector<detail::reduction_id> completed_reductions;

	horizon_instruction_record(const horizon_instruction& hinstr, command_id horizon_cid);
};

struct epoch_instruction_record : matchbox::implement_acceptor<instruction_record, epoch_instruction_record> {
	task_id epoch_task_id;
	command_id epoch_command_id;
	epoch_action epoch_action;
	std::vector<detail::reduction_id> completed_reductions;

	epoch_instruction_record(const epoch_instruction& einstr, command_id epoch_cid);
};

class instruction_recorder {
  public:
	using outbound_pilots = std::vector<outbound_pilot>;

	void record_await_push_command_id(const transfer_id& trid, const command_id cid);
	void record_instruction(std::unique_ptr<instruction_record> record) { m_recorded_instructions.push_back(std::move(record)); }
	void record_outbound_pilot(const outbound_pilot& pilot) { m_recorded_pilots.push_back(pilot); }
	void record_dependencies(const instruction& instr);

	template <typename InstructionRecord>
	friend instruction_recorder& operator<<(instruction_recorder& recorder, InstructionRecord record) {
		recorder.record_instruction(std::make_unique<InstructionRecord>(std::move(record)));
		return recorder;
	}

	friend instruction_recorder& operator<<(instruction_recorder& recorder, const outbound_pilot& pilot) {
		recorder.record_outbound_pilot(pilot);
		return recorder;
	}

	const std::vector<std::unique_ptr<instruction_record>>& get_instructions() const { return m_recorded_instructions; }

	const instruction_record& get_instruction(const instruction_id iid) const {
		const auto it = std::find_if(
		    m_recorded_instructions.begin(), m_recorded_instructions.end(), [=](const std::unique_ptr<instruction_record>& instr) { return instr->id == iid; });
		assert(it != m_recorded_instructions.end());
		return **it;
	}

	const outbound_pilots& get_outbound_pilots() const { return m_recorded_pilots; }
	command_id get_await_push_command_id(const transfer_id& trid) const;

  private:
	std::vector<std::unique_ptr<instruction_record>> m_recorded_instructions;
	std::vector<outbound_pilot> m_recorded_pilots;
	std::unordered_map<transfer_id, command_id> m_await_push_cids;
	std::unordered_map<buffer_id, std::string> m_buffer_debug_names;
};

} // namespace celerity::detail
