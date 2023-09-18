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
	box<3> box;
};

using instruction_dependency_list = std::vector<dependency_record<instruction_id>>;

struct instruction_record_base {
	instruction_id id;
	instruction_dependency_list dependencies;

	explicit instruction_record_base(const instruction& instr);
};

struct alloc_instruction_record : instruction_record_base {
	enum class alloc_origin {
		buffer,
		send,
	};

	detail::allocation_id allocation_id;
	detail::memory_id memory_id;
	size_t size;
	size_t alignment;
	alloc_origin origin;
	std::optional<buffer_allocation_record> buffer_allocation;

	alloc_instruction_record(const alloc_instruction& ainstr, const alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation);
};

struct free_instruction_record : instruction_record_base {
	detail::allocation_id allocation_id;
	detail::memory_id memory_id;
	size_t size;
	size_t alignment;
	std::optional<buffer_allocation_record> buffer_allocation;

	free_instruction_record(const free_instruction& finstr, const detail::memory_id mid, const size_t size, const size_t alignment,
	    const std::optional<buffer_allocation_record>& buffer_allocation);
};

struct init_buffer_instruction_record : instruction_record_base {
	detail::buffer_id buffer_id;
	detail::allocation_id host_allocation_id;
	size_t size;

	explicit init_buffer_instruction_record(const init_buffer_instruction& ibinstr);
};

struct export_instruction_record : instruction_record_base {
	allocation_id host_allocation_id;
	int dimensions;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	range<3> copy_range;
	size_t element_size;

	explicit export_instruction_record(const export_instruction& einstr);
};

struct copy_instruction_record : instruction_record_base {
	enum class copy_origin {
		linearize,
		resize,
		coherence,
	};

	memory_id source_memory;
	allocation_id source_allocation;
	memory_id dest_memory;
	allocation_id dest_allocation;
	int dimensions;
	range<3> source_range;
	range<3> dest_range;
	celerity::id<3> offset_in_source;
	celerity::id<3> offset_in_dest;
	range<3> copy_range;
	size_t element_size;
	copy_origin origin;
	buffer_id buffer;
	detail::box<3> box;

	copy_instruction_record(const copy_instruction& cinstr, const copy_origin origin, const buffer_id buffer, const detail::box<3>& box);
};

struct kernel_instruction_record : instruction_record_base {
	execution_target target;
	std::optional<detail::device_id> device_id;
	subrange<3> execution_range;
	access_allocation_map allocation_map;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string kernel_debug_name;
	std::vector<buffer_allocation_record> allocation_buffer_map;

	kernel_instruction_record(const kernel_instruction& kinstr, const task_id cg_tid, const command_id execution_cid, const std::string& kernel_debug_name,
	    std::vector<buffer_allocation_record> allocation_buffer_map);
};

struct send_instruction_record : instruction_record_base {
	transfer_id transfer_id;
	node_id dest_node_id;
	int tag;
	allocation_id source_allocation_id;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	range<3> send_range;
	size_t element_size;
	command_id push_cid;
	buffer_id buffer;
	celerity::id<3> offset_in_buffer;

	send_instruction_record(const send_instruction& sinstr, const command_id push_cid, const buffer_id buffer, const celerity::id<3>& offset_in_buffer);
};

struct recv_instruction_record : instruction_record_base {
	buffer_id buffer_id;
	transfer_id transfer_id;
	memory_id dest_memory_id;
	allocation_id dest_allocation_id;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	celerity::id<3> offset_in_buffer;
	range<3> recv_range;
	size_t element_size;

	recv_instruction_record(const recv_instruction& rinstr);
};

struct fence_instruction_record : instruction_record_base {
	struct buffer_variant {
		buffer_id bid;
		box<3> box;
	};
	struct host_object_variant {
		host_object_id hoid;
	};

	task_id tid;
	command_id cid;
	std::variant<buffer_variant, host_object_variant> variant;

	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, buffer_id bid, const box<3>& box);
	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, host_object_id hoid);
};

struct destroy_host_object_instruction_record : instruction_record_base {
	detail::host_object_id host_object_id;

	explicit destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr);
};

struct horizon_instruction_record : instruction_record_base {
	task_id horizon_task_id;
	command_id horizon_command_id;

	horizon_instruction_record(const horizon_instruction& hinstr, const command_id horizon_cid);
};

struct epoch_instruction_record : instruction_record_base {
	task_id epoch_task_id;
	command_id epoch_command_id;
	epoch_action epoch_action;

	epoch_instruction_record(const epoch_instruction& einstr, const command_id epoch_cid);
};

using instruction_record = std::variant<alloc_instruction_record, free_instruction_record, init_buffer_instruction_record, export_instruction_record,
    copy_instruction_record, kernel_instruction_record, send_instruction_record, recv_instruction_record, fence_instruction_record,
    destroy_host_object_instruction_record, horizon_instruction_record, epoch_instruction_record>;

struct pilot_message_record : pilot_message {
	node_id receiver;

	pilot_message_record(const pilot_message& pilot, const node_id receiver);
};

class instruction_recorder {
  public:
	using instruction_records = std::vector<instruction_record>;
	using pilot_messages = std::vector<pilot_message_record>;

	void record_await_push_command_id(const transfer_id trid, const command_id cid) { m_await_push_cids.emplace(trid, cid); }
	void record_buffer_debug_name(const buffer_id bid, const std::string& debug_name) { m_buffer_debug_names.emplace(bid, debug_name); }
	void record_instruction(instruction_record record) { m_recorded_instructions.push_back(std::move(record)); }
	void record_pilot_message(const pilot_message_record& pilot) { m_recorded_pilots.push_back(pilot); }
	void record_dependencies(const instruction& instr);

	friend instruction_recorder& operator<<(instruction_recorder& recorder, instruction_record record) {
		recorder.record_instruction(std::move(record));
		return recorder;
	}

	friend instruction_recorder& operator<<(instruction_recorder& recorder, const pilot_message_record& pilot) {
		recorder.record_pilot_message(pilot);
		return recorder;
	}

	const instruction_records& get_instructions() const { return m_recorded_instructions; }
	const pilot_messages& get_pilot_messages() const { return m_recorded_pilots; }
	command_id get_await_push_command_id(const transfer_id trid) const;
	const std::string& get_buffer_debug_name(const buffer_id bid) const;

  private:
	instruction_records m_recorded_instructions;
	pilot_messages m_recorded_pilots;
	std::unordered_map<transfer_id, command_id> m_await_push_cids;
	std::unordered_map<buffer_id, std::string> m_buffer_debug_names;
};

} // namespace celerity::detail
