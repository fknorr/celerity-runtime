#pragma once

#include "command.h"
#include "instruction_graph.h"
#include "task.h"

namespace celerity::detail {

class buffer_manager;
class task_manager;

// General recording

struct access_record {
	const buffer_id bid;
	const std::string buffer_name;
	const access_mode mode;
	const region<3> req;
};
using access_list = std::vector<access_record>;

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
	task_record(const task& from, const buffer_manager* buff_mngr);

	const task_id tid;
	const std::string debug_name;
	const collective_group_id cgid;
	const task_type type;
	const task_geometry geometry;
	const reduction_list reductions;
	const access_list accesses;
	const side_effect_map side_effect_map;
	const task_dependency_list dependencies;
};

class task_recorder {
  public:
	using task_records = std::vector<task_record>;

	task_recorder(const buffer_manager* buff_mngr = nullptr) : m_buff_mngr(buff_mngr) {}

	void record_task(const task& tsk);

	const task_records& get_tasks() const { return m_recorded_tasks; }

	const task_record& get_task(const task_id tid) const {
		const auto it = std::find_if(m_recorded_tasks.begin(), m_recorded_tasks.end(), [tid](const task_record& rec) { return rec.tid == tid; });
		assert(it != m_recorded_tasks.end());
		return *it;
	}

  private:
	task_records m_recorded_tasks;
	const buffer_manager* m_buff_mngr;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	const command_id cid;
	const command_type type;

	const std::optional<epoch_action> epoch_action;
	const std::optional<subrange<3>> execution_range;
	const std::optional<reduction_id> reduction_id;
	const std::optional<buffer_id> buffer_id;
	const std::string buffer_name;
	const std::optional<node_id> target;
	const std::optional<region<3>> await_region;
	const std::optional<subrange<3>> push_range;
	const std::optional<transfer_id> transfer_id;
	const std::optional<task_id> task_id;
	const std::optional<task_geometry> task_geometry;
	const bool is_reduction_initializer;
	const std::optional<access_list> accesses;
	const std::optional<reduction_list> reductions;
	const std::optional<side_effect_map> side_effects;
	const command_dependency_list dependencies;
	const std::string task_name;
	const std::optional<task_type> task_type;
	const std::optional<collective_group_id> collective_group_id;

	command_record(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_mngr);
};

class command_recorder {
  public:
	using command_records = std::vector<command_record>;

	command_recorder(const task_manager* task_mngr, const buffer_manager* buff_mngr = nullptr) : m_task_mngr(task_mngr), m_buff_mngr(buff_mngr) {}

	void record_command(const abstract_command& com);

	const command_records& get_commands() const { return m_recorded_commands; }

	const command_record& get_command(const command_id cid) const {
		const auto it = std::find_if(m_recorded_commands.begin(), m_recorded_commands.end(), [cid](const command_record& rec) { return rec.cid == cid; });
		assert(it != m_recorded_commands.end());
		return *it;
	}

  private:
	command_records m_recorded_commands;
	const task_manager* m_task_mngr;
	const buffer_manager* m_buff_mngr;
};

struct buffer_allocation_record {
	detail::buffer_id buffer_id;
	std::string debug_name;
	box<3> box;
};

using instruction_dependency_list = std::vector<dependency_record<instruction_id>>;

struct instruction_record_base {
	instruction_id id;
	instruction_backend backend;
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
	    std::optional<buffer_allocation_record> buffer_allocation);
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
	std::string buffer_debug_name;
	detail::box<3> box;

	copy_instruction_record(
	    const copy_instruction& cinstr, const copy_origin origin, const buffer_id buffer, std::string buffer_debug_name, const detail::box<3>& box);
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

	kernel_instruction_record(const kernel_instruction& kinstr, const task_id cg_tid, const command_id execution_cid, std::string kernel_debug_name,
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
	std::string buffer_debug_name;
	box<3> box;

	send_instruction_record(
	    const send_instruction& sinstr, const command_id push_cid, const buffer_id buffer, std::string buffer_debug_name, const detail::box<3> box);
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
	command_id await_push_cid;
	detail::buffer_id buffer;
	std::string buffer_debug_name;

	recv_instruction_record(const recv_instruction& rinstr, const command_id await_push_cid, const detail::buffer_id buffer, std::string buffer_debug_name);
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

using instruction_record = std::variant<alloc_instruction_record, free_instruction_record, copy_instruction_record, kernel_instruction_record,
    send_instruction_record, recv_instruction_record, horizon_instruction_record, epoch_instruction_record>;

class instruction_recorder {
  public:
	using instruction_records = std::vector<instruction_record>;
	using pilot_messages = std::vector<pilot_message>;

	void record_instruction(instruction_record record) { m_recorded_instructions.push_back(std::move(record)); }
	void record_pilot_message(const pilot_message& pilot) { m_recorded_pilots.push_back(pilot); }

	friend instruction_recorder& operator<<(instruction_recorder& recorder, instruction_record record) {
		recorder.record_instruction(std::move(record));
		return recorder;
	}

	friend instruction_recorder& operator<<(instruction_recorder& recorder, const pilot_message& pilot) {
		recorder.record_pilot_message(pilot);
		return recorder;
	}

	const instruction_records& get_instructions() const { return m_recorded_instructions; }
	const pilot_messages& get_pilot_messages() const { return m_recorded_pilots; }

  private:
	instruction_records m_recorded_instructions;
	pilot_messages m_recorded_pilots;
};

} // namespace celerity::detail
