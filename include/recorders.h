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

struct clone_collective_group_instruction_record : instruction_record_base {
	collective_group_id origin_collective_group_id;
	collective_group_id new_collective_group_id;

	explicit clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr);
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

	alloc_instruction_record(const alloc_instruction& ainstr, alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation);
};

struct free_instruction_record : instruction_record_base {
	detail::memory_id memory_id;
	detail::allocation_id allocation_id;
	size_t size;
	std::optional<buffer_allocation_record> buffer_allocation;

	free_instruction_record(const free_instruction& finstr, size_t size, const std::optional<buffer_allocation_record>& buffer_allocation);
};

struct init_buffer_instruction_record : instruction_record_base {
	detail::buffer_id buffer_id;
	detail::allocation_id host_allocation_id;
	size_t size;

	explicit init_buffer_instruction_record(const init_buffer_instruction& ibinstr);
};

struct export_instruction_record : instruction_record_base {
	buffer_id buffer;
	celerity::id<3> offset_in_buffer;
	allocation_id host_allocation_id;
	int dimensions;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	range<3> copy_range;
	size_t element_size;

	explicit export_instruction_record(const export_instruction& einstr, buffer_id buffer, const celerity::id<3>& offset_in_buffer);
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

	copy_instruction_record(const copy_instruction& cinstr, copy_origin origin, buffer_id bid, const detail::box<3>& box);
};

struct buffer_memory_allocation_record {
	detail::buffer_id buffer_id;
	detail::memory_id memory_id;
	box<3> box;
};

struct buffer_memory_reduction_record : buffer_memory_allocation_record {
	detail::reduction_id reduction_id;
};

struct buffer_allocation_access_record : access_allocation, buffer_memory_allocation_record {
	constexpr buffer_allocation_access_record(const access_allocation& aa, const buffer_memory_allocation_record& bmar)
	    : access_allocation(aa), buffer_memory_allocation_record(bmar) {}
};

struct buffer_allocation_reduction_record : access_allocation, buffer_memory_reduction_record {
	constexpr buffer_allocation_reduction_record(const access_allocation& aa, const buffer_memory_reduction_record& bmrr)
	    : access_allocation(aa), buffer_memory_reduction_record(bmrr) {}
};

struct launch_instruction_record : instruction_record_base {
	execution_target target;
	std::optional<detail::device_id> device_id;
	std::optional<detail::collective_group_id> collective_group_id;
	subrange<3> execution_range;
	std::vector<buffer_allocation_access_record> access_map;
	std::vector<buffer_allocation_reduction_record> reduction_map;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string kernel_debug_name;

	launch_instruction_record(const launch_instruction& linstr, task_id cg_tid, command_id execution_cid, const std::string& kernel_debug_name,
	    const std::vector<buffer_memory_allocation_record>& buffer_memory_allocation_map,
	    const std::vector<buffer_memory_reduction_record>& buffer_memory_reduction_map);
};

struct send_instruction_record : instruction_record_base {
	node_id dest_node_id;
	int tag;
	memory_id source_memory_id;
	allocation_id source_allocation_id;
	range<3> allocation_range;
	celerity::id<3> offset_in_allocation;
	range<3> send_range;
	size_t element_size;
	command_id push_cid;
	detail::transfer_id transfer_id;
	celerity::id<3> offset_in_buffer;

	send_instruction_record(const send_instruction& sinstr, command_id push_cid, const detail::transfer_id& trid, const celerity::id<3>& offset_in_buffer);
};

struct begin_receive_instruction_record : instruction_record_base {
	detail::transfer_id transfer_id;
	region<3> requested_region;
	std::vector<begin_receive_instruction::destination> destinations;
	size_t element_size;

	begin_receive_instruction_record(const begin_receive_instruction& brinstr);
};

struct await_receive_instruction_record : instruction_record_base {
	detail::transfer_id transfer_id;
	region<3> received_region;
	memory_id dest_memory_id;
	allocation_id dest_allocation_id;
	box<3> dest_allocation_box;

	await_receive_instruction_record(
	    const await_receive_instruction& arinstr, memory_id dest_memory_id, allocation_id dest_allocation_id, const box<3>& dest_allocation_box);
};

struct end_receive_instruction_record : instruction_record_base {
	detail::transfer_id transfer_id;

	end_receive_instruction_record(const end_receive_instruction& erinstr);
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

	horizon_instruction_record(const horizon_instruction& hinstr, command_id horizon_cid);
};

struct epoch_instruction_record : instruction_record_base {
	task_id epoch_task_id;
	command_id epoch_command_id;
	epoch_action epoch_action;

	epoch_instruction_record(const epoch_instruction& einstr, command_id epoch_cid);
};

using instruction_record = std::variant<clone_collective_group_instruction_record, alloc_instruction_record, free_instruction_record,
    init_buffer_instruction_record, export_instruction_record, copy_instruction_record, launch_instruction_record, send_instruction_record,
    begin_receive_instruction_record, await_receive_instruction_record, end_receive_instruction_record, fence_instruction_record,
    destroy_host_object_instruction_record, horizon_instruction_record, epoch_instruction_record>;

class instruction_recorder {
  public:
	using instruction_records = std::vector<instruction_record>;
	using outbound_pilots = std::vector<outbound_pilot>;

	void record_await_push_command_id(const transfer_id& trid, const command_id cid) { m_await_push_cids.emplace(trid, cid); }
	void record_buffer_debug_name(const buffer_id bid, const std::string& debug_name) { m_buffer_debug_names.emplace(bid, debug_name); }
	void record_instruction(instruction_record record) { m_recorded_instructions.push_back(std::move(record)); }
	void record_pilot_message(const outbound_pilot& pilot) { m_recorded_pilots.push_back(pilot); }
	void record_dependencies(const instruction& instr);

	friend instruction_recorder& operator<<(instruction_recorder& recorder, instruction_record record) {
		recorder.record_instruction(std::move(record));
		return recorder;
	}

	friend instruction_recorder& operator<<(instruction_recorder& recorder, const outbound_pilot& pilot) {
		recorder.record_pilot_message(pilot);
		return recorder;
	}

	const instruction_records& get_instructions() const { return m_recorded_instructions; }
	const outbound_pilots& get_outbound_pilots() const { return m_recorded_pilots; }
	command_id get_await_push_command_id(const transfer_id& trid) const;
	const std::string& get_buffer_debug_name(buffer_id bid) const;

  private:
	instruction_records m_recorded_instructions;
	outbound_pilots m_recorded_pilots;
	std::unordered_map<transfer_id, command_id> m_await_push_cids;
	std::unordered_map<buffer_id, std::string> m_buffer_debug_names;
};

} // namespace celerity::detail
