#pragma once

#include "ranges.h"
#include "types.h"

#include <vector>

namespace celerity::detail {

class abstract_command;
class instruction;
class instruction_graph;
class instruction_recorder;
struct outbound_pilot;
class task_manager;

class instruction_graph_generator {
  public:
	struct device_info {
		memory_id native_memory;
		// - which backends / features are supported?
	};

	struct policy_set {
		error_policy unsafe_oversubscription_error = error_policy::throw_exception;
		error_policy uninitialized_read_error = error_policy::throw_exception;
		error_policy overlapping_write_error = error_policy::throw_exception;
	};

	// TODO should take unordered_map<device_id, device_info> (runtime is responsible for device id allocation, not IGGEN)
	explicit instruction_graph_generator(const task_manager& tm, size_t num_nodes, node_id local_node_id, std::vector<device_info> devices,
	    instruction_graph& idag, instruction_recorder* recorder, const policy_set& policy = default_policy_set());
	instruction_graph_generator(const instruction_graph_generator&) = delete;
	instruction_graph_generator(instruction_graph_generator&&) = default;
	instruction_graph_generator& operator=(const instruction_graph_generator&) = delete;
	instruction_graph_generator& operator=(instruction_graph_generator&&) = default;
	~instruction_graph_generator();

	void set_uninitialized_read_policy(const error_policy policy);
	void set_overlapping_write_policy(const error_policy policy);

	void create_buffer(buffer_id bid, int dims, const range<3>& range, size_t elem_size, size_t elem_align, bool host_initialized);

	void set_buffer_debug_name(buffer_id bid, const std::string& name);

	void destroy_buffer(buffer_id bid);

	void create_host_object(host_object_id hoid, bool owns_instance);

	void destroy_host_object(host_object_id hoid);

	// Resulting instructions are in topological order of dependencies (i.e. sequential execution would fulfill all internal dependencies)
	std::pair<std::vector<const instruction*>, std::vector<outbound_pilot>> compile(const abstract_command& cmd);

  private:
	// default-constructs a policy_set - this must be a function because we can't use the implicit default constructor of policy_set, which has member
	// initializers, within its surrounding class (Clang)
	constexpr static policy_set default_policy_set() { return {}; }

	class impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
