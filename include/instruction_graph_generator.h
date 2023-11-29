#pragma once

#include "ranges.h"
#include "types.h"

#include <bitset>
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
	static constexpr size_t max_num_memories = 64;
	using memory_mask = std::bitset<max_num_memories>;

	struct device_info {
		memory_id native_memory;
	};
	struct memory_info {
		memory_mask copy_peers;
	};
	struct system_info {
		std::vector<device_info> devices;  ///< indexed by device_id
		std::vector<memory_info> memories; ///< indexed by memory_id
	};

	/// Implement this as the owner of instruction_graph_generator to receive callbacks on generated instructions and pilot messages.
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		/// Called whenever new instructions have been generated and inserted into the instruction graph.
		///
		/// The vector of instructions is in topological order of dependencies, and so is the concatenation of all vectors that are passed through this
		/// function. Topological order here means that sequential execution in that order would fulfill all instruction dependencies.
		///
		/// The instruction graph generator guarantees that these pointers are stable and the pointed-to instructions are not modified _except_ for
		/// intrusive_graph_node::get_dependents(), so other threads are safe to read from these pointers as long as they do not examine that member.
		virtual void flush_instructions(std::vector<const instruction*> instrs) = 0;

		/// Called whenever new pilot messages have been generated that must be transmitted to peer nodes before they can accept data transmitted through
		/// `send_instruction`s originating from the local node.
		virtual void flush_outbound_pilots(std::vector<outbound_pilot> pilots) = 0;
	};

	struct policy_set {
		error_policy unsafe_oversubscription_error = error_policy::throw_exception;
		error_policy uninitialized_read_error = error_policy::throw_exception;
		error_policy overlapping_write_error = error_policy::throw_exception;
	};

	/// Instruction graph generation requires information about the target system. `num_nodes` and `local_nid` effect the generation of communication
	/// instructions and reductions, and `system` is used to determine work assignment, memory allocation and data migration between memories.
	///
	/// Generated instructions are inserted into (and subsequently owned by) the provided `idag`, and if `dlg` is provided, it is notified about any newly
	/// created instructions and outbound pilots.
	///
	/// If and only if a `recorder` is present, the generator will collect debug information about each generated instruction and pass it to the recorder. Set
	/// this to `nullptr` in production code in order to avoid a performance penalty.
	///
	/// Specify a non-default `policy` to influence what user-errors are detected at runtime and how they are reported. The default is is to throw exceptions
	/// which catches errors early in tests, but users of this class will want to ease these settings. Any policy set to a value other than
	/// `error_policy::ignore` will have a performance penalty.
	explicit instruction_graph_generator(const task_manager& tm, size_t num_nodes, node_id local_nid, system_info system, instruction_graph& idag,
	    delegate* dlg = nullptr, instruction_recorder* recorder = nullptr, const policy_set& policy = default_policy_set());

	instruction_graph_generator(const instruction_graph_generator&) = delete;
	instruction_graph_generator(instruction_graph_generator&&) = default;
	instruction_graph_generator& operator=(const instruction_graph_generator&) = delete;
	instruction_graph_generator& operator=(instruction_graph_generator&&) = default;

	~instruction_graph_generator();

	/// Begin tracking local data distribution and dependencies on the buffer with id `bid`. Allocations are made lazily on first access.
	///
	/// Passing `user_allocation_id != null_allocation_id` means that the buffer is considered coherent in user memory and data will be lazily copied from that
	/// allocation when read from host tasks or device kernels.
	void create_buffer(buffer_id bid, int dims, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_allocation_id);

	/// Changing an existing buffer's debug name causes all future instructions to refer to that buffer by the new name (if a recorder is present).
	void set_buffer_debug_name(buffer_id bid, const std::string& name);

	void destroy_buffer(buffer_id bid);

	/// Begin tracking dependencies on the host object with id `hoid`. If `owns_instance` is true, a `destroy_host_object_instruction` will be emitted when
	/// `destroy_host_object` is subsequently called.
	void create_host_object(host_object_id hoid, bool owns_instance);

	/// End tracking the host object with id `hoid`. Emits `destroy_host_object_instruction` if `create_host_object` was called with `owns_instance == true`.
	void destroy_host_object(host_object_id hoid);

	/// Compiles a command-graph node into a set of instructions, which are inserted into the shared instruction graph, and updates tracking structures.
	void compile(const abstract_command& cmd);

  private:
	/// Default-constructs a `policy_set` - this must be a function because we can't use the implicit default constructor of `policy_set`, which has member
	/// initializers, within its surrounding class (Clang diagnostic).
	constexpr static policy_set default_policy_set() { return {}; }

	class impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
