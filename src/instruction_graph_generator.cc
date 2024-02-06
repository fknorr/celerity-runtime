#include "instruction_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "recorders.h"
#include "region_map.h"
#include "split.h"
#include "task.h"
#include "task_manager.h"
#include "tracy.h"
#include "types.h"

#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace celerity::detail::instruction_graph_generator_detail {

/// MPI, our only "production" communicator implementation, limits the extent of send/receive operations to INT_MAX elements in each dimension. Since we do not
/// require this value anywhere else we define it in instruction_graph_generator.cc, even though it does not logically originate here.
constexpr size_t communicator_max_coordinate = INT_MAX;

/// Helper for split_into_communicator_compatible_boxes().
void split_into_communicator_compatible_boxes_recurse(
    box_vector<3>& comm_boxes, const box<3>& full_box, id<3> min, id<3> max, const int slice_dim, const int dim) {
	assert(dim <= slice_dim);
	const auto& full_box_min = full_box.get_min();
	const auto& full_box_max = full_box.get_max();

	if(dim < slice_dim) {
		for(min[dim] = full_box_min[dim]; min[dim] < full_box_max[dim]; ++min[dim]) {
			max[dim] = min[dim] + 1;
			split_into_communicator_compatible_boxes_recurse(comm_boxes, full_box, min, max, slice_dim, dim + 1);
		}
	} else {
		for(min[dim] = full_box_min[dim]; min[dim] < full_box_max[dim]; min[dim] += communicator_max_coordinate) {
			max[dim] = std::min(full_box_max[dim], min[dim] + communicator_max_coordinate);
			comm_boxes.emplace_back(min, max);
		}
	}
}

/// The (MPI) communicator can only exchange boxes sized up to `communicator_max_coordinate` in each dimension. Larger boxes must be split - at the implied
/// transfer sizes, this will not have any measurable performance impact.
box_vector<3> split_into_communicator_compatible_boxes(const range<3>& buffer_range, const box<3>& full_box) {
	assert(box(subrange<3>(zeros, buffer_range)).covers(full_box));

	int slice_dim = 0;
	for(int d = 1; d < 3; ++d) {
		if(buffer_range[d] > communicator_max_coordinate) { slice_dim = d; }
	}

	box_vector<3> comm_boxes;
	split_into_communicator_compatible_boxes_recurse(comm_boxes, full_box, full_box.get_min(), full_box.get_max(), slice_dim, 0);
	return comm_boxes;
}

/// Determines whether two boxes are either overlapping or touching on edges (not corners). This means on 2-connectivity for 1d boxes, 4-connectivity for 2d
/// boxes and 6-connectivity for 3d boxes. For 0-dimensional boxes, always returns true.
template <int Dims>
bool boxes_edge_connected(const box<Dims>& box1, const box<Dims>& box2) {
	if(box1.empty() || box2.empty()) return false;

	const auto min = id_max(box1.get_min(), box2.get_min());
	const auto max = id_min(box1.get_max(), box2.get_max());
	bool touching = false;
	for(int d = 0; d < Dims; ++d) {
		if(min[d] > max[d]) return false; // fully disconnected, even across corners
		if(min[d] == max[d]) {
			// when boxes are touching (but not intersecting) in more than one dimension, they can only be connected via corners
			if(touching) return false;
			touching = true;
		}
	}
	return true;
}

/// Subdivide a region into connected partitions (where connectivity is established by `boxes_edge_connected`) and return the bounding box of each partition.
/// Note that the returned boxes are not necessarily disjoint event through the partitions always are.
///
/// This logic is employed to find connected subregions that might be satisfied by a peer through a single `send_instruction` and thus requires a contiguous
/// backing allocation.
template <int Dims>
box_vector<Dims> connected_subregion_bounding_boxes(const region<Dims>& region) {
	auto boxes = region.get_boxes();
	auto begin = boxes.begin();
	auto end = boxes.end();
	box_vector<3> bounding_boxes;
	while(begin != end) {
		auto connected_end = std::next(begin);
		auto connected_bounding_box = *begin; // optimization: skip connectivity checks if bounding box is disconnected
		for(; connected_end != end; ++connected_end) {
			const auto next_connected = std::find_if(connected_end, end, [&](const auto& candidate) {
				return boxes_edge_connected(connected_bounding_box, candidate)
				       && std::any_of(begin, connected_end, [&](const auto& box) { return boxes_edge_connected(candidate, box); });
			});
			if(next_connected == end) break;
			connected_bounding_box = bounding_box(connected_bounding_box, *next_connected);
			std::swap(*next_connected, *connected_end);
		}
		bounding_boxes.push_back(connected_bounding_box);
		begin = connected_end;
	}
	return bounding_boxes;
}

/// Iteratively replaces each pair of overlapping boxes by their bounding box such that in the modified set of boxes, no two boxes overlap, but all original
/// input boxes are covered.
///
/// When a kernel or host task has multiple accessors into a single allocation, each must be backed by a contiguous allocation. This makes it necessary to
/// contiguously allocate the bounding box of all overlapping accesses.
template <int Dims>
void merge_overlapping_bounding_boxes(box_vector<Dims>& boxes) {
restart:
	for(auto first = boxes.begin(); first != boxes.end(); ++first) {
		const auto last = std::remove_if(std::next(first), boxes.end(), [&](const auto& box) {
			const auto overlapping = !box_intersection(*first, box).empty();
			if(overlapping) { *first = bounding_box(*first, box); }
			return overlapping;
		});
		if(last != boxes.end()) {
			boxes.erase(last, boxes.end());
			goto restart; // NOLINT(cppcoreguidelines-avoid-goto)
		}
	}
}

/// Returns whether an iterator range of instruction pointers is topologically sorted, i.e. sequential execution would satisfy all internal dependencies.
template <typename Iterator>
bool is_topologically_sorted(Iterator begin, Iterator end) {
	for(auto check = begin; check != end; ++check) {
		for(const auto dep : (*check)->get_dependencies()) {
			if(std::find_if(std::next(check), end, [dep](const auto& node) { return node->get_id() == dep; }) != end) return false;
		}
	}
	return true;
}

/// In a set of potentially regions, removes the overlap between any two regions {A, B} by replacing them with {A ∖ B, A ∩ B, B ∖ A}.
///
/// This is used when generating copy and receive-instructions such that every data item is only copied or received once, but the following device kernels /
/// host tasks have their dependencies satisfied as soon as their subset of input data is available.
void symmetrically_split_overlapping_regions(std::vector<region<3>>& regions) {
restart:
	for(size_t i = 0; i < regions.size(); ++i) {
		for(size_t j = i + 1; j < regions.size(); ++j) {
			auto intersection = region_intersection(regions[i], regions[j]);
			if(!intersection.empty()) {
				regions[i] = region_difference(regions[i], intersection);
				regions[j] = region_difference(regions[j], intersection);
				regions.push_back(std::move(intersection));
				// if intersections above are actually subsets, we will end up with empty regions
				regions.erase(std::remove_if(regions.begin(), regions.end(), std::mem_fn(&region<3>::empty)), regions.end());
				goto restart; // NOLINT(cppcoreguidelines-avoid-goto)
			}
		}
	}
}

constexpr auto max_num_memories = instruction_graph_generator::max_num_memories;
using memory_mask = instruction_graph_generator::memory_mask;

memory_id next_location(const memory_mask& location, memory_id first) {
	for(size_t i = 0; i < max_num_memories; ++i) {
		const memory_id mem = (first + i) % max_num_memories;
		if(location[mem]) { return mem; }
	}
	utils::panic("data is requested to be read, but not located in any memory");
}

/// `allocation_id`s are "namespaced" to their memory ID, so we maintain the next `raw_allocation_id` for each memory separately.
struct memory_state {
	raw_allocation_id next_raw_aid = 1; // 0 is reserved for null_allocation_id
};

/// Per-allocation state for a single buffer. This is where we track last-writer instructions and access fronts.
struct buffer_allocation_state {
	struct access_front {
		gch::small_vector<instruction*> instructions; // ordered by id to allow equality comparison
		enum { allocate, read, write } mode = allocate;

		friend bool operator==(const access_front& lhs, const access_front& rhs) { return lhs.instructions == rhs.instructions && lhs.mode == rhs.mode; }
		friend bool operator!=(const access_front& lhs, const access_front& rhs) { return !(lhs == rhs); }
	};

	allocation_id aid;
	detail::box<3> box;                                ///< in virtual-buffer coordinates
	region_map<access_front> last_writers;             ///< in virtual-buffer coordinates
	region_map<access_front> last_concurrent_accesses; ///< in virtual-buffer coordinates

	explicit buffer_allocation_state(const allocation_id aid, alloc_instruction* const ainstr /* optional: null for user allocations */,
	    const detail::box<3>& allocated_box, const range<3>& buffer_range)
	    : aid(aid), box(allocated_box), last_writers(allocated_box), last_concurrent_accesses(allocated_box) {
		if(ainstr != nullptr) {
			last_writers.update_box(allocated_box, access_front{{ainstr}, access_front::allocate});
			last_concurrent_accesses.update_box(allocated_box, access_front{{ainstr}, access_front::allocate});
		}
	}

	template <typename BoxOrRegion>
	void commit_read(const BoxOrRegion& box_or_region, instruction* const instr) {
		if(box_or_region.empty()) return;
		for(auto& [box, front] : last_concurrent_accesses.get_region_values(box_or_region)) {
			if(front.mode == access_front::read) {
				// we call record_read as soon as the writing instructions is generated, so inserting at the end keeps the vector sorted
				assert(front.instructions.empty() || front.instructions.back()->get_id() < instr->get_id());
				front.instructions.push_back(instr);
			} else {
				front = {{instr}, access_front::read};
			}
			assert(std::is_sorted(front.instructions.begin(), front.instructions.end(), instruction_id_less()));
			last_concurrent_accesses.update_region(box, front);
		}
	}

	void commit_write(const detail::box<3>& box, instruction* const instr) {
		if(box.empty()) return;
		last_writers.update_box(box, access_front{{instr}, access_front::write});
		last_concurrent_accesses.update_box(box, access_front{{instr}, access_front::write});
	}

	void commit_write(const region<3>& region, instruction* const instr) {
		if(region.empty()) return;
		last_writers.update_region(region, access_front{{instr}, access_front::write});
		last_concurrent_accesses.update_region(region, access_front{{instr}, access_front::write});
	}

	void begin_combined_write(const detail::box<3>& box) {
		if(box.empty()) return;
		last_writers.update_box(box, access_front{{}, access_front::write});
		last_concurrent_accesses.update_box(box, access_front{{}, access_front::write});
	}

	void begin_combined_write(const region<3>& region) {
		if(region.empty()) return;
		last_writers.update_region(region, access_front{{}, access_front::write});
		last_concurrent_accesses.update_region(region, access_front{{}, access_front::write});
	}

	// TODO (all commit / read / write fns): naming?
	template <typename BoxOrRegion>
	void commit_combined_write(const BoxOrRegion& box_or_region, instruction* const instr) {
		if(box_or_region.empty()) return;
		for(auto& [box, front] : last_writers.get_region_values(box_or_region)) {
			assert(front.mode == access_front::write); // must call begin_overlapping_write first
			// we call commit() in order of generated instructions, so push_back is a sorted insert
			assert(front.instructions.empty() || front.instructions.back()->get_id() < instr->get_id());
			front.instructions.push_back(instr);
			assert(std::is_sorted(front.instructions.begin(), front.instructions.end(), instruction_id_less()));
			last_writers.update_region(box, front);
			last_concurrent_accesses.update_region(box, front);
		}
	}

	void apply_epoch(instruction* const epoch) {
		const auto apply_epoch_to_front = [epoch](const access_front& old_front) {
			const auto first_retained = std::upper_bound(old_front.instructions.begin(), old_front.instructions.end(), epoch, instruction_id_less());
			const auto last_retained = old_front.instructions.end();

			// only include the new epoch in the access front if it in fact subsumes another instruction
			if(first_retained == old_front.instructions.begin()) return old_front;

			access_front new_front;
			new_front.mode = old_front.mode;
			new_front.instructions.resize(1 + static_cast<size_t>(std::distance(first_retained, last_retained)));
			new_front.instructions.front() = epoch;
			std::copy(first_retained, last_retained, std::next(new_front.instructions.begin()));
			assert(std::is_sorted(new_front.instructions.begin(), new_front.instructions.end(), instruction_id_less()));
			return new_front;
		};
		last_writers.apply_to_values(apply_epoch_to_front);
		last_concurrent_accesses.apply_to_values(apply_epoch_to_front);
	}
};

/// Per-memory state for a single buffer. Dependencies and last writers are tracked on the contained allocations.
struct buffer_memory_state {
	// TODO bound the number of allocations per buffer in order to avoid runaway tracking overhead (similar to horizons)
	// TODO evaluate if it ever makes sense to use a region_map here, or if we're better off expecting very few allocations and sticking to a vector here
	std::vector<buffer_allocation_state> allocations; // disjoint

	const buffer_allocation_state& get_allocation(const allocation_id aid) const {
		const auto it = std::find_if(allocations.begin(), allocations.end(), [=](const buffer_allocation_state& a) { return a.aid == aid; });
		assert(it != allocations.end());
		return *it;
	}

	buffer_allocation_state& get_allocation(const allocation_id aid) { return const_cast<buffer_allocation_state&>(std::as_const(*this).get_allocation(aid)); }

	const buffer_allocation_state* find_contiguous_allocation(const box<3>& box) const {
		const auto it = std::find_if(allocations.begin(), allocations.end(), [&](const buffer_allocation_state& a) { return a.box.covers(box); });
		return it != allocations.end() ? &*it : nullptr;
	}

	buffer_allocation_state* find_contiguous_allocation(const box<3>& box) {
		return const_cast<buffer_allocation_state*>(std::as_const(*this).find_contiguous_allocation(box));
	}

	bool is_allocated_contiguously(const box<3>& box) const { return find_contiguous_allocation(box) != nullptr; }

	void apply_epoch(instruction* const epoch) {
		for(auto& alloc : allocations) {
			alloc.apply_epoch(epoch);
		}
	}
};

/// State for a single buffer.
struct buffer_state {
	/// Tracking structure for an await-push that already has a begin_receive_instruction, but not yet an end_receive_instruction.
	struct region_receive {
		task_id consumer_tid;
		region<3> received_region;
		box_vector<3> required_contiguous_allocations;

		region_receive(const task_id consumer_tid, region<3> received_region, box_vector<3> required_contiguous_allocations)
		    : consumer_tid(consumer_tid), received_region(std::move(received_region)),
		      required_contiguous_allocations(std::move(required_contiguous_allocations)) {}
	};

	struct gather_receive {
		task_id consumer_tid;
		reduction_id rid;
		box<3> gather_box;

		gather_receive(const task_id consumer_tid, const reduction_id rid, const box<3> gather_box)
		    : consumer_tid(consumer_tid), rid(rid), gather_box(gather_box) {}
	};

	std::string debug_name;
	celerity::range<3> range;
	size_t elem_size;
	size_t elem_align;
	dense_map<memory_id, buffer_memory_state> memories;
	region_map<instruction_graph_generator::memory_mask> up_to_date_memories; // TODO rename for vs original_write_memories?
	region_map<instruction*> original_writers;                                // TODO explain how and why this duplicates per_allocation_data::last_writers
	region_map<memory_id> original_write_memories;                            // only meaningful if newest_data_location[box] is non-empty

	// We store pending receives (await push regions) in a vector instead of a region map since we must process their entire regions en-bloc rather than on
	// a per-element basis.
	std::vector<region_receive> pending_receives;
	std::vector<gather_receive> pending_gathers;

	explicit buffer_state(const celerity::range<3>& range, const size_t elem_size, const size_t elem_align, const size_t n_memories)
	    : range(range), elem_size(elem_size), elem_align(elem_align), memories(n_memories), up_to_date_memories(range), original_writers(range),
	      original_write_memories(range) {}

	void commit_original_write(const region<3>& region, instruction* const instr, const memory_id mid) {
		original_writers.update_region(region, instr);
		original_write_memories.update_region(region, mid);
		up_to_date_memories.update_region(region, memory_mask().set(mid));
	}

	void apply_epoch(instruction* const epoch) {
		for(auto& memory : memories) {
			memory.apply_epoch(epoch);
		}
		original_writers.apply_to_values(
		    [epoch](instruction* const instr) -> instruction* { return instr != nullptr && instr->get_id() < epoch->get_id() ? epoch : instr; });

		// This is an opportune point to verify that all await-pushes are fully consumed eventually. On epoch application,
		// original_writers[*].await_receives potentially points to instructions before the new epoch, but when compiling a horizon or epoch command, all
		// previous await-pushes should have been consumed by the task command they were generated for.
		assert(pending_receives.empty());
		assert(pending_gathers.empty());
	}
};

struct host_object_state {
	bool owns_instance;
	instruction* last_side_effect = nullptr;

	explicit host_object_state(const bool owns_instance, instruction* const last_epoch) : owns_instance(owns_instance), last_side_effect(last_epoch) {}

	void apply_epoch(instruction* const epoch) {
		if(last_side_effect != nullptr && last_side_effect->get_id() < epoch->get_id()) { last_side_effect = epoch; }
	}
};

struct collective_group_state {
	instruction* last_host_task = nullptr;

	void apply_epoch(instruction* const epoch) {
		if(last_host_task != nullptr && last_host_task->get_id() < epoch->get_id()) { last_host_task = epoch; }
	}
};

/// We submit the set of instructions and pilots generated for a call to compile() en-bloc to relieve contention on the executor queue lock. To collect all
/// instructions that are generated in the call stack without polluting internal state, we pass a `batch&` output parameter to any function that
/// transitively generates instructions or pilots.
struct batch { // NOLINT(cppcoreguidelines-special-member-functions) (do not complain about the asserting destructor)
	std::vector<const instruction*> generated_instructions;
	std::vector<outbound_pilot> generated_pilots;
	int base_priority = 0;

#ifndef NDEBUG
	~batch() {
		if(std::uncaught_exceptions() == 0) { assert(generated_instructions.empty() && generated_pilots.empty() && "unflushed batch detected"); }
	}
#endif
};

struct localized_chunk {
	detail::memory_id memory_id = host_memory_id;
	box<3> execution_range;
};

// We assign instruction priorities heuristically by deciding on a batch::base_priority at the beginning of a compile_* function and offsetting it by the
// instruction-type specific values defined here.
// clang-format off
template <typename Instruction> constexpr int instruction_type_priority = 0; // higher means more urgent
template <> constexpr int instruction_type_priority<free_instruction> = -1; // only free when forced to - nothing except an epoch or horizon will depend on this
template <> constexpr int instruction_type_priority<alloc_instruction> = 1; // allocations have very high latency, so we postpone them as much as possible
template <> constexpr int instruction_type_priority<copy_instruction> = 2;
template <> constexpr int instruction_type_priority<await_receive_instruction> = 2;
template <> constexpr int instruction_type_priority<split_receive_instruction> = 2;
template <> constexpr int instruction_type_priority<receive_instruction> = 2;
template <> constexpr int instruction_type_priority<send_instruction> = 2;
template <> constexpr int instruction_type_priority<fence_instruction> = 3;
template <> constexpr int instruction_type_priority<host_task_instruction> = 4; // we expect kernels to have the highest latency of any instruction
template <> constexpr int instruction_type_priority<device_kernel_instruction> = 4; 
template <> constexpr int instruction_type_priority<epoch_instruction> = 5; // epochs and horizons are low-latency and stop the task buffers from reaching capacity
template <> constexpr int instruction_type_priority<horizon_instruction> = 5;
// clang-format on

class impl {
  public:
	impl(const task_manager& tm, size_t num_nods, node_id local_nid, instruction_graph_generator::system_info system, instruction_graph& idag,
	    instruction_graph_generator::delegate* dlg, instruction_recorder* recorder, const instruction_graph_generator::policy_set& policy);

	void create_buffer(buffer_id bid, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid = null_allocation_id);
	void set_buffer_debug_name(buffer_id bid, const std::string& name);
	void destroy_buffer(buffer_id bid);
	void create_host_object(host_object_id hoid, bool owns_instance);
	void destroy_host_object(host_object_id hoid);
	void compile(const abstract_command& cmd);

  private:
	using batch = instruction_graph_generator_detail::batch;
	using buffer_allocation_state = instruction_graph_generator_detail::buffer_allocation_state;
	using access_front = instruction_graph_generator_detail::buffer_allocation_state::access_front;
	using localized_chunk = instruction_graph_generator_detail::localized_chunk;
	using gather_receive = instruction_graph_generator_detail::buffer_state::gather_receive;
	using region_receive = instruction_graph_generator_detail::buffer_state::region_receive;

	inline static const box<3> scalar_reduction_box{zeros, ones};

	// construction parameters (immutable)
	instruction_graph* m_idag;
	const task_manager* m_tm; // TODO commands should reference tasks by pointer, not id - then we wouldn't need this member.
	size_t m_num_nodes;
	node_id m_local_nid;
	instruction_graph_generator::system_info m_system;
	instruction_graph_generator::delegate* m_delegate;
	instruction_recorder* m_recorder;
	instruction_graph_generator::policy_set m_policy;

	instruction_id m_next_instruction_id = 0;
	message_id m_next_message_id = 0;

	instruction* m_last_horizon = nullptr;
	instruction* m_last_epoch = nullptr;

	std::unordered_set<instruction_id> m_execution_front;

	dense_map<memory_id, instruction_graph_generator_detail::memory_state> m_memories; // indexed by memory_id
	std::unordered_map<buffer_id, instruction_graph_generator_detail::buffer_state> m_buffers;
	std::unordered_map<host_object_id, instruction_graph_generator_detail::host_object_state> m_host_objects;
	std::unordered_map<collective_group_id, instruction_graph_generator_detail::collective_group_state> m_collective_groups;

	std::vector<allocation_id> m_unreferenced_user_allocations; // from completed buffer - collected by the next horizon / epoch instruction

	allocation_id new_allocation_id(const memory_id mid);

	template <typename Instruction, typename... CtorParams>
	Instruction* create(batch& batch, CtorParams&&... ctor_args);

	message_id create_outbound_pilot(batch& batch, node_id target, const transfer_id& trid, const box<3>& box);

	void add_dependency(instruction* const from, instruction* const to, const instruction_dependency_origin record_origin);

	template <typename BoxOrRegion>
	void add_dependencies_on_last_writers(instruction* const accessing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region,
	    const instruction_dependency_origin record_origin_for_initialized_data);

	template <typename BoxOrRegion>
	void read_from_allocation(instruction* const reading_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region);

	template <typename BoxOrRegion>
	void add_dependencies_on_access_front(instruction* const accessing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region,
	    const instruction_dependency_origin record_origin_for_read_write_front);

	template <typename BoxOrRegion>
	void write_to_allocation(instruction* const writing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region);

	void apply_epoch(instruction* const epoch);

	void collapse_execution_front_to(instruction* const horizon_or_epoch);

	// Re-allocation of one buffer on one memory never interacts with other buffers or other memories backing the same buffer, this function can be called
	// in any order of allocation requirements without generating additional dependencies.
	void allocate_contiguously(batch& batch, buffer_id bid, memory_id mid, box_vector<3>&& required_contiguous_boxes);

	void commit_pending_region_receive(batch& batch, buffer_id bid, const region_receive& receives, const dense_map<memory_id, std::vector<region<3>>>& reads);

	// To avoid multi-hop copies, all read requirements for one buffer must be satisfied on all memories simultaneously. We deliberately allow multiple,
	// potentially-overlapping regions per memory to avoid aggregated copies introducing synchronization points between otherwise independent instructions.
	void locally_satisfy_read_requirements(batch& batch, buffer_id bid, memory_id dest_mid, const std::vector<region<3>>& reads);

	void satisfy_buffer_requirements_for_regular_access(
	    batch& batch, buffer_id bid, const task& tsk, const subrange<3>& local_sr, const std::vector<localized_chunk>& local_chunks);

	void satisfy_buffer_requirements_as_reduction_output(batch& batch, buffer_id bid, reduction_id rid, const std::vector<localized_chunk>& local_chunks);

	void satisfy_buffer_requirements(batch& batch, buffer_id bid, const task& tsk, const subrange<3>& local_sr, bool is_reduction_initializer,
	    const std::vector<localized_chunk>& local_chunks);

	void compile_execution_command(batch& batch, const execution_command& ecmd);
	void compile_push_command(batch& batch, const push_command& pcmd);
	void defer_await_push_command(const await_push_command& apcmd);
	void compile_reduction_command(batch& batch, const reduction_command& rcmd);
	void compile_fence_command(batch& batch, const fence_command& fcmd);
	void compile_horizon_command(batch& batch, const horizon_command& hcmd);
	void compile_epoch_command(batch& batch, const epoch_command& ecmd);

	void flush_batch(batch&& batch);

	std::string print_buffer_debug_label(buffer_id bid) const;
};


impl::impl(const task_manager& tm, size_t num_nodes, node_id local_nid, instruction_graph_generator::system_info system, instruction_graph& idag,
    instruction_graph_generator::delegate* dlg, instruction_recorder* const recorder, const instruction_graph_generator::policy_set& policy)
    : m_idag(&idag), m_tm(&tm), m_num_nodes(num_nodes), m_local_nid(local_nid), m_system(std::move(system)), m_delegate(dlg), m_recorder(recorder),
      m_policy(policy), m_memories(m_system.memories.size()) //
{
#ifndef NDEBUG
	assert(m_system.memories.size() <= max_num_memories);
	assert(std::all_of(m_system.devices.begin(), m_system.devices.end(),
	    [&](const instruction_graph_generator::device_info& device) { return device.native_memory < m_system.memories.size(); }));
	for(memory_id mid_a = 0; mid_a < m_system.memories.size(); ++mid_a) {
		assert(m_system.memories[mid_a].copy_peers[mid_a]);
		for(memory_id mid_b = mid_a + 1; mid_b < m_system.memories.size(); ++mid_b) {
			assert(m_system.memories[mid_a].copy_peers[mid_b] == m_system.memories[mid_b].copy_peers[mid_a]
			       && "system_info::memories::copy_peers must be reflexive");
		}
	}
#endif

	batch epoch_batch;
	m_idag->begin_epoch(task_manager::initial_epoch_task);
	const auto initial_epoch = create<epoch_instruction>(epoch_batch, task_manager::initial_epoch_task, epoch_action::none, instruction_garbage{});
	if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*initial_epoch, command_id(0 /* or so we assume */)); }
	m_last_epoch = initial_epoch;
	m_collective_groups.emplace(root_collective_group_id, initial_epoch);
	flush_batch(std::move(epoch_batch));
}

void impl::create_buffer(const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, allocation_id user_aid) //
{
	const auto [iter, inserted] =
	    m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(range, elem_size, elem_align, m_system.memories.size()));
	assert(inserted);

	if(user_aid != null_allocation_id) {
		assert(user_aid.get_memory_id() == user_memory_id);
		const box entire_buffer = subrange({}, range);

		auto& buffer = iter->second;
		auto& memory = buffer.memories[user_memory_id];
		auto& allocation = memory.allocations.emplace_back(user_aid, nullptr /* alloc_instruction */, entire_buffer, buffer.range);

		allocation.commit_write(entire_buffer, m_last_epoch);
		buffer.commit_original_write(entire_buffer, m_last_epoch, user_memory_id);
	}
}

void impl::set_buffer_debug_name(const buffer_id bid, const std::string& name) { m_buffers.at(bid).debug_name = name; }

void impl::destroy_buffer(const buffer_id bid) {
	const auto iter = m_buffers.find(bid);
	assert(iter != m_buffers.end());
	auto& buffer = iter->second;

	batch free_batch;
	for(memory_id mid = 0; mid < buffer.memories.size(); ++mid) {
		if(mid == user_memory_id) continue;

		auto& memory = buffer.memories[mid];
		for(auto& allocation : memory.allocations) {
			const auto free_instr = create<free_instruction>(free_batch, allocation.aid);
			if(m_recorder != nullptr) {
				*m_recorder << free_instruction_record(
				    *free_instr, allocation.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.debug_name, allocation.box});
			}
			add_dependencies_on_access_front(free_instr, allocation, allocation.box, instruction_dependency_origin::allocation_lifetime);
			// no need to modify the access front - we're removing the buffer altogether!
		}
	}
	flush_batch(std::move(free_batch));

	m_buffers.erase(iter);
}

void impl::create_host_object(const host_object_id hoid, const bool owns_instance) {
	assert(m_host_objects.count(hoid) == 0);
	m_host_objects.emplace(std::piecewise_construct, std::tuple(hoid), std::tuple(owns_instance, m_last_epoch));
}

void impl::destroy_host_object(const host_object_id hoid) {
	const auto iter = m_host_objects.find(hoid);
	assert(iter != m_host_objects.end());

	auto& obj = iter->second;
	if(obj.owns_instance) {
		batch destroy_batch;
		const auto destroy_instr = create<destroy_host_object_instruction>(destroy_batch, hoid);
		if(m_recorder != nullptr) { *m_recorder << destroy_host_object_instruction_record(*destroy_instr); }
		add_dependency(destroy_instr, obj.last_side_effect, instruction_dependency_origin::side_effect);
		flush_batch(std::move(destroy_batch));
	}

	m_host_objects.erase(iter);
}

allocation_id impl::new_allocation_id(const memory_id mid) {
	assert(mid < m_memories.size());
	assert(mid != user_memory_id && "user allocation ids are not managed by the instruction graph generator");
	return allocation_id(mid, m_memories[mid].next_raw_aid++);
}

template <typename Instruction, typename... CtorParams>
Instruction* impl::create(batch& batch, CtorParams&&... ctor_args) {
	const auto id = m_next_instruction_id++;
	const auto priority = batch.base_priority + instruction_graph_generator_detail::instruction_type_priority<Instruction>;
	auto instr = std::make_unique<Instruction>(id, priority, std::forward<CtorParams>(ctor_args)...);
	const auto ptr = instr.get();
	m_idag->push_instruction(std::move(instr));
	m_execution_front.insert(id);
	batch.generated_instructions.push_back(ptr);
	return ptr;
}

message_id impl::create_outbound_pilot(batch& current_batch, const node_id target, const transfer_id& trid, const box<3>& box) {
	const message_id msgid = m_next_message_id++;
	const outbound_pilot pilot{target, pilot_message{msgid, trid, box}};
	current_batch.generated_pilots.push_back(pilot);
	if(m_recorder != nullptr) { *m_recorder << pilot; }
	return msgid;
}

void impl::add_dependency(instruction* const from, instruction* const to, const instruction_dependency_origin record_origin) {
	from->add_dependency(to->get_id());
	if(m_recorder != nullptr) { m_recorder->record_dependency(from->get_id(), to->get_id(), record_origin); }
	m_execution_front.erase(to->get_id());
}

template <typename BoxOrRegion>
void impl::add_dependencies_on_last_writers(instruction* const accessing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region,
    const instruction_dependency_origin record_origin_for_initialized_data) {
	for(const auto& [box, front] : allocation.last_writers.get_region_values(region)) {
		for(const auto writer : front.instructions) {
			add_dependency(accessing_instruction, writer,
			    front.mode == access_front::allocate ? instruction_dependency_origin::allocation_lifetime : record_origin_for_initialized_data);
		}
	}
}

template <typename BoxOrRegion>
void impl::read_from_allocation(instruction* const reading_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region) {
	add_dependencies_on_last_writers(reading_instruction, allocation, region, instruction_dependency_origin::read_from_allocation);
	allocation.commit_read(region, reading_instruction);
}

template <typename BoxOrRegion>
void impl::add_dependencies_on_access_front(instruction* const accessing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region,
    const instruction_dependency_origin record_origin_for_read_write_front) {
	for(const auto& [box, front] : allocation.last_concurrent_accesses.get_region_values(region)) {
		for(const auto dep_instr : front.instructions) {
			add_dependency(accessing_instruction, dep_instr,
			    front.mode == access_front::allocate ? instruction_dependency_origin::allocation_lifetime : record_origin_for_read_write_front);
		}
	}
}

template <typename BoxOrRegion>
void impl::write_to_allocation(instruction* const writing_instruction, buffer_allocation_state& allocation, const BoxOrRegion& region) {
	add_dependencies_on_access_front(writing_instruction, allocation, region, instruction_dependency_origin::write_to_allocation);
	allocation.commit_write(region, writing_instruction);
}

void impl::apply_epoch(instruction* const epoch) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::apply_epoch", SlateBlue, "apply epoch");

	for(auto& [_, buffer] : m_buffers) {
		buffer.apply_epoch(epoch);
	}
	for(auto& [_, host_object] : m_host_objects) {
		host_object.apply_epoch(epoch);
	}
	for(auto& [_, collective_group] : m_collective_groups) {
		collective_group.apply_epoch(epoch);
	}
	m_last_epoch = epoch;
}

void impl::collapse_execution_front_to(instruction* const horizon_or_epoch) {
	for(const auto iid : m_execution_front) {
		if(iid == horizon_or_epoch->get_id()) continue;
		// we can't use instruction_graph_generator::add_depencency since it modifies the m_execution_front which we're iterating over here
		horizon_or_epoch->add_dependency(iid);
		if(m_recorder != nullptr) { m_recorder->record_dependency(horizon_or_epoch->get_id(), iid, instruction_dependency_origin::execution_front); }
	}
	m_execution_front.clear();
	m_execution_front.insert(horizon_or_epoch->get_id());
}

void impl::allocate_contiguously(batch& current_batch, const buffer_id bid, const memory_id mid, box_vector<3>&& required_contiguous_boxes) //
{
	if(required_contiguous_boxes.empty()) return;

	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::allocate", DodgerBlue, "allocate");

	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories[mid];

	// We currently only ever *grow* the allocation of buffers on each memory, which means that we must merge (re-size) existing allocations that overlap with
	// but do not fully contain one of the required contiguous boxes. *Overlapping* here strictly means having a non-empty intersection; two boxes that merely
	// "touch" can continue to co-exist
	auto&& contiguous_boxes_after_reallocation = std::move(required_contiguous_boxes);
	const auto first_empty =
	    std::remove_if(contiguous_boxes_after_reallocation.begin(), contiguous_boxes_after_reallocation.end(), [](const box<3>& box) { return box.empty(); });
	contiguous_boxes_after_reallocation.erase(first_empty, contiguous_boxes_after_reallocation.end());
	for(auto& alloc : memory.allocations) {
		contiguous_boxes_after_reallocation.push_back(alloc.box);
	}
	instruction_graph_generator_detail::merge_overlapping_bounding_boxes(contiguous_boxes_after_reallocation);

	// Allocations that are now fully contained in (but not equal to) one of the newly contiguous bounding boxes will be freed at the end of the reallocation
	// step, because we currently disallow overlapping allocations for simplicity. These will function as sources for resize-copies below.
	const auto first_old_allocations_to_be_freed =
	    std::partition(memory.allocations.begin(), memory.allocations.end(), [&](const buffer_allocation_state& allocation) {
		    return std::find(contiguous_boxes_after_reallocation.begin(), contiguous_boxes_after_reallocation.end(), allocation.box)
		           != contiguous_boxes_after_reallocation.end();
	    });
	const auto existing_allocations_end = memory.allocations.end();

	// Derive the set of new boxes to allocate by removing all existing boxes from the set of contiguous boxes.
	auto&& newly_allocated_boxes = std::move(contiguous_boxes_after_reallocation);
	const auto last_new_allocation = std::remove_if(newly_allocated_boxes.begin(), newly_allocated_boxes.end(),
	    [&](auto& box) { return std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == box; }); });
	newly_allocated_boxes.erase(last_new_allocation, newly_allocated_boxes.end());
	if(newly_allocated_boxes.empty()) return;

	// Opportunistically merge adjacent boxes to keep the number of allocations and the tracking overhead low. This will not introduce artificial
	// synchronization points because resize-copies are still rooted on the original last-writers.
	merge_adjacent_boxes(newly_allocated_boxes);

	// We collect new allocations in a vector *separate* from memory.allocations to not invalidate iterators (and avoid accidentally resize-copying from them).
	std::vector<buffer_allocation_state> new_allocations;
	new_allocations.reserve(newly_allocated_boxes.size());

	// Allocate new memory and initialize it via resize-copies, if applicable.
	for(const auto& new_box : newly_allocated_boxes) {
		const auto aid = new_allocation_id(mid);
		const auto alloc_instr = create<alloc_instruction>(current_batch, aid, new_box.get_area() * buffer.elem_size, buffer.elem_align);
		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(
			    *alloc_instr, alloc_instruction_record::alloc_origin::buffer, buffer_allocation_record{bid, buffer.debug_name, new_box}, std::nullopt);
		}

		add_dependency(alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

		// New allocations are kept in a separate vector until reallocation is complete
		auto& dest_allocation = new_allocations.emplace_back(aid, alloc_instr, new_box, buffer.range);

		// Since allocations don't overlap, we only need to consider those to be freed as source for resize-copies
		for(auto source_it = first_old_allocations_to_be_freed; source_it != existing_allocations_end; ++source_it) {
			auto& source_allocation = *source_it;

			// Only copy those boxes to the new allocation that are still up-to-date in the old allocation. The caller of allocate_contiguously should remove
			// any region from up_to_date_memories that they intend to discard / overwrite immediately to avoid dead resize copies.
			// TODO investigate a garbage-collection heuristic that omits these copies if there are other up-to-date memories and we do not expect the region to
			// be read again on this memory.
			const auto full_copy_box = box_intersection(dest_allocation.box, source_allocation.box);
			if(full_copy_box.empty()) continue; // not every freed allocation necessarily intersects with every new allocation

			box_vector<3> live_copy_boxes;
			for(const auto& [copy_box, location] : buffer.up_to_date_memories.get_region_values(full_copy_box)) {
				if(location.test(mid)) { live_copy_boxes.push_back(copy_box); }
			}
			if(live_copy_boxes.empty()) continue; // even if allocations intersect, the entire intersection might be overwritten

			region<3> live_copy_region(std::move(live_copy_boxes));
			const auto copy_instr = create<copy_instruction>(
			    current_batch, source_allocation.aid, dest_allocation.aid, source_allocation.box, dest_allocation.box, live_copy_region, buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::resize, bid, buffer.debug_name);
			}

			read_from_allocation(copy_instr, source_allocation, live_copy_region);
			write_to_allocation(copy_instr, dest_allocation, live_copy_region);
		}
	}

	// Free old allocations now that any resize-copies have been issued.
	// TODO consider keeping old allocations around until their box is written to (or at least until the end of the current instruction batch) in order to
	// resolve "buffer-locking" anti-dependencies
	for(auto it = first_old_allocations_to_be_freed; it != memory.allocations.end(); ++it) {
		auto& allocation = *it;

		const auto free_instr = create<free_instruction>(current_batch, allocation.aid);
		if(m_recorder != nullptr) {
			*m_recorder << free_instruction_record(
			    *free_instr, allocation.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.debug_name, allocation.box});
		}

		add_dependencies_on_access_front(free_instr, allocation, allocation.box, instruction_dependency_origin::allocation_lifetime);
	}

	// TODO garbage-collect allocations that are not up-to-date and not written to in this task?

	memory.allocations.erase(first_old_allocations_to_be_freed, memory.allocations.end());
	memory.allocations.insert(memory.allocations.end(), std::make_move_iterator(new_allocations.begin()), std::make_move_iterator(new_allocations.end()));
}

void impl::commit_pending_region_receive(
    batch& current_batch, const buffer_id bid, const region_receive& receive, const dense_map<memory_id, std::vector<region<3>>>& reads) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::commit_receive", MediumOrchid, "commit recv");

	const auto trid = transfer_id(receive.consumer_tid, bid, no_reduction_id);
	const auto mid = host_memory_id;

	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories[mid];

	std::vector<buffer_allocation_state*> allocations;
	for(const auto& min_contiguous_box : receive.required_contiguous_allocations) {
		const auto alloc = memory.find_contiguous_allocation(min_contiguous_box);
		assert(alloc != nullptr); // allocated explicitly in satisfy_buffer_requirements
		if(std::find(allocations.begin(), allocations.end(), alloc) == allocations.end()) { allocations.push_back(alloc); }
	}

	for(const auto alloc : allocations) {
		const auto alloc_recv_region = region_intersection(alloc->box, receive.received_region);
		std::vector<region<3>> independent_await_regions;
		for(const auto& independent_read_regions : reads) {
			for(const auto& read_region : independent_read_regions) {
				const auto await_region = region_intersection(read_region, alloc_recv_region);
				if(!await_region.empty()) { independent_await_regions.push_back(await_region); }
			}
		}
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(independent_await_regions);

		if(independent_await_regions.size() > 1) {
			const auto split_recv_instr = create<split_receive_instruction>(current_batch, trid, alloc_recv_region, alloc->aid, alloc->box, buffer.elem_size);
			if(m_recorder != nullptr) { *m_recorder << split_receive_instruction_record(*split_recv_instr, buffer.debug_name); }

			// We add dependencies to the begin_receive_instruction as if it were a writer, but update the last_writers only at the await_receive_instruction.
			// The actual write happens somewhere in-between these instructions as orchestrated by the receive_arbiter, and any other accesses need to ensure
			// that there are no pending transfers for the region they are trying to read or to access (TODO).
			add_dependencies_on_access_front(split_recv_instr, *alloc, alloc_recv_region, instruction_dependency_origin::write_to_allocation);

#ifndef NDEBUG
			region<3> full_await_region;
			for(const auto& await_region : independent_await_regions) {
				full_await_region = region_union(full_await_region, await_region);
			}
			assert(full_await_region == alloc_recv_region);
#endif

			for(const auto& await_region : independent_await_regions) {
				const auto await_instr = create<await_receive_instruction>(current_batch, trid, await_region);
				if(m_recorder != nullptr) { *m_recorder << await_receive_instruction_record(*await_instr, buffer.debug_name); }

				add_dependency(await_instr, split_recv_instr, instruction_dependency_origin::split_receive);

				alloc->commit_write(await_region, await_instr);
				buffer.original_writers.update_region(await_region, await_instr);
			}
		} else {
			assert(independent_await_regions.size() == 1 && independent_await_regions[0] == alloc_recv_region);

			const auto recv_instr = create<receive_instruction>(current_batch, trid, alloc_recv_region, alloc->aid, alloc->box, buffer.elem_size);
			if(m_recorder != nullptr) { *m_recorder << receive_instruction_record(*recv_instr, buffer.debug_name); }

			write_to_allocation(recv_instr, *alloc, alloc_recv_region);
			buffer.original_writers.update_region(alloc_recv_region, recv_instr);
		}
	}

	buffer.original_write_memories.update_region(receive.received_region, mid);
	buffer.up_to_date_memories.update_region(receive.received_region, memory_mask().set(mid));
}

void impl::locally_satisfy_read_requirements(
    batch& current_batch, const buffer_id bid, const memory_id dest_mid, const std::vector<region<3>>& concurrent_reads) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::local_coherence", Salmon, "local coherence");

	auto& buffer = m_buffers.at(bid);

	// Given the full region to be read, find all regions not up to date in `dest_mid`.
	std::vector<region<3>> unsatisfied_concurrent_reads;
	for(const auto& read_region : concurrent_reads) {
		box_vector<3> unsatisfied_boxes;
		for(const auto& [box, location] : buffer.up_to_date_memories.get_region_values(read_region)) {
			if(location.any() /* gracefully handle uninitialized read */ && !location.test(dest_mid)) { unsatisfied_boxes.push_back(box); }
		}
		if(!unsatisfied_boxes.empty()) { unsatisfied_concurrent_reads.emplace_back(std::move(unsatisfied_boxes)); }
	}
	if(unsatisfied_concurrent_reads.empty()) return;

	// Transform vectors of potentially-overlapping unsatisfied regions into disjoint regions - e.g. if there are two overlapping regions [first, second], split
	// them into three disjoint regions [first_only, overlap, second_only]. As soon as the coherence of either [first_only, overlap] or [overlap, second_only]
	// is is established, the consumer of [first] or [second] can start executing.
	instruction_graph_generator_detail::symmetrically_split_overlapping_regions(unsatisfied_concurrent_reads);

	// Collect the regions to be copied from each memory. As long as we don't touch `buffer.up_to_date_memories` we could also create copy_instructions directly
	// within this loop, but separating these two steps keeps code more readable at the expense of allocating one more vector.
	std::vector<std::tuple<memory_id, region<3>>> concurrent_copies_from_source;
	for(auto& unsatisfied_region : unsatisfied_concurrent_reads) {
		// Split the region on original writers to enable concurrency between the write of one region and a copy on another, already written region.
		std::unordered_map<instruction_id, box_vector<3>> unsatisfied_boxes_by_writer;
		for(const auto& [writer_box, original_writer] : buffer.original_writers.get_region_values(unsatisfied_region)) {
			assert(original_writer != nullptr);
			unsatisfied_boxes_by_writer[original_writer->get_id()].push_back(writer_box);
		}

		for(auto& [_, unsatisfied_boxes] : unsatisfied_boxes_by_writer) {
			const region unsatisfied_region(std::move(unsatisfied_boxes));
			// There can be multiple original-writer memories if the original writer has been subsumed by an epoch or a horizon.
			std::unordered_map<memory_id, box_vector<3>> copy_from_source;

			for(const auto& [copy_box, original_write_mid] : buffer.original_write_memories.get_region_values(unsatisfied_region)) {
				if(m_system.memories[original_write_mid].copy_peers.test(dest_mid)) {
					// Prefer copying from the original writer's memory to avoid introducing copy-chains between the instructions of multiple commands.
					copy_from_source[original_write_mid].push_back(copy_box);
				} else {
					// If this is not possible, we expect that an earlier call to locally_satisfy_read_requirements has made the host memory coherent, meaning
					// that we are in the second hop of a host-staged copy.
#ifndef NDEBUG
					assert(m_system.memories[dest_mid].copy_peers.test(host_memory_id));
					for(const auto& [_, location] : buffer.up_to_date_memories.get_region_values(copy_box)) {
						assert(location.test(host_memory_id));
					}
#endif
					copy_from_source[host_memory_id].push_back(copy_box);
				}
			}
			for(auto& [source_mid, copy_boxes] : copy_from_source) {
				assert(!copy_boxes.empty());
				concurrent_copies_from_source.emplace_back(source_mid, region(std::move(copy_boxes)));
			}
		}
	}

	// Create, between pairs of source and dest allocations, the copy instructions according to the previously collected regions.
	for(auto& [source_mid, copy_from_memory_region] : concurrent_copies_from_source) {
		assert(dest_mid != source_mid);
		for(auto& source_allocation : buffer.memories[source_mid].allocations) {
			const auto read_from_allocation_region = region_intersection(copy_from_memory_region, source_allocation.box);
			if(read_from_allocation_region.empty()) continue;

			for(auto& dest_allocation : buffer.memories[dest_mid].allocations) {
				const auto copy_between_allocations_region = region_intersection(read_from_allocation_region, dest_allocation.box);
				if(copy_between_allocations_region.empty()) continue;

				const auto copy_instr = create<copy_instruction>(current_batch, source_allocation.aid, dest_allocation.aid, source_allocation.box,
				    dest_allocation.box, copy_between_allocations_region, buffer.elem_size);
				if(m_recorder != nullptr) {
					*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::coherence, bid, buffer.debug_name);
				}

				read_from_allocation(copy_instr, source_allocation, copy_between_allocations_region);
				write_to_allocation(copy_instr, dest_allocation, copy_between_allocations_region);
			}
		}
	}

	// Update buffer.up_to_date_memories for the entire updated region at once.
	for(const auto& region : unsatisfied_concurrent_reads) {
		for(auto& [box, location] : buffer.up_to_date_memories.get_region_values(region)) {
			buffer.up_to_date_memories.update_region(box, memory_mask(location).set(dest_mid));
		}
	}
}

void impl::satisfy_buffer_requirements(batch& current_batch, const buffer_id bid, const task& tsk, const subrange<3>& local_sr,
    const bool local_node_is_reduction_initializer, const std::vector<localized_chunk>& concurrent_local_chunks) //
{
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::coherence", SandyBrown, "B{} coherence", bid);

	assert(!concurrent_local_chunks.empty());
	const auto& bam = tsk.get_buffer_access_map();

	// Invalidate any buffer region that will immediately be overwritten (and not also read) to avoid preserving it across buffer resizes (and to catch
	// read-write access conflicts, TODO)

	auto& buffer = m_buffers.at(bid);

	dense_map<memory_id, box_vector<3>> required_contiguous_allocations(m_memories.size());

	box_vector<3> accessed_boxes; // which elements have are accessed (to figure out applying receives)
	box_vector<3> consumed_boxes; // which elements are accessed with a consuming access (these need to be preserved across resizes)
	for(const auto mode : access::all_modes) {
		const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), local_sr, tsk.get_global_size());
		accessed_boxes.append(req.get_boxes());
		if(access::mode_traits::is_consumer(mode)) { consumed_boxes.append(req.get_boxes()); }
	}

	assert(std::count_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; }) <= 1
	       && "task defines multiple reductions on the same buffer");
	const auto reduction = std::find_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; });
	if(reduction != tsk.get_reductions().end()) {
		for(const auto& chunk : concurrent_local_chunks) {
			required_contiguous_allocations[chunk.memory_id].push_back(scalar_reduction_box);
		}
		const auto include_current_value = local_node_is_reduction_initializer && reduction->init_from_buffer;
		if(concurrent_local_chunks.size() > 1 || include_current_value) {
			// we insert a host-side reduce-instruction in the multi-chunk scenario; its result will end up in the host buffer allocation
			required_contiguous_allocations[host_memory_id].push_back(scalar_reduction_box);
		}
		accessed_boxes.push_back(scalar_reduction_box);
		if(include_current_value) {
			// scalar_reduction_box will be copied into the local-reduction gather buffer ahead of the kernel instruction
			consumed_boxes.push_back(scalar_reduction_box);
		}
	}

	const region accessed_region(std::move(accessed_boxes));
	const region consumed_region(std::move(consumed_boxes));

	// Boxes that are accessed but not consumed do not need to be preserved across resizes. This set operation is not equivalent to accumulating all
	// non-consumer mode accesses above, since a kernel can have both a read_only and a discard_write access for the same buffer element, and Celerity must
	// treat the overlap as-if it were a read_write access according to the SYCL spec.
	// We maintain a box_vector here because we also add all received boxes, as these are overwritten by a recv_instruction before being read from the kernel.
	box_vector<3> discarded_boxes = region_difference(accessed_region, consumed_region).into_boxes();

	// Collect all receives that we must apply before executing this command.

	const auto first_applied_receive = std::partition(buffer.pending_receives.begin(), buffer.pending_receives.end(),
	    [&](const region_receive& r) { return region_intersection(accessed_region, r.received_region).empty(); });
	for(auto it = first_applied_receive; it != buffer.pending_receives.end(); ++it) {
		// we (re) allocate before receiving, but there's no need to preserve previous data at the receive location
		discarded_boxes.append(it->received_region.get_boxes());
		// begin_receive_instruction needs contiguous allocations for the bounding boxes of potentially received fragments
		required_contiguous_allocations[host_memory_id].insert(
		    required_contiguous_allocations[host_memory_id].end(), it->required_contiguous_allocations.begin(), it->required_contiguous_allocations.end());
	}

	std::vector<region_receive> applied_receives;
	if(first_applied_receive != buffer.pending_receives.end()) {
		applied_receives.assign(first_applied_receive, buffer.pending_receives.end());
		buffer.pending_receives.erase(first_applied_receive, buffer.pending_receives.end());
	}

	if(reduction != tsk.get_reductions().end()) {
		assert(std::all_of(buffer.pending_receives.begin(), buffer.pending_receives.end(), [&](const region_receive& r) {
			return region_intersection(r.received_region, scalar_reduction_box).empty();
		}) && std::all_of(buffer.pending_gathers.begin(), buffer.pending_gathers.end(), [&](const gather_receive& r) {
			return box_intersection(r.gather_box, scalar_reduction_box).empty();
		}) && "buffer has an unprocessed await-push into a region that is going to be used as a reduction output");
	}

	const region discarded_region = region(std::move(discarded_boxes));

	// Detect and report uninitialized reads

	if(m_policy.uninitialized_read_error != error_policy::ignore) {
		box_vector<3> uninitialized_reads;
		const auto locally_required_region = region_difference(consumed_region, discarded_region);
		for(const auto& [box, location] : buffer.up_to_date_memories.get_region_values(locally_required_region)) {
			if(!location.any()) { uninitialized_reads.push_back(box); }
		}
		if(!uninitialized_reads.empty()) {
			// Observing an uninitialized read that is not visible in the TDAG means we have a bug.
			utils::report_error(m_policy.uninitialized_read_error,
			    "Instructions for {} are trying to read {} {}, which is neither found locally nor has been await-pushed before.", print_task_debug_label(tsk),
			    print_buffer_debug_label(bid), detail::region(std::move(uninitialized_reads)));
		}
	}

	// do not preserve any received or overwritten region across receives or buffer resizes later on
	buffer.up_to_date_memories.update_region(discarded_region, memory_mask());

	dense_map<memory_id, std::vector<region<3>>> local_chunk_reads(m_memories.size());
	for(const auto& chunk : concurrent_local_chunks) {
		required_contiguous_allocations[chunk.memory_id].append(
		    bam.get_required_contiguous_boxes(bid, tsk.get_dimensions(), chunk.execution_range.get_subrange(), tsk.get_global_size()));

		box_vector<3> chunk_read_boxes;
		for(const auto mode : access::consumer_modes) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), chunk.execution_range.get_subrange(), tsk.get_global_size());
			chunk_read_boxes.append(req.get_boxes());
		}
		if(!chunk_read_boxes.empty()) { local_chunk_reads[chunk.memory_id].push_back(region(std::move(chunk_read_boxes))); }
	}
	if(local_node_is_reduction_initializer && reduction != tsk.get_reductions().end() && reduction->init_from_buffer) {
		local_chunk_reads[host_memory_id].emplace_back(scalar_reduction_box);
	}

	// collect all required staging allocations
	for(memory_id read_mid = 0; read_mid < local_chunk_reads.size(); ++read_mid) {
		for(auto& read_region : local_chunk_reads[read_mid]) {
			box_vector<3> host_staged_boxes;
			for(const auto& [box, location] : buffer.up_to_date_memories.get_region_values(read_region)) {
				if(location.any() && !location.test(read_mid) && (m_system.memories[read_mid].copy_peers & location).none()) {
					assert(read_mid != host_memory_id);
					required_contiguous_allocations[host_memory_id].push_back(box);
					host_staged_boxes.push_back(box);
				}
			}
			if(!host_staged_boxes.empty()) { local_chunk_reads[host_memory_id].emplace_back(std::move(host_staged_boxes)); }
		}
	}

	for(memory_id mid = 0; mid < required_contiguous_allocations.size(); ++mid) {
		allocate_contiguously(current_batch, bid, mid, std::move(required_contiguous_allocations[mid]));
	}

	for(const auto& receive : applied_receives) {
		commit_pending_region_receive(current_batch, bid, receive, local_chunk_reads);
	}

	// Create the necessary coherence copy instructions to satisfy all remaining requirements locally. The iterations of this loop are independent with the
	// notable exception of host_memory_id in the presence of staging copies: There we rely on the fact that `host_memory_id < device_memory_id` to allow
	// coherence copies to device memory to create device -> host -> device copy chains.
	for(memory_id mid = 0; mid < local_chunk_reads.size(); ++mid) {
		locally_satisfy_read_requirements(current_batch, bid, mid, local_chunk_reads[mid]);
	}
}


void impl::compile_execution_command(batch& command_batch, const execution_command& ecmd) {
	const auto& tsk = *m_tm->get_task(ecmd.get_tid());
	const auto& bam = tsk.get_buffer_access_map();

	// 0) collectively generate any non-existing collective group

	if(const auto cgid = tsk.get_collective_group_id(); cgid != non_collective_group_id && m_collective_groups.count(cgid) == 0) {
		auto& root_cg = m_collective_groups.at(root_collective_group_id);
		const auto clone_cg_isntr = create<clone_collective_group_instruction>(command_batch, root_collective_group_id, tsk.get_collective_group_id());
		if(m_recorder != nullptr) { *m_recorder << clone_collective_group_instruction_record(*clone_cg_isntr); }

		add_dependency(clone_cg_isntr, root_cg.last_host_task, instruction_dependency_origin::collective_group_order);
		root_cg.last_host_task = clone_cg_isntr;
		m_collective_groups.emplace(cgid, clone_cg_isntr);
	}

	struct reads_writes {
		region<3> reads;
		region<3> writes;

		bool empty() const { return reads.empty() && writes.empty(); }
	};

	struct partial_instruction : localized_chunk {
		device_id did = -1;
		std::unordered_map<buffer_id, reads_writes> rw_map;
		side_effect_map se_map;
		detail::instruction* instruction = nullptr;
	};

	if(tsk.get_execution_target() == execution_target::device && m_system.devices.empty()) { utils::panic("no device on which to execute device kernel"); }

	std::vector<partial_instruction> cmd_instrs;
	{
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::split", Teal, "split");

		const bool is_splittable_locally =
		    tsk.has_variable_split() && tsk.get_side_effect_map().empty() && tsk.get_collective_group_id() == non_collective_group_id;
		const auto split = tsk.get_hint<experimental::hints::split_2d>() != nullptr ? split_2d : split_1d;

		const auto command_sr = ecmd.get_execution_range();
		const auto command_chunk = chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size());

		std::vector<chunk<3>> coarse_chunks;
		if(is_splittable_locally && tsk.get_execution_target() == execution_target::device) {
			coarse_chunks = split(command_chunk, tsk.get_granularity(), m_system.devices.size());
		} else {
			coarse_chunks = {command_chunk};
		}

		size_t oversubscribe_factor = 1;
		if(const auto oversubscribe = tsk.get_hint<experimental::hints::oversubscribe>(); oversubscribe != nullptr) {
			// Our local reduction setup uses the normal per-device backing buffer allocation as the reduction output of each device. Since we can't track
			// overlapping allocations at the moment, we have no way of oversubscribing reduction kernels without introducing a data race between multiple "fine
			// chunks" on the final write. This could be solved by creating separate reduction-output allocations for each device chunk and not touching the
			// actual buffer allocation. This is left as *future work* for a general overhaul of reductions.
			if(is_splittable_locally && tsk.get_reductions().empty()) {
				oversubscribe_factor = oversubscribe->get_factor();
			} else if(m_policy.unsafe_oversubscription_error != error_policy::ignore) {
				utils::report_error(m_policy.unsafe_oversubscription_error, "Refusing to oversubscribe {}{}.", print_task_debug_label(tsk),
				    !tsk.get_reductions().empty()                              ? " because it performs a reduction"
				    : !tsk.get_side_effect_map().empty()                       ? " because it has side effects"
				    : tsk.get_collective_group_id() != non_collective_group_id ? " because it participates in a collective group"
				    : !tsk.has_variable_split()                                ? " because its iteration space cannot be split"
				                                                               : "");
			}
		}

		for(size_t i = 0; i < coarse_chunks.size(); ++i) {
			for(const auto& fine_chunk : split(coarse_chunks[i], tsk.get_granularity(), oversubscribe_factor)) {
				auto& instr = cmd_instrs.emplace_back();
				instr.execution_range = box(subrange(fine_chunk.offset, fine_chunk.range));
				if(tsk.get_execution_target() == execution_target::device) {
					assert(i < m_system.devices.size());
					instr.did = device_id(i);
					instr.memory_id = m_system.devices[i].native_memory;
				} else {
					instr.memory_id = host_memory_id;
				}
			}
		}
	}

	// detect overlapping writes

	if(m_policy.overlapping_write_error != error_policy::ignore) {
		box_vector<3> local_chunk_boxes(cmd_instrs.size(), box<3>());
		std::transform(cmd_instrs.begin(), cmd_instrs.end(), local_chunk_boxes.begin(), [](const partial_instruction& pi) { return pi.execution_range; });

		if(const auto overlapping_writes = detect_overlapping_writes(tsk, local_chunk_boxes); !overlapping_writes.empty()) {
			auto error = fmt::format("{} has overlapping writes on N{} in", print_task_debug_label(tsk, true /* title case */), m_local_nid);
			for(const auto& [bid, overlap] : overlapping_writes) {
				fmt::format_to(std::back_inserter(error), " {} {}", print_buffer_debug_label(bid), overlap);
			}
			error += ". Choose a non-overlapping range mapper for the write access or constrain the split to make the access non-overlapping.";
			utils::report_error(m_policy.overlapping_write_error, "{}", error);
		}
	}

	// buffer requirements

	auto accessed_bids = bam.get_accessed_buffers();
	for(const auto& rinfo : tsk.get_reductions()) {
		accessed_bids.insert(rinfo.bid);
	}
	for(const auto bid : accessed_bids) {
		satisfy_buffer_requirements(command_batch, bid, tsk, ecmd.get_execution_range(), ecmd.is_reduction_initializer(),
		    std::vector<localized_chunk>(cmd_instrs.begin(), cmd_instrs.end()));
	}

	struct partial_local_reduction {
		bool include_current_value = false;
		size_t current_value_offset = 0;
		size_t num_chunks = 1;
		size_t chunk_size = 0;
		allocation_id gather_aid = null_allocation_id;
		alloc_instruction* gather_alloc_instr = nullptr;
	};
	std::vector<partial_local_reduction> local_reductions(tsk.get_reductions().size());

	for(size_t i = 0; i < tsk.get_reductions().size(); ++i) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::collect_reduction", LightSkyBlue, "collect reductions");

		auto& red = local_reductions[i];
		const auto [rid, bid, reduction_task_includes_buffer_value] = tsk.get_reductions()[i];
		auto& buffer = m_buffers.at(bid);

		red.include_current_value = reduction_task_includes_buffer_value && ecmd.is_reduction_initializer();
		red.current_value_offset = red.include_current_value ? 1 : 0;
		red.num_chunks = red.current_value_offset + cmd_instrs.size();
		red.chunk_size = scalar_reduction_box.get_area() * buffer.elem_size;

		if(red.num_chunks <= 1) continue;

		red.gather_aid = new_allocation_id(host_memory_id);
		red.gather_alloc_instr = create<alloc_instruction>(command_batch, red.gather_aid, red.num_chunks * red.chunk_size, buffer.elem_align);
		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(*red.gather_alloc_instr, alloc_instruction_record::alloc_origin::gather,
			    buffer_allocation_record{bid, buffer.debug_name, scalar_reduction_box}, red.num_chunks);
		}

		add_dependency(red.gather_alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

		if(red.include_current_value) {
			auto source_allocation =
			    buffer.memories[host_memory_id].find_contiguous_allocation(scalar_reduction_box); // provided by satisfy_buffer_requirements
			assert(source_allocation != nullptr);

			// copy to local gather space

			const auto current_value_copy_instr = create<copy_instruction>(
			    command_batch, source_allocation->aid, red.gather_aid, source_allocation->box, scalar_reduction_box, scalar_reduction_box, buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(*current_value_copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.debug_name);
			}

			add_dependency(current_value_copy_instr, red.gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);
			read_from_allocation(current_value_copy_instr, *source_allocation, scalar_reduction_box);
		}
	}

	// collect updated regions

	for(const auto bid : bam.get_accessed_buffers()) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::collect_accesses", IndianRed, "collect reads/writes");

		for(auto& insn : cmd_instrs) {
			reads_writes rw;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.execution_range.get_subrange(), tsk.get_global_size());
				if(access::mode_traits::is_consumer(mode)) { rw.reads = region_union(rw.reads, req); }
				if(access::mode_traits::is_producer(mode)) { rw.writes = region_union(rw.writes, req); }
			}
			insn.rw_map.emplace(bid, std::move(rw));
		}
	}

	for(const auto& rinfo : tsk.get_reductions()) {
		for(auto& instr : cmd_instrs) {
			auto& rw_map = instr.rw_map[rinfo.bid]; // allow default-insert
			rw_map.writes = region_union(rw_map.writes, scalar_reduction_box);
		}
	}

	// collect side effects

	if(!tsk.get_side_effect_map().empty()) {
		assert(cmd_instrs.size() == 1); // split instructions for host tasks with side effects would race
		assert(cmd_instrs[0].memory_id == host_memory_id);
		cmd_instrs[0].se_map = tsk.get_side_effect_map();
	}

	// 4) create the actual command instructions

	for(auto& instr : cmd_instrs) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::create_instruction", Coral, "create instruction");

		buffer_access_allocation_map allocation_map(bam.get_num_accesses());
		buffer_access_allocation_map reduction_map(tsk.get_reductions().size());

		std::vector<buffer_memory_record> buffer_memory_access_map;       // if (m_recorder)
		std::vector<buffer_reduction_record> buffer_memory_reduction_map; // if (m_recorder)
		if(m_recorder != nullptr) {
			buffer_memory_access_map.resize(bam.get_num_accesses());
			buffer_memory_reduction_map.resize(tsk.get_reductions().size());
		}

		for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
			const auto [bid, mode] = bam.get_nth_access(i);
			const auto accessed_box = bam.get_requirements_for_nth_access(i, tsk.get_dimensions(), instr.execution_range.get_subrange(), tsk.get_global_size());
			const auto& buffer = m_buffers.at(bid);
			if(!accessed_box.empty()) {
				const auto& allocations = buffer.memories[instr.memory_id].allocations;
				const auto allocation_it =
				    std::find_if(allocations.begin(), allocations.end(), [&](const buffer_allocation_state& alloc) { return alloc.box.covers(accessed_box); });
				assert(allocation_it != allocations.end());
				const auto& alloc = *allocation_it;
				allocation_map[i] =
				    buffer_access_allocation{alloc.aid, alloc.box, accessed_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.debug_name)};
				if(m_recorder != nullptr) { buffer_memory_access_map[i] = buffer_memory_record{bid, buffer.debug_name}; }
			} else {
				allocation_map[i] = buffer_access_allocation{null_allocation_id, {}, {} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.debug_name)};
				if(m_recorder != nullptr) { buffer_memory_access_map[i] = buffer_memory_record{bid, buffer.debug_name}; }
			}
		}

		for(size_t i = 0; i < tsk.get_reductions().size(); ++i) {
			const auto& rinfo = tsk.get_reductions()[i];
			// TODO copy-pasted from directly above
			const auto& buffer = m_buffers.at(rinfo.bid);
			const auto& allocations = buffer.memories[instr.memory_id].allocations;
			const auto allocation_it = std::find_if(
			    allocations.begin(), allocations.end(), [&](const buffer_allocation_state& alloc) { return alloc.box.covers(scalar_reduction_box); });
			assert(allocation_it != allocations.end());
			const auto& alloc = *allocation_it;
			reduction_map[i] =
			    buffer_access_allocation{alloc.aid, alloc.box, scalar_reduction_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, rinfo.bid, buffer.debug_name)};
			if(m_recorder != nullptr) { buffer_memory_reduction_map[i] = buffer_reduction_record{rinfo.bid, buffer.debug_name, rinfo.rid}; }
		}

		if(tsk.get_execution_target() == execution_target::device) {
			assert(instr.execution_range.get_area() > 0);
			assert(instr.memory_id != host_memory_id);
			// TODO how do I know it's a SYCL kernel and not a CUDA kernel?
			const auto device_kernel_instr =
			    create<device_kernel_instruction>(command_batch, instr.did, tsk.get_launcher<device_kernel_launcher>(), instr.execution_range,
			        std::move(allocation_map), std::move(reduction_map) CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, tsk.get_id(), tsk.get_debug_name()));
			if(m_recorder != nullptr) {
				*m_recorder << device_kernel_instruction_record(
				    *device_kernel_instr, ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), buffer_memory_access_map, buffer_memory_reduction_map);
			}
			instr.instruction = device_kernel_instr;
		} else {
			assert(tsk.get_execution_target() == execution_target::host);
			assert(instr.memory_id == host_memory_id);
			assert(reduction_map.empty());
			const auto host_task_instr =
			    create<host_task_instruction>(command_batch, tsk.get_launcher<host_task_launcher>(), instr.execution_range, tsk.get_global_size(),
			        std::move(allocation_map), tsk.get_collective_group_id() CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, tsk.get_id(), tsk.get_debug_name()));
			if(m_recorder != nullptr) {
				*m_recorder << host_task_instruction_record(*host_task_instr, ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), buffer_memory_access_map);
			}
			instr.instruction = host_task_instr;
		}
	}

	// 5) compute dependencies between command instructions and previous copy, allocation, and command (!) instructions

	// TODO this will not work correctly for oversubscription
	//	 - read-all + write-1:1 cannot be oversubscribed at all, chunks would need a global read->write barrier (how would the kernel even look like?)
	//	 - oversubscribed host tasks would need dependencies between their chunks based on side effects and collective groups
	for(const auto& instr : cmd_instrs) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::dependencies", DarkGray, "dependencies");

		for(const auto& [bid, rw] : instr.rw_map) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[instr.memory_id];

			for(auto& allocation : memory.allocations) {
				add_dependencies_on_last_writers(
				    instr.instruction, allocation, region_intersection(rw.reads, allocation.box), instruction_dependency_origin::read_from_allocation);
				add_dependencies_on_access_front(
				    instr.instruction, allocation, region_intersection(rw.writes, allocation.box), instruction_dependency_origin::write_to_allocation);
			}
		}
		for(const auto& [hoid, order] : instr.se_map) {
			assert(instr.memory_id == host_memory_id);
			if(const auto last_side_effect = m_host_objects.at(hoid).last_side_effect) {
				add_dependency(instr.instruction, last_side_effect, instruction_dependency_origin::side_effect);
			}
		}
		if(const auto cgid = tsk.get_collective_group_id(); cgid != non_collective_group_id) {
			assert(instr.memory_id == host_memory_id);
			auto& group = m_collective_groups.at(cgid); // created previously with clone_collective_group_instruction
			add_dependency(instr.instruction, group.last_host_task, instruction_dependency_origin::collective_group_order);
		}
	}

	// 6a) clear region maps in the areas we write to - we gracefully handle overlapping writes by treating the set of all conflicting writers as last writers
	// of an allocation

	for(const auto& instr : cmd_instrs) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::begin_access", LightSeaGreen, "begin accesses");

		for(const auto& [bid, rw] : instr.rw_map) {
			assert(instr.instruction != nullptr);
			auto& buffer = m_buffers.at(bid);
			for(auto& alloc : buffer.memories[instr.memory_id].allocations) {
				alloc.begin_combined_write(region_intersection(alloc.box, rw.writes));
			}
		}
	}

	// 6b) update data locations and last writers resulting from command instructions

	for(const auto& instr : cmd_instrs) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::commit_access", LightSeaGreen, "commit accesses");

		for(const auto& [bid, rw] : instr.rw_map) {
			assert(instr.instruction != nullptr);
			auto& buffer = m_buffers.at(bid);

			// TODO can we merge this with loop 5) and use read_from_allocation / write_to_allocation?
			// if so, have similar functions for side effects / collective groups
			for(auto& alloc : buffer.memories[instr.memory_id].allocations) {
				alloc.commit_read(region_intersection(alloc.box, rw.reads), instr.instruction);
				alloc.commit_combined_write(region_intersection(alloc.box, rw.writes), instr.instruction);
			}

			buffer.commit_original_write(rw.writes, instr.instruction, instr.memory_id);
		}
		for(const auto& [hoid, order] : instr.se_map) {
			assert(instr.memory_id == host_memory_id);
			m_host_objects.at(hoid).last_side_effect = instr.instruction;
		}
		if(const auto cgid = tsk.get_collective_group_id(); cgid != non_collective_group_id) {
			assert(instr.memory_id == host_memory_id);
			m_collective_groups.at(tsk.get_collective_group_id()).last_host_task = instr.instruction;
		}
	}

	// 7) insert local reduction instructions and update tracking structures accordingly

	for(size_t i = 0; i < tsk.get_reductions().size(); ++i) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::local_reduction", DeepSkyBlue, "local reduction");
		const auto& red = local_reductions[i];

		// for a single-device configuration that doesn't include the current value, the above update of last writers is sufficient
		if(red.num_chunks == 1) continue;

		const auto [rid, bid, reduction_task_includes_buffer_value] = tsk.get_reductions()[i];
		auto& buffer = m_buffers.at(bid);
		auto& host_memory = buffer.memories[host_memory_id];

		// gather space has been allocated before in order to preserve the current buffer value where necessary
		std::vector<copy_instruction*> gather_copy_instrs;
		gather_copy_instrs.reserve(cmd_instrs.size());
		for(size_t j = 0; j < cmd_instrs.size(); ++j) {
			const auto source_mid = cmd_instrs[j].memory_id;
			auto source_allocation = buffer.memories[source_mid].find_contiguous_allocation(scalar_reduction_box);
			assert(source_allocation != nullptr);

			// copy to local gather space

			const auto copy_instr =
			    create<copy_instruction>(command_batch, source_allocation->aid, red.gather_aid + (red.current_value_offset + j) * buffer.elem_size,
			        source_allocation->box, scalar_reduction_box, scalar_reduction_box, buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.debug_name);
			}

			add_dependency(copy_instr, red.gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);
			read_from_allocation(copy_instr, *source_allocation, scalar_reduction_box);

			gather_copy_instrs.push_back(copy_instr);
		}

		const auto dest_allocation = host_memory.find_contiguous_allocation(scalar_reduction_box);
		assert(dest_allocation != nullptr);

		// reduce

		const auto reduce_instr = create<reduce_instruction>(command_batch, rid, red.gather_aid, red.num_chunks, dest_allocation->aid);
		if(m_recorder != nullptr) {
			*m_recorder << reduce_instruction_record(
			    *reduce_instr, std::nullopt, bid, buffer.debug_name, scalar_reduction_box, reduce_instruction_record::reduction_scope::local);
		}

		for(auto& copy_instr : gather_copy_instrs) {
			add_dependency(reduce_instr, copy_instr, instruction_dependency_origin::read_from_allocation);
		}
		write_to_allocation(reduce_instr, *dest_allocation, scalar_reduction_box);
		buffer.commit_original_write(scalar_reduction_box, reduce_instr, host_memory_id);

		// free local gather space

		const auto gather_free_instr = create<free_instruction>(command_batch, red.gather_aid);
		if(m_recorder != nullptr) { *m_recorder << free_instruction_record(*gather_free_instr, red.num_chunks * red.chunk_size, std::nullopt); }

		add_dependency(gather_free_instr, reduce_instr, instruction_dependency_origin::allocation_lifetime);
	}

	// 7) insert epoch and horizon dependencies, apply epochs, optionally record the instruction

	for(auto& instr : cmd_instrs) {
		// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
		// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
		if(instr.instruction->get_dependencies().empty()) { add_dependency(instr.instruction, m_last_epoch, instruction_dependency_origin::last_epoch); }
	}
}


void impl::compile_push_command(batch& command_batch, const push_command& pcmd) {
	// Prioritize all instructions participating in a "push" to hide the latency of establishing local coherence behind the typically much longer latencies of
	// inter-node communication
	command_batch.base_priority = 10;

	const auto trid = pcmd.get_transfer_id();

	auto& buffer = m_buffers.at(trid.bid);
	const auto push_box = box(pcmd.get_range());

	// We want to generate the fewest number of send instructions possible without introducing new synchronization points between chunks of the same
	// command that generated the pushed data. This will allow compute-transfer overlap, especially in the case of oversubscribed splits.
	std::unordered_map<instruction_id, box_vector<3>> individual_send_boxes;
	// Since we now send boxes individually, we do not need to allocate the entire push_box contiguously.
	box_vector<3> required_host_allocation;
	for(auto& [box, writer] : buffer.original_writers.get_region_values(push_box)) {
		individual_send_boxes[writer->get_id()].push_back(box);
		required_host_allocation.push_back(box);
	}

	allocate_contiguously(command_batch, trid.bid, host_memory_id, std::move(required_host_allocation));

	for(auto& [_, boxes] : individual_send_boxes) {
		// Since the original writer is unique for each buffer item, all writer_regions will be disjoint and we can pull data from device memories for each
		// writer without any effect from the order of operations
		locally_satisfy_read_requirements(command_batch, trid.bid, host_memory_id, {region(box_vector<3>(boxes))});
	}

	for(auto& [_, boxes] : individual_send_boxes) {
		for(const auto& full_box : boxes) {
			for(const auto& box : instruction_graph_generator_detail::split_into_communicator_compatible_boxes(buffer.range, full_box)) {
				const message_id msgid = create_outbound_pilot(command_batch, pcmd.get_target(), trid, box);

				const auto allocation = buffer.memories[host_memory_id].find_contiguous_allocation(box);
				assert(allocation != nullptr); // we allocate_contiguously above

				const auto offset_in_allocation = box.get_offset() - allocation->box.get_offset();
				const auto send_instr = create<send_instruction>(command_batch, pcmd.get_target(), msgid, allocation->aid, allocation->box.get_range(),
				    offset_in_allocation, box.get_range(), buffer.elem_size);
				if(m_recorder != nullptr) { *m_recorder << send_instruction_record(*send_instr, pcmd.get_cid(), trid, buffer.debug_name, box.get_offset()); }

				read_from_allocation(send_instr, *allocation, box);
			}
		}
	}

	// If not all nodes contribute partial results to a global reductions, the remaining ones need to notify their peers that they should not expect any data.
	// This is done by announcing an empty box through the pilot message.
	assert(push_box.empty() == individual_send_boxes.empty());
	if(individual_send_boxes.empty()) {
		assert(trid.rid != no_reduction_id);
		create_outbound_pilot(command_batch, pcmd.get_target(), trid, box<3>());
	}
}


void impl::defer_await_push_command(const await_push_command& apcmd) {
	// We do not generate instructions for await-push commands immediately upon receiving them; instead, we buffer them and generate
	// recv-instructions as soon as data is to be read by another instruction. This way, we can split the recv instructions and avoid
	// unnecessary synchronization points between chunks that can otherwise profit from a transfer-compute overlap.

	const auto& trid = apcmd.get_transfer_id();
	if(m_recorder != nullptr) { m_recorder->record_await_push_command_id(trid, apcmd.get_cid()); }

	auto& buffer = m_buffers.at(trid.bid);

#ifndef NDEBUG
	for(const auto& receive : buffer.pending_receives) {
		assert((trid.rid != no_reduction_id || receive.consumer_tid != trid.consumer_tid)
		       && "received multiple await-pushes for the same consumer-task, buffer and reduction id");
		assert(region_intersection(receive.received_region, apcmd.get_region()).empty()
		       && "received an await-push command into a previously await-pushed region without an intermediate read");
	}
	for(const auto& gather : buffer.pending_gathers) {
		assert(std::pair(gather.consumer_tid, gather.rid) != std::pair(trid.consumer_tid, gather.rid)
		       && "received multiple await-pushes for the same consumer-task, buffer and reduction id");
		assert(region_intersection(gather.gather_box, apcmd.get_region()).empty()
		       && "received an await-push command into a previously await-pushed region without an intermediate read");
	}
#endif

	if(trid.rid == 0) {
		buffer.pending_receives.emplace_back(
		    trid.consumer_tid, apcmd.get_region(), instruction_graph_generator_detail::connected_subregion_bounding_boxes(apcmd.get_region()));
	} else {
		assert(apcmd.get_region().get_boxes().size() == 1);
		buffer.pending_gathers.emplace_back(trid.consumer_tid, trid.rid, apcmd.get_region().get_boxes().front());
	}
}


void impl::compile_reduction_command(batch& command_batch, const reduction_command& rcmd) {
	const auto scalar_reduction_box = box<3>({0, 0, 0}, {1, 1, 1});
	const auto [rid, bid, init_from_buffer] = rcmd.get_reduction_info();

	auto& buffer = m_buffers.at(bid);

	assert(buffer.pending_gathers.size() == 1 && "received reduction command that is not preceded by an appropriate await-push");
	const auto& gather = buffer.pending_gathers.front();
	assert(gather.gather_box == scalar_reduction_box);

	// allocate the gather space

	const auto gather_aid = new_allocation_id(host_memory_id);
	const auto node_chunk_size = gather.gather_box.get_area() * buffer.elem_size;
	const auto gather_alloc_instr = create<alloc_instruction>(command_batch, gather_aid, m_num_nodes * node_chunk_size, buffer.elem_align);
	if(m_recorder != nullptr) {
		*m_recorder << alloc_instruction_record(*gather_alloc_instr, alloc_instruction_record::alloc_origin::gather,
		    buffer_allocation_record{bid, buffer.debug_name, gather.gather_box}, m_num_nodes);
	}
	add_dependency(gather_alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

	// fill the gather space with the reduction identity, so that the gather_receive_command can simply ignore empty boxes sent by peers that do not contribute
	// to the reduction, and we can skip the gather-copy instruction if we ourselves do not contribute a partial result.

	const auto fill_identity_instr = create<fill_identity_instruction>(command_batch, rid, gather_aid, m_num_nodes);
	if(m_recorder != nullptr) { *m_recorder << fill_identity_instruction_record(*fill_identity_instr); }
	add_dependency(fill_identity_instr, gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);

	// if the local node contributes to the reduction, copy the contribution to the appropriate position in the gather space

	copy_instruction* local_gather_copy_instr = nullptr;
	const auto contribution_location = buffer.up_to_date_memories.get_region_values(scalar_reduction_box).front().second;
	if(contribution_location.any()) {
		const auto source_mid = next_location(contribution_location, host_memory_id);
		const auto source_allocation = buffer.memories[source_mid].find_contiguous_allocation(scalar_reduction_box);
		assert(source_allocation != nullptr); // if scalar_box is up to date in that memory, it (the single element) must also be contiguous

		local_gather_copy_instr = create<copy_instruction>(command_batch, source_allocation->aid, gather_aid + m_local_nid * buffer.elem_size,
		    source_allocation->box, scalar_reduction_box, scalar_reduction_box, buffer.elem_size);
		if(m_recorder != nullptr) {
			*m_recorder << copy_instruction_record(*local_gather_copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.debug_name);
		}
		add_dependency(local_gather_copy_instr, fill_identity_instr, instruction_dependency_origin::write_to_allocation);
		read_from_allocation(local_gather_copy_instr, *source_allocation, scalar_reduction_box);
	}

	// gather remote contributions

	const transfer_id trid(gather.consumer_tid, bid, gather.rid);
	const auto gather_recv_instr = create<gather_receive_instruction>(command_batch, trid, gather_aid, node_chunk_size);
	if(m_recorder != nullptr) { *m_recorder << gather_receive_instruction_record(*gather_recv_instr, buffer.debug_name, gather.gather_box, m_num_nodes); }
	add_dependency(gather_recv_instr, fill_identity_instr, instruction_dependency_origin::write_to_allocation);

	// perform the global reduction

	allocate_contiguously(command_batch, bid, host_memory_id, {scalar_reduction_box});

	auto& host_memory = buffer.memories[host_memory_id];
	auto dest_allocation = host_memory.find_contiguous_allocation(scalar_reduction_box);
	assert(dest_allocation != nullptr);

	const auto reduce_instr = create<reduce_instruction>(command_batch, rid, gather_aid, m_num_nodes, dest_allocation->aid);
	if(m_recorder != nullptr) {
		*m_recorder << reduce_instruction_record(
		    *reduce_instr, rcmd.get_cid(), bid, buffer.debug_name, scalar_reduction_box, reduce_instruction_record::reduction_scope::global);
	}
	add_dependency(reduce_instr, gather_recv_instr, instruction_dependency_origin::read_from_allocation);
	if(local_gather_copy_instr != nullptr) { add_dependency(reduce_instr, local_gather_copy_instr, instruction_dependency_origin::read_from_allocation); }
	write_to_allocation(reduce_instr, *dest_allocation, scalar_reduction_box);
	buffer.commit_original_write(scalar_reduction_box, reduce_instr, host_memory_id);

	// free the gather space

	const auto gather_free_instr = create<free_instruction>(command_batch, gather_aid);
	if(m_recorder != nullptr) { *m_recorder << free_instruction_record(*gather_free_instr, m_num_nodes * node_chunk_size, std::nullopt); }
	add_dependency(gather_free_instr, reduce_instr, instruction_dependency_origin::allocation_lifetime);

	buffer.pending_gathers.clear();
}


void impl::compile_fence_command(batch& command_batch, const fence_command& fcmd) {
	const auto& tsk = *m_tm->get_task(fcmd.get_tid());
	assert(tsk.get_reductions().empty());

	const auto& bam = tsk.get_buffer_access_map();
	const auto& sem = tsk.get_side_effect_map();
	assert(bam.get_num_accesses() + sem.size() == 1);

	if(bam.get_num_accesses() != 0) {
		// fences encode their buffer requirements through buffer_access_map with a fixed range mapper (this is rather ugly)
		const auto bid = *bam.get_accessed_buffers().begin();
		const auto fence_region = bam.get_mode_requirements(bid, access_mode::read, 0, {}, zeros);
		const auto fence_box = !fence_region.empty() ? fence_region.get_boxes().front() : box<3>();

		const auto user_allocation_id = tsk.get_fence_promise()->get_user_allocation_id();
		auto& buffer = m_buffers.at(bid);

		copy_instruction* copy_instr = nullptr;
		// gracefully handle empty-range buffer fences
		if(!fence_box.empty()) {
			// We make the host buffer coherent first in order to apply pending await-pushes.
			// TODO this enforces a contiguous host-buffer allocation which may cause unnecessary resizes.
			const std::vector chunks{localized_chunk{host_memory_id, {}}};
			satisfy_buffer_requirements(command_batch, bid, tsk, {}, false /* is_reduction_initializer: irrelevant */, chunks);

			const auto host_buffer_allocation = buffer.memories[host_memory_id].find_contiguous_allocation(fence_box);
			assert(host_buffer_allocation != nullptr);

			copy_instr = create<copy_instruction>(
			    command_batch, host_buffer_allocation->aid, user_allocation_id, host_buffer_allocation->box, fence_box, fence_box, buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::fence, bid, buffer.debug_name);
			}

			read_from_allocation(copy_instr, *host_buffer_allocation, fence_box);
		}

		const auto fence_instr = create<fence_instruction>(command_batch, tsk.get_fence_promise());
		if(m_recorder != nullptr) {
			*m_recorder << fence_instruction_record(*fence_instr, tsk.get_id(), fcmd.get_cid(), bid, buffer.debug_name, fence_box.get_subrange());
		}

		if(!fence_box.empty()) {
			add_dependency(fence_instr, copy_instr, instruction_dependency_origin::read_from_allocation);
		} else {
			// an empty-range buffer fence has no data dependencies but must still be executed to fulfill its promise - attach it to the current epoch.
			add_dependency(fence_instr, m_last_epoch, instruction_dependency_origin::last_epoch);
		}

		// we will just assume that the runtime does not intend to re-use this allocation
		m_unreferenced_user_allocations.push_back(user_allocation_id);
	}

	if(!sem.empty()) {
		// host-object fences encode their host-object id in the task side effect map (which is also very ugly)
		const auto [hoid, _] = *sem.begin();

		auto& obj = m_host_objects.at(hoid);
		const auto fence_instr = create<fence_instruction>(command_batch, tsk.get_fence_promise());
		if(m_recorder != nullptr) { *m_recorder << fence_instruction_record(*fence_instr, tsk.get_id(), fcmd.get_cid(), hoid); }

		add_dependency(fence_instr, obj.last_side_effect, instruction_dependency_origin::side_effect);
		obj.last_side_effect = fence_instr;
	}
}


void impl::compile_horizon_command(batch& command_batch, const horizon_command& hcmd) {
	m_idag->begin_epoch(hcmd.get_tid());
	instruction_garbage garbage{hcmd.get_completed_reductions(), std::move(m_unreferenced_user_allocations)};
	const auto horizon = create<horizon_instruction>(command_batch, hcmd.get_tid(), std::move(garbage));
	if(m_recorder != nullptr) { *m_recorder << horizon_instruction_record(*horizon, hcmd.get_cid()); }

	collapse_execution_front_to(horizon);
	if(m_last_horizon != nullptr) { apply_epoch(m_last_horizon); }
	m_last_horizon = horizon;
}


void impl::compile_epoch_command(batch& command_batch, const epoch_command& ecmd) {
	m_idag->begin_epoch(ecmd.get_tid());
	instruction_garbage garbage{ecmd.get_completed_reductions(), std::move(m_unreferenced_user_allocations)};
	const auto epoch = create<epoch_instruction>(command_batch, ecmd.get_tid(), ecmd.get_epoch_action(), std::move(garbage));
	if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*epoch, ecmd.get_cid()); }

	collapse_execution_front_to(epoch);
	apply_epoch(epoch);
	m_last_horizon = nullptr;
}


void impl::flush_batch(batch&& batch) {
	// sanity check: every instruction except the initial epoch must be temporally anchored through at least one dependency
	assert(std::all_of(batch.generated_instructions.begin(), batch.generated_instructions.end(),
	    [](const auto instr) { return instr->get_id() == 0 || !instr->get_dependencies().empty(); }));
	assert(instruction_graph_generator_detail::is_topologically_sorted(batch.generated_instructions.begin(), batch.generated_instructions.end()));

	if(m_delegate != nullptr && !batch.generated_instructions.empty()) { m_delegate->flush_instructions(std::move(batch.generated_instructions)); }
	if(m_delegate != nullptr && !batch.generated_pilots.empty()) { m_delegate->flush_outbound_pilots(std::move(batch.generated_pilots)); }

#ifndef NDEBUG // ~batch() checks if it has been flushed, which we want to acknowledge even if m_delegate == nullptr
	batch.generated_instructions = {};
	batch.generated_pilots = {};
#endif
}

void impl::compile(const abstract_command& cmd) {
	batch command_batch;
	matchbox::match(
	    cmd,                                                                                    //
	    [&](const execution_command& ecmd) { compile_execution_command(command_batch, ecmd); }, //
	    [&](const push_command& pcmd) { compile_push_command(command_batch, pcmd); },           //
	    [&](const await_push_command& apcmd) { defer_await_push_command(apcmd); },              //
	    [&](const horizon_command& hcmd) { compile_horizon_command(command_batch, hcmd); },     //
	    [&](const epoch_command& ecmd) { compile_epoch_command(command_batch, ecmd); },         //
	    [&](const reduction_command& rcmd) { compile_reduction_command(command_batch, rcmd); }, //
	    [&](const fence_command& fcmd) { compile_fence_command(command_batch, fcmd); }          //
	);
	flush_batch(std::move(command_batch));
}

std::string impl::print_buffer_debug_label(const buffer_id bid) const {
	const auto& debug_name = m_buffers.at(bid).debug_name;
	return !debug_name.empty() ? fmt::format("B{} \"{}\"", bid, debug_name) : fmt::format("B{}", bid);
}

} // namespace celerity::detail::instruction_graph_generator_detail

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, const size_t num_nodes, const node_id local_nid, system_info system,
    instruction_graph& idag, delegate* dlg, instruction_recorder* const recorder, const policy_set& policy)
    : m_impl(new instruction_graph_generator_detail::impl(tm, num_nodes, local_nid, std::move(system), idag, dlg, recorder, policy)) {}

instruction_graph_generator::~instruction_graph_generator() = default;

void instruction_graph_generator::create_buffer(
    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::create_buffer", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("create buffer B{}", bid);
	m_impl->create_buffer(bid, range, elem_size, elem_align, user_allocation_id);
}

void instruction_graph_generator::set_buffer_debug_name(const buffer_id bid, const std::string& name) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::set_buffer_debug_name", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("set buffer name B{} \"{}\"", bid, name);
	m_impl->set_buffer_debug_name(bid, name);
}

void instruction_graph_generator::destroy_buffer(const buffer_id bid) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::destroy_buffer", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("destroy buffer B{}", bid);
	m_impl->destroy_buffer(bid);
}

void instruction_graph_generator::create_host_object(const host_object_id hoid, const bool owns_instance) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::create_host_object", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("create host object H{}", hoid);
	m_impl->create_host_object(hoid, owns_instance);
}

void instruction_graph_generator::destroy_host_object(const host_object_id hoid) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::destroy_host_object", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("destroy host object H{}", hoid);
	m_impl->destroy_host_object(hoid);
}

// Resulting instructions are in topological order of dependencies (i.e. sequential execution would fulfill all internal dependencies)
void instruction_graph_generator::compile(const abstract_command& cmd) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("idag::compile_command", NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("compile C{}", cmd.get_cid());
	m_impl->compile(cmd);
}

} // namespace celerity::detail
