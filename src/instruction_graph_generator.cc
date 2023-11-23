#include "instruction_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
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


namespace celerity::detail {

class instruction_graph_generator::impl {
  public:
	impl(const task_manager& tm, size_t num_nods, node_id local_nid, system_info system, instruction_graph& idag, instruction_recorder* recorder,
	    const policy_set& policy);

	void create_buffer(buffer_id bid, int dims, const range<3>& range, size_t elem_size, size_t elem_align, bool host_initialized);

	void set_buffer_debug_name(buffer_id bid, const std::string& name);

	void destroy_buffer(buffer_id bid);

	void create_host_object(host_object_id hoid, bool owns_instance);

	void destroy_host_object(host_object_id hoid);

	// Resulting instructions are in topological order of dependencies (i.e. sequential execution would fulfill all internal dependencies)
	std::pair<std::vector<const instruction*>, std::vector<outbound_pilot>> compile(const abstract_command& cmd);

  private:
	using data_location = std::bitset<max_num_memories>;

	struct buffer_memory_per_allocation_data {
		struct access_front {
			gch::small_vector<instruction*> front; // sorted by id to allow equality comparison
			enum { read, write } mode = write;

			friend bool operator==(const access_front& lhs, const access_front& rhs) { return lhs.front == rhs.front && lhs.mode == rhs.mode; }
			friend bool operator!=(const access_front& lhs, const access_front& rhs) { return !(lhs == rhs); }
		};

		struct allocated_box {
			allocation_id aid;
			box<3> box;

			friend bool operator==(const allocated_box& lhs, const allocated_box& rhs) { return lhs.aid == rhs.aid && lhs.box == rhs.box; }
			friend bool operator!=(const allocated_box& lhs, const allocated_box& rhs) { return !(lhs == rhs); }
		};

		allocation_id aid;
		detail::box<3> box;
		region_map<instruction*> last_writers;  // in virtual-buffer coordinates
		region_map<access_front> access_fronts; // in virtual-buffer coordinates

		explicit buffer_memory_per_allocation_data(int buffer_dims, const allocation_id aid, const detail::box<3>& allocated_box, const range<3>& buffer_range)
		    : aid(aid), box(allocated_box), last_writers(buffer_range, buffer_dims), access_fronts(buffer_range, buffer_dims) {}

		void record_read(const region<3>& region, instruction* const instr) {
			for(auto& [box, record] : access_fronts.get_region_values(region)) {
				if(record.mode == access_front::read) {
					// sorted insert
					const auto at = std::lower_bound(record.front.begin(), record.front.end(), instr, instruction_id_less());
					assert(at == record.front.end() || *at != instr);
					record.front.insert(at, instr);
				} else {
					record = {{instr}, access_front::read};
				}
				assert(std::is_sorted(record.front.begin(), record.front.end(), instruction_id_less()));
				access_fronts.update_region(box, record);
			}
		}

		void record_write(const region<3>& region, instruction* const instr) {
			last_writers.update_region(region, instr);
			access_fronts.update_region(region, access_front{{instr}, access_front::write});
		}

		void apply_epoch(instruction* const epoch) {
			last_writers.apply_to_values([epoch](instruction* const instr) -> instruction* {
				if(instr == nullptr) return nullptr;
				return instr->get_id() > epoch->get_id() ? instr : epoch;
			});
			access_fronts.apply_to_values([epoch](access_front record) {
				const auto new_front_end = std::remove_if(record.front.begin(), record.front.end(), //
				    [epoch](instruction* const instr) { return instr->get_id() < epoch->get_id(); });
				if(new_front_end != record.front.end()) {
					record.front.erase(new_front_end, record.front.end());
					record.front.push_back(epoch);
				}
				assert(std::is_sorted(record.front.begin(), record.front.end(), instruction_id_less()));
				return record;
			});
		}
	};

	struct buffer_per_memory_data {
		// TODO bound the number of allocations per buffer in order to avoid runaway tracking overhead (similar to horizons)
		std::vector<buffer_memory_per_allocation_data> allocations; // disjoint

		const buffer_memory_per_allocation_data& get_allocation(const allocation_id aid) const {
			const auto it = std::find_if(allocations.begin(), allocations.end(), [=](const buffer_memory_per_allocation_data& a) { return a.aid == aid; });
			assert(it != allocations.end());
			return *it;
		}

		buffer_memory_per_allocation_data& get_allocation(const allocation_id aid) {
			return const_cast<buffer_memory_per_allocation_data&>(std::as_const(*this).get_allocation(aid));
		}

		const buffer_memory_per_allocation_data* find_contiguous_allocation(const box<3>& box) const {
			const auto it = std::find_if(allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& a) { return a.box.covers(box); });
			return it != allocations.end() ? &*it : nullptr;
		}

		buffer_memory_per_allocation_data* find_contiguous_allocation(const box<3>& box) {
			return const_cast<buffer_memory_per_allocation_data*>(std::as_const(*this).find_contiguous_allocation(box));
		}

		bool is_allocated_contiguously(const box<3>& box) const { return find_contiguous_allocation(box) != nullptr; }

		void apply_epoch(instruction* const epoch) {
			for(auto& alloc : allocations) {
				alloc.apply_epoch(epoch);
			}
		}
	};

	struct per_buffer_data {
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

		std::string name;
		int dims;
		range<3> range;
		size_t elem_size;
		size_t elem_align;
		std::vector<buffer_per_memory_data> memories;
		region_map<data_location> newest_data_location; // TODO rename for vs original_write_memories?
		region_map<instruction*> original_writers;
		region_map<memory_id> original_write_memories; // only meaningful if newest_data_location[box] is non-empty

		// We store pending receives (await push regions) in a vector instead of a region map since we must process their entire regions en-bloc rather than on
		// a per-element basis.
		std::vector<region_receive> pending_receives;
		std::vector<gather_receive> pending_gathers;

		explicit per_buffer_data(int dims, const celerity::range<3>& range, const size_t elem_size, const size_t elem_align, const size_t n_memories)
		    : dims(dims), range(range), elem_size(elem_size), elem_align(elem_align), memories(n_memories), newest_data_location(range, dims),
		      original_writers(range, dims), original_write_memories(range, dims) {}

		void apply_epoch(instruction* const epoch) {
			for(auto& memory : memories) {
				memory.apply_epoch(epoch);
			}
			original_writers.apply_to_values([epoch](instruction* const instr) -> instruction* {
				if(instr != nullptr && instr->get_id() < epoch->get_id()) {
					return epoch;
				} else {
					return instr;
				}
			});

			// This is an opportune point to verify that all await-pushes are fully consumed eventually. On epoch application,
			// original_writers[*].await_receives potentially points to instructions before the new epoch, but when compiling a horizon or epoch command, all
			// previous await-pushes should have been consumed by the task command they were generated for.
			assert(pending_receives.empty());
			assert(pending_gathers.empty());
		}
	};

	struct per_host_object_data {
		bool owns_instance;
		instruction* last_side_effect = nullptr;

		explicit per_host_object_data(const bool owns_instance, instruction* const last_epoch) : owns_instance(owns_instance), last_side_effect(last_epoch) {}

		void apply_epoch(instruction* const epoch) {
			if(last_side_effect != nullptr && last_side_effect->get_id() < epoch->get_id()) { last_side_effect = epoch; }
		}
	};

	struct per_collective_group_data {
		instruction* last_host_task = nullptr;

		void apply_epoch(instruction* const epoch) {
			if(last_host_task && last_host_task->get_id() < epoch->get_id()) { last_host_task = epoch; }
		}
	};

	struct localized_chunk {
		memory_id memory_id = host_memory_id;
		subrange<3> subrange;
	};

	inline static const box<3> scalar_reduction_box{zeros, ones};

	instruction_graph& m_idag;
	std::vector<outbound_pilot> m_pending_pilots;
	instruction_id m_next_iid = 0;
	allocation_id m_next_aid = null_allocation_id + 1;
	int m_next_p2p_tag = 10; // TODO
	const task_manager& m_tm;
	size_t m_num_nodes;
	node_id m_local_nid;
	system_info m_system;
	policy_set m_policy;
	instruction* m_last_horizon = nullptr;
	instruction* m_last_epoch = nullptr;
	// we iterate over m_execution_front, so to keep IDAG generation deterministic, its internal order must not depend on pointer values
	std::unordered_set<instruction*, instruction_hash_by_id> m_execution_front;
	std::unordered_map<buffer_id, per_buffer_data> m_buffers;
	std::unordered_map<host_object_id, per_host_object_data> m_host_objects;
	std::unordered_map<collective_group_id, per_collective_group_data> m_collective_groups;
	std::vector<const instruction*> m_current_batch; // TODO this should NOT be a member but an output parameter to compile_*()
	instruction_recorder* m_recorder;

	static memory_id next_location(const data_location& location, memory_id first);

	template <typename Instruction, typename... CtorParams>
	Instruction& create(CtorParams&&... ctor_args) {
		const auto id = m_next_iid++;
		auto instr = std::make_unique<Instruction>(id, std::forward<CtorParams>(ctor_args)...);
		const auto ptr = instr.get();
		m_idag.push_instruction(std::move(instr));
		m_execution_front.insert(ptr);
		m_current_batch.push_back(ptr);
		return *ptr;
	}

	void add_dependency(instruction& from, instruction& to, const dependency_kind kind) {
		from.add_dependency({&to, kind, dependency_origin::instruction});
		if(kind == dependency_kind::true_dep) { m_execution_front.erase(&to); }
	}

	void apply_epoch(instruction* const epoch) {
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

		// TODO prune graph. Should we re-write node dependencies?
		//	 - pro: No accidentally following stale pointers
		//   - con: Thread safety (but how would a consumer know which dependency edges can be followed)?
	}

	void collapse_execution_front_to(instruction* const horizon) {
		for(const auto instr : m_execution_front) {
			if(instr != horizon) { horizon->add_dependency({instr, dependency_kind::true_dep, dependency_origin::instruction}); }
		}
		m_execution_front.clear();
		m_execution_front.insert(horizon);
	}

	// Re-allocation of one buffer on one memory never interacts with other buffers or other memories backing the same buffer, this function can be called
	// in any order of allocation requirements without generating additional dependencies.
	void allocate_contiguously(buffer_id bid, memory_id mid, const bounding_box_set& boxes);

	void commit_pending_region_receive(
	    buffer_id bid, const per_buffer_data::region_receive& receives, const std::vector<std::pair<memory_id, region<3>>>& reads);

	// To avoid multi-hop copies, all read requirements for one buffer must be satisfied on all memories simultaneously. We deliberately allow multiple,
	// potentially-overlapping regions per memory to avoid aggregated copies introducing synchronization points between otherwise independent instructions.
	void locally_satisfy_read_requirements(buffer_id bid, const std::vector<std::pair<memory_id, region<3>>>& reads);

	void satisfy_buffer_requirements_for_regular_access(
	    buffer_id bid, const task& tsk, const subrange<3>& local_sr, const std::vector<localized_chunk>& local_chunks);

	void satisfy_buffer_requirements_as_reduction_output(buffer_id bid, reduction_id rid, const std::vector<localized_chunk>& local_chunks);

	void satisfy_buffer_requirements(
	    buffer_id bid, const task& tsk, const subrange<3>& local_sr, bool is_reduction_initializer, const std::vector<localized_chunk>& local_chunks);

	int create_pilot_message(node_id target, const transfer_id& trid, const box<3>& box);

	void compile_execution_command(const execution_command& ecmd);
	void compile_push_command(const push_command& pcmd);
	void compile_await_push_command(const await_push_command& apcmd);
	void compile_reduction_command(const reduction_command& rcmd);
	void compile_fence_command(const fence_command& fcmd);
};


// 2-connectivity for 1d boxes, 4-connectivity for 2d boxes and 6-connectivity for 3d boxes.
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

// TODO can this re-use some code from bounding_box_set?
template <int Dims>
std::pair<std::vector<region<Dims>>, box_vector<Dims>> edge_connected_subregions_with_bounding_boxes(box_vector<Dims> boxes) {
	std::vector<region<Dims>> subregions;
	box_vector<Dims> bounding_boxes;

	auto begin = boxes.begin();
	const auto end = boxes.end();
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
		subregions.push_back(region<Dims>(box_vector<Dims>(begin, connected_end)));
		bounding_boxes.push_back(connected_bounding_box);
		begin = connected_end;
	}
	return {std::move(subregions), std::move(bounding_boxes)};
}

// This is different from bounding_box_set merges, which work on box_intersections instead of boxes_edge_connected (TODO unit-test this)
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

instruction_graph_generator::impl::impl(const task_manager& tm, size_t num_nodes, node_id local_nid, system_info system, instruction_graph& idag,
    instruction_recorder* const recorder, const policy_set& policy)
    : m_idag(idag), m_tm(tm), m_num_nodes(num_nodes), m_local_nid(local_nid), m_system(std::move(system)), m_policy(policy), m_recorder(recorder) //
{
	assert(m_system.memories.size() <= max_num_memories);
	assert(std::all_of(
	    m_system.devices.begin(), m_system.devices.end(), [&](const device_info& device) { return device.native_memory < m_system.memories.size(); }));

	m_idag.begin_epoch(task_manager::initial_epoch_task);
	const auto initial_epoch = &create<epoch_instruction>(task_manager::initial_epoch_task, epoch_action::none);
	if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*initial_epoch, command_id(0 /* or so we assume */)); }
	m_last_epoch = initial_epoch;
	m_collective_groups.emplace(root_collective_group_id, per_collective_group_data{initial_epoch});
}

void instruction_graph_generator::impl::create_buffer(
    const buffer_id bid, const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const bool host_initialized) //
{
	const auto [iter, inserted] =
	    m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(dims, range, elem_size, elem_align, m_system.memories.size()));
	assert(inserted);

	if(host_initialized) {
		// eagerly allocate and fill entire host buffer. TODO this should only be done as-needed as part of satisfy_read_requirements, but eager operations
		// saves us from a) tracking user allocations (needs a separate memory_id) as well as generating chained user -> host -> device copies which we would
		// want to do to guarantee that host -> device copies are always made from pinned memory.

		auto& buffer = iter->second;
		auto& host_memory = buffer.memories.at(host_memory_id);
		const box entire_buffer = subrange({}, buffer.range);

		// this will be the first allocation for the buffer - no need to go through allocate_contiguously()
		const auto host_aid = m_next_aid++;
		auto& alloc_instr = create<alloc_instruction>(host_aid, host_memory_id, range.size() * elem_size, elem_align);
		add_dependency(alloc_instr, *m_last_epoch, dependency_kind::true_dep);

		auto& init_instr = create<init_buffer_instruction>(bid, host_aid, range.size() * elem_size);
		add_dependency(init_instr, alloc_instr, dependency_kind::true_dep);

		auto& allocation = host_memory.allocations.emplace_back(buffer.dims, host_aid, entire_buffer, buffer.range);
		allocation.record_write(entire_buffer, &init_instr);
		buffer.original_writers.update_region(entire_buffer, &init_instr);
		buffer.newest_data_location.update_region(entire_buffer, data_location().set(host_memory_id));
		buffer.original_write_memories.update_region(entire_buffer, host_memory_id);

		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(
			    alloc_instr, alloc_instruction_record::alloc_origin::buffer, buffer_allocation_record{bid, buffer.name, entire_buffer}, std::nullopt);
			*m_recorder << init_buffer_instruction_record(init_instr, buffer.name);
		}

		// we return the generated instructions with the next call to compile().
		// TODO this should probably follow a callback mechanism instead - we currently do the same for the initial epoch.
	}
}

void instruction_graph_generator::impl::set_buffer_debug_name(const buffer_id bid, const std::string& name) { m_buffers.at(bid).name = name; }

void instruction_graph_generator::impl::destroy_buffer(const buffer_id bid) {
	const auto iter = m_buffers.find(bid);
	assert(iter != m_buffers.end());
	auto& buffer = iter->second;

	for(memory_id mid = 0; mid < buffer.memories.size(); ++mid) {
		auto& memory = buffer.memories[mid];
		for(auto& allocation : memory.allocations) {
			const auto free_instr = &create<free_instruction>(mid, allocation.aid);
			for(const auto& [_, front] : allocation.access_fronts.get_region_values(allocation.box)) {
				for(const auto access_instr : front.front) {
					add_dependency(*free_instr, *access_instr, dependency_kind::true_dep);
				}
			}
			if(m_recorder != nullptr) {
				*m_recorder << free_instruction_record(
				    *free_instr, allocation.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.name, allocation.box});
			}
		}
	}

	m_buffers.erase(iter);
}

void instruction_graph_generator::impl::create_host_object(const host_object_id hoid, const bool owns_instance) {
	assert(m_host_objects.count(hoid) == 0);
	m_host_objects.emplace(hoid, per_host_object_data(owns_instance, m_last_epoch));
}

void instruction_graph_generator::impl::destroy_host_object(const host_object_id hoid) {
	const auto iter = m_host_objects.find(hoid);
	assert(iter != m_host_objects.end());

	auto& obj = iter->second;
	if(obj.owns_instance) {
		const auto destroy_instr = &create<destroy_host_object_instruction>(hoid);
		add_dependency(*destroy_instr, *obj.last_side_effect, dependency_kind::true_dep);
		if(m_recorder != nullptr) { *m_recorder << destroy_host_object_instruction_record(*destroy_instr); }
	}

	m_host_objects.erase(iter);
}

memory_id instruction_graph_generator::impl::next_location(const data_location& location, memory_id first) {
	for(size_t i = 0; i < max_num_memories; ++i) {
		const memory_id mem = (first + i) % max_num_memories;
		if(location[mem]) { return mem; }
	}
	utils::panic("data is requested to be read, but not located in any memory");
}

// TODO decide if this should only receive non-contiguous boxes (and assert that) or it should filter for non-contiguous boxes itself
void instruction_graph_generator::impl::allocate_contiguously(const buffer_id bid, const memory_id mid, const bounding_box_set& boxes) {
	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories[mid];

	bounding_box_set contiguous_after_reallocation;
	for(auto& alloc : memory.allocations) {
		contiguous_after_reallocation.insert(alloc.box);
	}
	for(auto& box : boxes) {
		contiguous_after_reallocation.insert(box);
	}

	std::vector<allocation_id> free_after_reallocation;
	for(auto& alloc : memory.allocations) {
		if(std::none_of(contiguous_after_reallocation.begin(), contiguous_after_reallocation.end(), [&](auto& box) { return alloc.box == box; })) {
			free_after_reallocation.push_back(alloc.aid);
		}
	}

	// ?? TODO what does this even do?
	auto unmerged_new_allocation = std::move(contiguous_after_reallocation).into_vector();
	const auto last_new_allocation = std::remove_if(unmerged_new_allocation.begin(), unmerged_new_allocation.end(),
	    [&](auto& box) { return std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == box; }); });
	unmerged_new_allocation.erase(last_new_allocation, unmerged_new_allocation.end());

	// region-merge adjacent boxes that need to be allocated (usually for oversubscriptions). This should not introduce problematic synchronization points since
	// ??? TODO this is an old comment but does not seem to have an implementation

	// TODO but it does introduce synchronization between producers on the resize-copies, which we want to avoid. To resolve this, allocate the fused boxes as
	// before, but use the non-fused boxes as copy destinations.
	region new_allocations(std::move(unmerged_new_allocation));

	// TODO don't copy data that will be overwritten (have an additional region<3> to_be_overwritten parameter)

	for(const auto& dest_box : new_allocations.get_boxes()) {
		auto& dest = memory.allocations.emplace_back(buffer.dims, m_next_aid++, dest_box, buffer.range);
		const auto alloc_instr = &create<alloc_instruction>(dest.aid, mid, dest.box.get_area() * buffer.elem_size, buffer.elem_align);
		add_dependency(*alloc_instr, *m_last_epoch, dependency_kind::true_dep);
		dest.record_write(dest.box, alloc_instr); // TODO figure out how to make alloc_instr the "epoch" for any subsequent reads or writes.

		for(auto& source : memory.allocations) {
			if(std::find_if(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const allocation_id aid) { return aid == source.aid; })
			    == free_after_reallocation.end()) {
				// we modify memory.allocations in-place, so we need to be careful not to attempt copying from a new allocation to itself.
				// Since we don't have overlapping allocations, any copy source must currently be one that will be freed after reallocation.
				continue;
			}

			// only copy those boxes to the new allocation that are still up-to-date in the old allocation
			// TODO investigate a garbage-collection heuristic that omits these copies if we do not expect them to be read from again on this memory
			const auto full_copy_box = box_intersection(dest.box, source.box);
			box_vector<3> live_copy_boxes;
			for(const auto& [copy_box, location] : buffer.newest_data_location.get_region_values(full_copy_box)) {
				if(location.test(mid)) { live_copy_boxes.push_back(copy_box); }
			}
			region<3> live_copy_region(std::move(live_copy_boxes));

			// TODO v--- this is duplicated in satisfy_read_requirements

			for(const auto& copy_box : live_copy_region.get_boxes()) {
				assert(!copy_box.empty());

				const auto [source_offset, source_range] = source.box.get_subrange();
				const auto [dest_offset, dest_range] = dest_box.get_subrange();
				const auto [copy_offset, copy_range] = copy_box.get_subrange();

				// TODO to avoid introducing a synchronization point on oversubscription, split into multiple copies if that will allow unimpeded
				// oversubscribed-producer to oversubscribed-consumer data flow.

				const auto copy_instr = &create<copy_instruction>(buffer.dims, mid, source.aid, source_range, copy_offset - source_offset, mid, dest.aid,
				    dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);

				for(const auto& [_, dep_instr] : source.last_writers.get_region_values(copy_box)) { // TODO copy-pasta
					assert(dep_instr != nullptr);
					add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
				}
				source.record_read(copy_box, copy_instr);

				for(const auto& [_, front] : dest.access_fronts.get_region_values(copy_box)) { // TODO copy-pasta
					for(const auto dep_instr : front.front) {
						add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
					}
				}
				dest.record_write(copy_box, copy_instr);

				if(m_recorder != nullptr) {
					*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::resize, bid, buffer.name, copy_box);
				}
			}
		}

		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(
			    *alloc_instr, alloc_instruction_record::alloc_origin::buffer, buffer_allocation_record{bid, buffer.name, dest_box}, std::nullopt);
		}
	}

	// TODO consider keeping old allocations around until their box is written to in order to resolve "buffer-locking" anti-dependencies
	for(const auto free_aid : free_after_reallocation) {
		const auto& allocation = *std::find_if(memory.allocations.begin(), memory.allocations.end(), [&](const auto& a) { return a.aid == free_aid; });
		const auto free_instr = &create<free_instruction>(mid, allocation.aid);
		for(const auto& [_, front] : allocation.access_fronts.get_region_values(allocation.box)) { // TODO copy-pasta
			for(const auto dep_instr : front.front) {
				add_dependency(*free_instr, *dep_instr, dependency_kind::true_dep);
			}
		}
		if(m_recorder != nullptr) {
			*m_recorder << free_instruction_record(
			    *free_instr, allocation.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.name, allocation.box});
		}
	}

	// TODO garbage-collect allocations that are both stale and not written to? We cannot re-fetch buffer subranges from their original producer without
	// some sort of inter-node pull semantics if the GC turned out to be a misprediction, but we can swap allocations to the host when we run out of device
	// memory. Basically we would annotate each allocation with an last-used value to implement LRU semantics.

	const auto end_retain_after_allocation = std::remove_if(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) {
		return std::any_of(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const auto aid) { return alloc.aid == aid; });
	});
	memory.allocations.erase(end_retain_after_allocation, memory.allocations.end());
}

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

void instruction_graph_generator::impl::commit_pending_region_receive(
    const buffer_id bid, const per_buffer_data::region_receive& receive, const std::vector<std::pair<memory_id, region<3>>>& reads) {
	const auto trid = transfer_id(receive.consumer_tid, bid, no_reduction_id);
	const auto mid = host_memory_id;

	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories.at(mid);

	std::vector<buffer_memory_per_allocation_data*> allocations;
	for(const auto& min_contiguous_box : receive.required_contiguous_allocations) {
		const auto alloc = memory.find_contiguous_allocation(min_contiguous_box);
		assert(alloc != nullptr); // allocated explicitly in satisfy_buffer_requirements
		if(std::find(allocations.begin(), allocations.end(), alloc) == allocations.end()) { allocations.push_back(alloc); }
	}

	for(const auto alloc : allocations) {
		const auto alloc_recv_region = region_intersection(alloc->box, receive.received_region);
		std::vector<region<3>> independent_await_regions;
		for(const auto& [_, read_region] : reads) {
			const auto await_region = region_intersection(read_region, alloc_recv_region);
			if(!await_region.empty()) { independent_await_regions.push_back(await_region); }
		}
		symmetrically_split_overlapping_regions(independent_await_regions);

		if(independent_await_regions.size() > 1) {
			const auto split_recv_instr = &create<split_receive_instruction>(trid, alloc_recv_region, mid, alloc->aid, alloc->box, buffer.elem_size);
			if(m_recorder != nullptr) { *m_recorder << split_receive_instruction_record(*split_recv_instr, buffer.name); }

			// We add dependencies to the begin_receive_instruction as if it were a writer, but update the last_writers only at the await_receive_instruction.
			// The actual write happens somewhere in-between these instructions as orchestrated by the receive_arbiter, and any other accesses need to ensure
			// that there are no pending transfers for the region they are trying to read or to access (TODO).
			for(const auto& [_, front] : alloc->access_fronts.get_region_values(alloc_recv_region)) { // TODO copy-pasta
				for(const auto dep_instr : front.front) {
					add_dependency(*split_recv_instr, *dep_instr, dependency_kind::true_dep);
				}
			}

#ifndef NDEBUG
			region<3> full_await_region;
			for(const auto& await_region : independent_await_regions) {
				full_await_region = region_union(full_await_region, await_region);
			}
			assert(full_await_region == alloc_recv_region);
#endif

			for(const auto& await_region : independent_await_regions) {
				const auto await_instr = &create<await_receive_instruction>(trid, await_region);
				if(m_recorder != nullptr) { *m_recorder << await_receive_instruction_record(*await_instr, buffer.name); }

				add_dependency(*await_instr, *split_recv_instr, dependency_kind::true_dep);

				alloc->record_write(await_region, await_instr);
				buffer.original_writers.update_region(await_region, await_instr);
			}
		} else {
			assert(independent_await_regions.size() == 1 && independent_await_regions[0] == alloc_recv_region);

			auto& recv_instr = create<receive_instruction>(trid, alloc_recv_region, mid, alloc->aid, alloc->box, buffer.elem_size);
			if(m_recorder != nullptr) { *m_recorder << receive_instruction_record(recv_instr, buffer.name); }

			for(const auto& [_, front] : alloc->access_fronts.get_region_values(alloc_recv_region)) { // TODO copy-pasta
				for(const auto dep_instr : front.front) {
					add_dependency(recv_instr, *dep_instr, dependency_kind::true_dep);
				}
			}

			alloc->record_write(alloc_recv_region, &recv_instr);
			buffer.original_writers.update_region(alloc_recv_region, &recv_instr);
		}
	}

	buffer.original_write_memories.update_region(receive.received_region, mid);
	buffer.newest_data_location.update_region(receive.received_region, data_location().set(mid));
}

void instruction_graph_generator::impl::locally_satisfy_read_requirements(const buffer_id bid, const std::vector<std::pair<memory_id, region<3>>>& reads) {
	auto& buffer = m_buffers.at(bid);

	std::unordered_map<memory_id, std::vector<region<3>>> unsatisfied_reads;
	for(const auto& [mid, read_region] : reads) {
		box_vector<3> unsatisfied_boxes;
		for(const auto& [box, location] : buffer.newest_data_location.get_region_values(read_region)) {
			if(!location.test(mid)) { unsatisfied_boxes.push_back(box); }
		}
		region<3> unsatisfied_region(std::move(unsatisfied_boxes));
		if(!unsatisfied_region.empty()) { unsatisfied_reads[mid].push_back(std::move(unsatisfied_region)); }
	}

	// transform vectors of potentially-overlapping unsatisfied regions into disjoint regions
	for(auto& [mid, regions] : unsatisfied_reads) {
		symmetrically_split_overlapping_regions(regions);
	}

	// Next, satisfy any remaining reads by copying locally from the newest data location
	struct copy_template {
		memory_id source_mid;
		memory_id dest_mid;
		region<3> region;
	};

	// TODO host copy staging if p2p is not enabled. This must be a separate pass altogether, because, in case of a broadcast from one device to multiple
	// devices, we only want to generate a single d2h followed by any number of h2ds.
	//    1. find regions that need to be copied through the host but are not yet present on the host (disjoint like `reads`)
	//    2. perform d2h copies, update access fronts / last writers
	//    3. in the actual copy loop, use the last-host-writer instead of the original writer as copy source

	std::vector<copy_template> pending_copies;
	for(auto& [dest_mid, disjoint_reader_regions] : unsatisfied_reads) {
		if(disjoint_reader_regions.empty()) continue; // if fully satisfied by incoming transfers

		auto& buffer = m_buffers.at(bid);
		for(auto& reader_region : disjoint_reader_regions) {
			if(m_policy.uninitialized_read_error != error_policy::ignore) {
				box_vector<3> uninitialized_reads;
				for(const auto& [box, sources] : buffer.newest_data_location.get_region_values(reader_region)) {
					if(!sources.any()) { uninitialized_reads.push_back(box); }
				}
				if(!uninitialized_reads.empty()) {
					// Observing an uninitialized read that is not visible in the TDAG means we have a bug.
					utils::report_error(m_policy.uninitialized_read_error,
					    // TODO print task / buffer names
					    "Instruction is trying to read B{} {}, which is neither found locally nor has been await-pushed before.", bid,
					    detail::region(std::move(uninitialized_reads)));
				}
			}

			// Split the region on original writers to enable concurrency between the write of one region and a copy on another, already written region.
			std::unordered_map<instruction_id, region<3>> writer_regions;
			for(const auto& [writer_box, original_writer] : buffer.original_writers.get_region_values(reader_region)) {
				if(original_writer == nullptr) { continue /* gracefully handle an uninitialized read */; }
				auto& region = writer_regions[original_writer->get_id()]; // allow default-insert
				region = region_union(region, writer_box);
			}

			for(const auto& [_, original_writer_region] : writer_regions) {
				// there can be multiple original-writer memories if the original writer has been subsumed by an epoch or a horizon
				std::unordered_map<memory_id, region<3>> source_memory_regions; // TODO rename
				for(const auto& [copy_box, source_mid] : buffer.original_write_memories.get_region_values(original_writer_region)) {
					auto& memory = source_memory_regions[source_mid]; // allow default-insert
					memory = region_union(memory, copy_box);
				}
				for(auto& [source_mid, copy_region] : source_memory_regions) {
					pending_copies.push_back({source_mid, dest_mid, std::move(copy_region)});
				}
			}
		}
	}

	for(auto& copy : pending_copies) {
		assert(copy.dest_mid != copy.source_mid);
		// TODO iterating over boxes here could mean that we generate excess copy instructions if region normalization produces more boxes within the
		// box_intersection below than is necessary. Instead try working on a region_intersection and resolving the matching allocation boxes afterwards.
		for(const box<3>& box : copy.region.get_boxes()) {
			for(auto& source : buffer.memories[copy.source_mid].allocations) {
				const auto read_box = box_intersection(box, source.box);
				if(read_box.empty()) continue;

				for(auto& dest : buffer.memories[copy.dest_mid].allocations) {
					const auto copy_box = box_intersection(read_box, dest.box);
					if(copy_box.empty()) continue;

					const auto [source_offset, source_range] = source.box.get_subrange();
					const auto [dest_offset, dest_range] = dest.box.get_subrange();
					const auto [copy_offset, copy_range] = copy_box.get_subrange();

					const auto copy_instr = &create<copy_instruction>(buffer.dims, copy.source_mid, source.aid, source_range, copy_offset - source_offset,
					    copy.dest_mid, dest.aid, dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);

					for(const auto& [_, last_writer_instr] : source.last_writers.get_region_values(copy_box)) {
						assert(last_writer_instr != nullptr);
						add_dependency(*copy_instr, *last_writer_instr, dependency_kind::true_dep);
					}
					source.record_read(copy_box, copy_instr);

					for(const auto& [_, front] : dest.access_fronts.get_region_values(copy_box)) { // TODO copy-pasta
						for(const auto dep_instr : front.front) {
							add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
						}
					}
					dest.record_write(copy_box, copy_instr);

					for(auto& [box, location] : buffer.newest_data_location.get_region_values(copy_box)) {
						buffer.newest_data_location.update_region(box, data_location(location).set(copy.dest_mid));
					}

					if(m_recorder != nullptr) {
						*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::coherence, bid, buffer.name, copy_box);
					}
				}
			}
		}
	}
}


void instruction_graph_generator::impl::satisfy_buffer_requirements(const buffer_id bid, const task& tsk, const subrange<3>& local_sr,
    const bool local_node_is_reduction_initializer, const std::vector<localized_chunk>& local_chunks) //
{
	assert(!local_chunks.empty());
	const auto& bam = tsk.get_buffer_access_map();

	assert(std::count_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; }) <= 1
	       && "task defines multiple reductions on the same buffer");

	// collect all receives that we must apply before execution of this command
	// Invalidate any buffer region that will immediately be overwritten (and not also read) to avoid preserving it across buffer resizes (and to catch
	// read-write access conflicts, TODO)

	auto& buffer = m_buffers.at(bid);

	std::unordered_map<memory_id, bounding_box_set> contiguous_allocations;
	std::vector<per_buffer_data::region_receive> applied_receives;

	region<3> accessed;  // which elements have are accessed (to figure out applying receives)
	region<3> discarded; // which elements are received with a non-consuming access or received (these don't need to be preserved)
	for(const auto mode : access::all_modes) {
		const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), local_sr, tsk.get_global_size());
		accessed = region_union(accessed, req);
		if(!access::mode_traits::is_consumer(mode)) { discarded = region_union(discarded, req); }
	}

	const auto reduction = std::find_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; });
	if(reduction != tsk.get_reductions().end()) {
		for(const auto& chunk : local_chunks) {
			contiguous_allocations[chunk.memory_id].insert(scalar_reduction_box);
		}
		const auto include_current_value = local_node_is_reduction_initializer && reduction->init_from_buffer;
		if(local_chunks.size() > 1 || include_current_value) {
			// we insert a host-side reduce-instruction in the multi-chunk scenario; its result will end up in the host buffer allocation
			contiguous_allocations[host_memory_id].insert(scalar_reduction_box);
		}
		if(include_current_value) {
			// scalar_reduction_box will be copied into the local-reduction gather buffer ahead of the kernel instruction
			accessed = region_union(accessed, scalar_reduction_box);
			discarded = region_difference(discarded, scalar_reduction_box);
		}
	}

	const auto first_applied_receive = std::partition(buffer.pending_receives.begin(), buffer.pending_receives.end(),
	    [&](const per_buffer_data::region_receive& r) { return region_intersection(accessed, r.received_region).empty(); });
	for(auto it = first_applied_receive; it != buffer.pending_receives.end(); ++it) {
		// we (re) allocate before receiving, but there's no need to preserve previous data at the receive location
		discarded = region_union(discarded, it->received_region);
		// begin_receive_instruction needs contiguous allocations for the bounding boxes of potentially received fragments
		contiguous_allocations[host_memory_id].insert(it->required_contiguous_allocations.begin(), it->required_contiguous_allocations.end());
	}

	if(first_applied_receive != buffer.pending_receives.end()) {
		applied_receives.insert(applied_receives.end(), first_applied_receive, buffer.pending_receives.end());
		buffer.pending_receives.erase(first_applied_receive, buffer.pending_receives.end());
	}

	if(reduction != tsk.get_reductions().end()) {
		assert(std::all_of(buffer.pending_receives.begin(), buffer.pending_receives.end(), [&](const per_buffer_data::region_receive& r) {
			return region_intersection(r.received_region, scalar_reduction_box).empty();
		}) && std::all_of(buffer.pending_gathers.begin(), buffer.pending_gathers.end(), [&](const per_buffer_data::gather_receive& r) {
			return box_intersection(r.gather_box, scalar_reduction_box).empty();
		}) && "buffer has an unprocessed await-push in a region that is going to be used as a reduction output");
	}

	// do not preserve any received or overwritten region across receives
	buffer.newest_data_location.update_region(discarded, data_location());

	for(const auto& chunk : local_chunks) {
		const auto chunk_boxes = bam.get_required_contiguous_boxes(bid, tsk.get_dimensions(), chunk.subrange, tsk.get_global_size());
		contiguous_allocations[chunk.memory_id].insert(chunk_boxes.begin(), chunk_boxes.end());
	}

	for(const auto& [mid, boxes] : contiguous_allocations) {
		allocate_contiguously(bid, mid, boxes);
	}

	std::vector<std::pair<memory_id, region<3>>> local_chunk_reads;
	for(const auto& chunk : local_chunks) {
		for(const auto mode : access::consumer_modes) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), chunk.subrange, tsk.get_global_size());
			if(!req.empty()) { local_chunk_reads.emplace_back(chunk.memory_id, req); }
		}
	}
	if(local_node_is_reduction_initializer && reduction != tsk.get_reductions().end() && reduction->init_from_buffer) {
		local_chunk_reads.emplace_back(host_memory_id, scalar_reduction_box);
	}

	for(const auto& receive : applied_receives) {
		commit_pending_region_receive(bid, receive, local_chunk_reads);
	}

	// 3) create device <-> host or device <-> device copy instructions to satisfy all command-instruction reads

	locally_satisfy_read_requirements(bid, local_chunk_reads);
}


int instruction_graph_generator::impl::create_pilot_message(const node_id target, const transfer_id& trid, const box<3>& box) {
	int tag = m_next_p2p_tag++;
	m_pending_pilots.push_back(outbound_pilot{target, pilot_message{tag, trid, box}});
	if(m_recorder != nullptr) { *m_recorder << m_pending_pilots.back(); }
	return tag;
}


void instruction_graph_generator::impl::compile_execution_command(const execution_command& ecmd) {
	const auto& tsk = *m_tm.get_task(ecmd.get_tid());
	const auto& bam = tsk.get_buffer_access_map();

	// 0) collectively generate any non-existing collective group

	if(const auto cgid = tsk.get_collective_group_id(); cgid != non_collective_group_id && m_collective_groups.count(cgid) == 0) {
		auto& root_cg = m_collective_groups.at(root_collective_group_id);
		const auto clone_cg_isntr = &create<clone_collective_group_instruction>(root_collective_group_id, tsk.get_collective_group_id());
		if(m_recorder != nullptr) { *m_recorder << clone_collective_group_instruction_record(*clone_cg_isntr); }
		add_dependency(*clone_cg_isntr, *root_cg.last_host_task, dependency_kind::true_dep);
		root_cg.last_host_task = clone_cg_isntr;
		m_collective_groups.emplace(cgid, per_collective_group_data{clone_cg_isntr});
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
		instruction* instruction = nullptr;
	};

	if(tsk.get_execution_target() == execution_target::device && m_system.devices.empty()) { utils::panic("no device on which to execute device kernel"); }

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
		// chunks" on the final write. This could be solved by creating separate reduction-output allocations for each device chunk and not touching the actual
		// buffer allocation. This is left as *future work* for a general overhaul of reductions.
		if(is_splittable_locally && tsk.get_reductions().empty()) {
			oversubscribe_factor = oversubscribe->get_factor();
		} else if(m_policy.unsafe_oversubscription_error != error_policy::ignore) {
			utils::report_error(m_policy.unsafe_oversubscription_error, "Refusing to oversubscribe{} T{}{}{}.",
			    tsk.get_execution_target() == execution_target::device ? " device kernel"
			    : tsk.get_execution_target() == execution_target::host ? " host task"
			                                                           : "",
			    tsk.get_id(), //
			    !tsk.get_debug_name().empty() ? fmt::format(" \"{}\"", tsk.get_debug_name()) : "",
			    !tsk.get_reductions().empty()                              ? " because it performs a reduction"
			    : !tsk.get_side_effect_map().empty()                       ? " because it has side effects"
			    : tsk.get_collective_group_id() != non_collective_group_id ? " because it participates in a collective group"
			    : !tsk.has_variable_split()                                ? " because its iteration space cannot be split"
			                                                               : "");
		}
	}

	std::vector<partial_instruction> cmd_instrs;
	for(size_t i = 0; i < coarse_chunks.size(); ++i) {
		for(const auto& fine_chunk : split(coarse_chunks[i], tsk.get_granularity(), oversubscribe_factor)) {
			auto& instr = cmd_instrs.emplace_back();
			instr.subrange = subrange<3>(fine_chunk.offset, fine_chunk.range);
			if(tsk.get_execution_target() == execution_target::device) {
				assert(i < m_system.devices.size());
				instr.did = device_id(i);
				instr.memory_id = m_system.devices[i].native_memory;
			} else {
				instr.memory_id = host_memory_id;
			}
		}
	}

	// detect overlapping writes

	if(m_policy.overlapping_write_error != error_policy::ignore) {
		box_vector<3> local_chunk_boxes(cmd_instrs.size(), box<3>());
		std::transform(cmd_instrs.begin(), cmd_instrs.end(), local_chunk_boxes.begin(), [](const partial_instruction& pi) { return pi.subrange; });

		if(const auto overlapping_writes = detect_overlapping_writes(tsk, local_chunk_boxes); !overlapping_writes.empty()) {
			auto error = fmt::format("Task T{}", tsk.get_id());
			if(!tsk.get_debug_name().empty()) { fmt::format_to(std::back_inserter(error), " \"{}\"", tsk.get_debug_name()); }
			fmt::format_to(std::back_inserter(error), " has overlapping writes on N{} in", m_local_nid);
			for(const auto& [bid, overlap] : overlapping_writes) {
				fmt::format_to(std::back_inserter(error), " B{} {}", bid, overlap);
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
		satisfy_buffer_requirements(
		    bid, tsk, ecmd.get_execution_range(), ecmd.is_reduction_initializer(), std::vector<localized_chunk>(cmd_instrs.begin(), cmd_instrs.end()));
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
		auto& red = local_reductions[i];
		const auto [rid, bid, reduction_task_includes_buffer_value] = tsk.get_reductions()[i];
		auto& buffer = m_buffers.at(bid);

		red.include_current_value = reduction_task_includes_buffer_value && ecmd.is_reduction_initializer();
		red.current_value_offset = red.include_current_value ? 1 : 0;
		red.num_chunks = red.current_value_offset + cmd_instrs.size();
		red.chunk_size = scalar_reduction_box.get_area() * buffer.elem_size;

		if(red.num_chunks <= 1) continue;

		red.gather_aid = m_next_aid++;
		red.gather_alloc_instr = &create<alloc_instruction>(red.gather_aid, host_memory_id, red.num_chunks * red.chunk_size, buffer.elem_align);
		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(*red.gather_alloc_instr, alloc_instruction_record::alloc_origin::gather,
			    buffer_allocation_record{bid, buffer.name, scalar_reduction_box}, red.num_chunks);
		}
		add_dependency(*red.gather_alloc_instr, *m_last_epoch, dependency_kind::true_dep);

		if(red.include_current_value) {
			auto source = buffer.memories.at(host_memory_id).find_contiguous_allocation(scalar_reduction_box); // provided by satisfy_buffer_requirements
			assert(source != nullptr);

			// copy to local gather space

			const auto current_value_copy_instr = &create<copy_instruction>(buffer.dims, host_memory_id, source->aid, source->box.get_range(),
			    scalar_reduction_box.get_offset() - source->box.get_offset(), host_memory_id, red.gather_aid, range_cast<3>(range<1>(red.num_chunks)), id<3>(),
			    scalar_reduction_box.get_range(), buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(
				    *current_value_copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.name, scalar_reduction_box);
			}
			add_dependency(*current_value_copy_instr, *red.gather_alloc_instr, dependency_kind::true_dep);
			for(const auto& [_, dep_instr] : source->last_writers.get_region_values(scalar_reduction_box)) { // TODO copy-pasta
				assert(dep_instr != nullptr);
				add_dependency(*current_value_copy_instr, *dep_instr, dependency_kind::true_dep);
			}
			source->record_read(scalar_reduction_box, current_value_copy_instr);
		}
	}

	// collect updated regions

	for(const auto bid : bam.get_accessed_buffers()) {
		for(auto& insn : cmd_instrs) {
			reads_writes rw;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.subrange, tsk.get_global_size());
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
			const auto accessed_box = bam.get_requirements_for_nth_access(i, tsk.get_dimensions(), instr.subrange, tsk.get_global_size());
			const auto& buffer = m_buffers.at(bid);
			if(!accessed_box.empty()) {
				const auto& allocations = buffer.memories[instr.memory_id].allocations;
				const auto allocation_it = std::find_if(
				    allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& alloc) { return alloc.box.covers(accessed_box); });
				assert(allocation_it != allocations.end());
				const auto& alloc = *allocation_it;
				allocation_map[i] = buffer_access_allocation{alloc.aid, alloc.box, accessed_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.name)};
				if(m_recorder != nullptr) { buffer_memory_access_map[i] = buffer_memory_record{bid, buffer.name, instr.memory_id}; }
			} else {
				allocation_map[i] = buffer_access_allocation{null_allocation_id, {}, {} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.name)};
				if(m_recorder != nullptr) { buffer_memory_access_map[i] = buffer_memory_record{bid, buffer.name, instr.memory_id}; }
			}
		}

		for(size_t i = 0; i < tsk.get_reductions().size(); ++i) {
			const auto& rinfo = tsk.get_reductions()[i];
			// TODO copy-pasted from directly above
			const auto& buffer = m_buffers.at(rinfo.bid);
			const auto& allocations = buffer.memories[instr.memory_id].allocations;
			const auto allocation_it = std::find_if(
			    allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& alloc) { return alloc.box.covers(scalar_reduction_box); });
			assert(allocation_it != allocations.end());
			const auto& alloc = *allocation_it;
			reduction_map[i] =
			    buffer_access_allocation{alloc.aid, alloc.box, scalar_reduction_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, rinfo.bid, buffer.name)};
			if(m_recorder != nullptr) { buffer_memory_reduction_map[i] = buffer_reduction_record{rinfo.bid, buffer.name, instr.memory_id, rinfo.rid}; }
		}

		if(tsk.get_execution_target() == execution_target::device) {
			assert(instr.subrange.range.size() > 0);
			assert(instr.memory_id != host_memory_id);
			// TODO how do I know it's a SYCL kernel and not a CUDA kernel?
			const auto device_kernel_instr = &create<device_kernel_instruction>(instr.did, tsk.get_launcher<device_kernel_launcher>(), instr.subrange,
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
			const auto host_task_instr = &create<host_task_instruction>(tsk.get_launcher<host_task_launcher>(), instr.subrange, tsk.get_global_size(),
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
		for(const auto& [bid, rw] : instr.rw_map) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[instr.memory_id];
			for(auto& alloc : memory.allocations) {
				const auto reads_from_alloc = region_intersection(rw.reads, alloc.box);
				for(const auto& [_, last_writer_instr] : alloc.last_writers.get_region_values(reads_from_alloc)) {
					assert(last_writer_instr != nullptr);
					add_dependency(*instr.instruction, *last_writer_instr, dependency_kind::true_dep);
				}
			}
		}
		// TODO ^--v--- these blocks are mostly identical
		for(const auto& [bid, rw] : instr.rw_map) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[instr.memory_id];
			for(auto& alloc : memory.allocations) {
				const auto writes_to_alloc = region_intersection(rw.writes, alloc.box);
				for(const auto& [_, front] : alloc.access_fronts.get_region_values(writes_to_alloc)) {
					for(const auto dep_instr : front.front) {
						add_dependency(*instr.instruction, *dep_instr, dependency_kind::true_dep);
					}
				}
			}
		}
		for(const auto& [hoid, order] : instr.se_map) {
			assert(instr.memory_id == host_memory_id);
			if(const auto last_side_effect = m_host_objects.at(hoid).last_side_effect) {
				add_dependency(*instr.instruction, *last_side_effect, dependency_kind::true_dep);
			}
		}
		if(const auto cgid = tsk.get_collective_group_id(); cgid != non_collective_group_id) {
			assert(instr.memory_id == host_memory_id);
			auto& group = m_collective_groups.at(cgid); // created previously with clone_collective_group_instruction
			add_dependency(*instr.instruction, *group.last_host_task, dependency_kind::true_dep);
		}
	}

	// 6) update data locations and last writers resulting from command instructions

	for(const auto& instr : cmd_instrs) {
		for(const auto& [bid, rw] : instr.rw_map) {
			assert(instr.instruction != nullptr);
			auto& buffer = m_buffers.at(bid);
			buffer.newest_data_location.update_region(rw.writes, data_location().set(instr.memory_id));
			buffer.original_write_memories.update_region(rw.writes, instr.memory_id);
			buffer.original_writers.update_region(rw.writes, instr.instruction);

			for(auto& alloc : buffer.memories[instr.memory_id].allocations) {
				// TODO we're calling record_read / record_write for partially overlapping regions here. Is this really the right API or should this modify
				// last_writers and access_front directly?
				for(const auto& box : rw.reads.get_boxes()) {
					const auto read_box = box_intersection(alloc.box, box);
					if(!read_box.empty()) { alloc.record_read(read_box, instr.instruction); }
				}
				for(const auto& box : rw.writes.get_boxes()) {
					const auto write_box = box_intersection(alloc.box, box);
					if(!write_box.empty()) { alloc.record_write(write_box, instr.instruction); }
				}
			}
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
		const auto [rid, bid, reduction_task_includes_buffer_value] = tsk.get_reductions()[i];
		auto& buffer = m_buffers.at(bid);
		const auto& red = local_reductions[i];

		// for a single-device configuration that doesn't include the current value, just update last writers (reduction is already complete locally)
		if(red.num_chunks == 1) {
			const auto& instr = cmd_instrs.front();
			for(auto& alloc : buffer.memories[instr.memory_id].allocations) {
				// TODO copy-pasted from rw-map updates above
				const auto write_box = box_intersection(alloc.box, scalar_reduction_box);
				if(!write_box.empty()) { alloc.record_write(write_box, instr.instruction); }
			}
			buffer.original_writers.update_box(scalar_reduction_box, instr.instruction);
			buffer.original_write_memories.update_box(scalar_reduction_box, instr.memory_id);
			buffer.newest_data_location.update_box(scalar_reduction_box, data_location().set(instr.memory_id));
			continue;
		}

		auto& host_memory = buffer.memories.at(host_memory_id);

		// gather space has been allocated before in order to preserve the current buffer value where necessary
		std::vector<copy_instruction*> gather_copy_instrs;
		gather_copy_instrs.reserve(cmd_instrs.size());
		for(size_t j = 0; j < cmd_instrs.size(); ++j) {
			const auto source_mid = cmd_instrs[j].memory_id;
			auto source = buffer.memories.at(source_mid).find_contiguous_allocation(scalar_reduction_box);
			assert(source != nullptr);

			// copy to local gather space

			const auto copy_instr = &create<copy_instruction>(std::max(1, buffer.dims), source_mid, source->aid, source->box.get_range(),
			    scalar_reduction_box.get_offset() - source->box.get_offset(), host_memory_id, red.gather_aid, range_cast<3>(range<1>(red.num_chunks)),
			    id_cast<3>(id<1>(red.current_value_offset + j)), scalar_reduction_box.get_range(), buffer.elem_size);
			if(m_recorder != nullptr) {
				*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.name, scalar_reduction_box);
			}
			add_dependency(*copy_instr, *red.gather_alloc_instr, dependency_kind::true_dep);
			for(const auto& [_, dep_instr] : source->last_writers.get_region_values(scalar_reduction_box)) { // TODO copy-pasta
				assert(dep_instr != nullptr);                                                                // TODO isn't this necessarily the cmd_instr?
				add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
			}
			source->record_read(scalar_reduction_box, copy_instr);
			gather_copy_instrs.push_back(copy_instr);
		}

		const auto dest = host_memory.find_contiguous_allocation(scalar_reduction_box);
		assert(dest != nullptr);

		// reduce

		const auto reduce_instr = &create<reduce_instruction>(rid, host_memory_id, red.gather_aid, red.num_chunks, dest->aid);
		if(m_recorder != nullptr) {
			*m_recorder << reduce_instruction_record(
			    *reduce_instr, std::nullopt, bid, buffer.name, scalar_reduction_box, reduce_instruction_record::reduction_scope::local);
		}
		for(auto& copy_instr : gather_copy_instrs) {
			add_dependency(*reduce_instr, *copy_instr, dependency_kind::true_dep);
		}
		for(const auto& [_, front] : dest->access_fronts.get_region_values(scalar_reduction_box)) {
			for(const auto access_instr : front.front) {
				add_dependency(*reduce_instr, *access_instr, dependency_kind::true_dep);
			}
		}
		dest->record_write(scalar_reduction_box, reduce_instr);
		buffer.original_writers.update_region(scalar_reduction_box, reduce_instr);
		buffer.original_write_memories.update_region(scalar_reduction_box, host_memory_id);
		buffer.newest_data_location.update_region(scalar_reduction_box, data_location().set(host_memory_id));

		// free local gather space

		const auto gather_free_instr = &create<free_instruction>(host_memory_id, red.gather_aid);
		if(m_recorder != nullptr) { *m_recorder << free_instruction_record(*gather_free_instr, red.num_chunks * red.chunk_size, std::nullopt); }
		add_dependency(*gather_free_instr, *reduce_instr, dependency_kind::true_dep);
	}

	// 7) insert epoch and horizon dependencies, apply epochs, optionally record the instruction

	for(auto& instr : cmd_instrs) {
		// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
		// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
		const auto deps = instr.instruction->get_dependencies();
		if(std::none_of(deps.begin(), deps.end(), [](const instruction::dependency& dep) { return dep.kind == dependency_kind::true_dep; })) {
			add_dependency(*instr.instruction, *m_last_epoch, dependency_kind::true_dep);
		}
	}
}


void instruction_graph_generator::impl::compile_push_command(const push_command& pcmd) {
	const auto trid = pcmd.get_transfer_id();

	auto& buffer = m_buffers.at(trid.bid);
	const auto push_box = box(pcmd.get_range());

	// We want to generate the fewest number of send instructions possible without introducing new synchronization points between chunks of the same
	// command that generated the pushed data. This will allow compute-transfer overlap, especially in the case of oversubscribed splits.
	std::unordered_map<instruction_id, region<3>> writer_regions;
	for(auto& [box, writer] : buffer.original_writers.get_region_values(push_box)) {
		auto& region = writer_regions[writer->get_id()]; // allow default-insert
		region = region_union(region, box);
	}

	for(auto& [_, region] : writer_regions) {
		// Since the original writer is unique for each buffer item, all writer_regions will be disjoint and we can pull data from device memories for each
		// writer without any effect from the order of operations
		allocate_contiguously(trid.bid, host_memory_id, bounding_box_set(region.get_boxes()));
		locally_satisfy_read_requirements(trid.bid, {{host_memory_id, region}});
	}

	for(auto& [_, region] : writer_regions) {
		for(const auto& box : region.get_boxes()) {
			const int tag = create_pilot_message(pcmd.get_target(), trid, box);

			const auto allocation = buffer.memories.at(host_memory_id).find_contiguous_allocation(box);
			assert(allocation != nullptr); // we allocate_contiguously above

			const auto offset_in_allocation = box.get_offset() - allocation->box.get_offset();
			const auto send_instr = &create<send_instruction>(
			    pcmd.get_target(), tag, host_memory_id, allocation->aid, allocation->box.get_range(), offset_in_allocation, box.get_range(), buffer.elem_size);

			if(m_recorder != nullptr) {
				const auto offset_in_buffer = box.get_offset();
				*m_recorder << send_instruction_record(*send_instr, pcmd.get_cid(), trid, buffer.name, offset_in_buffer);
			}

			for(const auto& [_, dep_instr] : allocation->last_writers.get_region_values(box)) { // TODO copy-pasta
				assert(dep_instr != nullptr);
				add_dependency(*send_instr, *dep_instr, dependency_kind::true_dep);
			}
			allocation->record_read(box, send_instr);
		}
	}

	// If not all nodes contribute partial results to a global reductions, the remaining ones need to notify their peers that they should not expect any data.
	// This is done by announcing an empty box through the pilot message.
	assert(push_box.empty() == writer_regions.empty());
	if(writer_regions.empty()) {
		assert(trid.rid != no_reduction_id);
		create_pilot_message(pcmd.get_target(), trid, box<3>());
	}
}


void instruction_graph_generator::impl::compile_await_push_command(const await_push_command& apcmd) {
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
		buffer.pending_receives.emplace_back(trid.consumer_tid, apcmd.get_region(), connected_subregion_bounding_boxes(apcmd.get_region()));
	} else {
		assert(apcmd.get_region().get_boxes().size() == 1);
		buffer.pending_gathers.emplace_back(trid.consumer_tid, trid.rid, apcmd.get_region().get_boxes().front());
	}
}


void instruction_graph_generator::impl::compile_reduction_command(const reduction_command& rcmd) {
	const auto scalar_reduction_box = box<3>({0, 0, 0}, {1, 1, 1});
	const auto [rid, bid, init_from_buffer] = rcmd.get_reduction_info();

	auto& buffer = m_buffers.at(bid);

	assert(buffer.pending_gathers.size() == 1 && "received reduction command that is not preceded by an appropriate await-push");
	const auto& gather = buffer.pending_gathers.front();
	assert(gather.gather_box == scalar_reduction_box);

	// allocate the gather space

	const auto gather_aid = m_next_aid++;
	const auto node_chunk_size = gather.gather_box.get_area() * buffer.elem_size;
	const auto gather_alloc_instr = &create<alloc_instruction>(gather_aid, host_memory_id, m_num_nodes * node_chunk_size, buffer.elem_align);
	if(m_recorder != nullptr) {
		*m_recorder << alloc_instruction_record(
		    *gather_alloc_instr, alloc_instruction_record::alloc_origin::gather, buffer_allocation_record{bid, buffer.name, gather.gather_box}, m_num_nodes);
	}
	add_dependency(*gather_alloc_instr, *m_last_epoch, dependency_kind::true_dep);

	// fill the gather space with the reduction identity, so that the gather_receive_command can simply ignore empty boxes sent by peers that do not contribute
	// to the reduction, and we can skip the gather-copy instruction if we ourselves do not contribute a partial result.

	const auto fill_identity_instr = &create<fill_identity_instruction>(rid, host_memory_id, gather_aid, m_num_nodes);
	if(m_recorder != nullptr) { *m_recorder << fill_identity_instruction_record(*fill_identity_instr); }
	add_dependency(*fill_identity_instr, *gather_alloc_instr, dependency_kind::true_dep);

	// if the local node contributes to the reduction, copy the contribution to the appropriate position in the gather space

	copy_instruction* local_gather_copy_instr = nullptr;
	const auto contribution_location = buffer.newest_data_location.get_region_values(scalar_reduction_box).front().second;
	if(contribution_location.any()) {
		const auto source_mid = next_location(contribution_location, host_memory_id);
		const auto source_alloc = buffer.memories.at(source_mid).find_contiguous_allocation(scalar_reduction_box);
		assert(source_alloc != nullptr); // if scalar_box is up to date in that memory, it (the single element) must also be contiguous

		local_gather_copy_instr = &create<copy_instruction>(std::max(1, buffer.dims), source_mid, source_alloc->aid, source_alloc->box.get_range(),
		    scalar_reduction_box.get_offset() - source_alloc->box.get_offset(), host_memory_id, gather_aid, range_cast<3>(range<1>(m_num_nodes)),
		    id_cast<3>(id<1>(m_local_nid)), scalar_reduction_box.get_range(), buffer.elem_size);
		if(m_recorder != nullptr) {
			*m_recorder << copy_instruction_record(
			    *local_gather_copy_instr, copy_instruction_record::copy_origin::gather, bid, buffer.name, scalar_reduction_box);
		}
		add_dependency(*local_gather_copy_instr, *fill_identity_instr, dependency_kind::true_dep);
		for(const auto& [_, dep_instr] : source_alloc->last_writers.get_region_values(scalar_reduction_box)) { // TODO copy-pasta
			assert(dep_instr != nullptr);
			add_dependency(*local_gather_copy_instr, *dep_instr, dependency_kind::true_dep);
		}
		source_alloc->record_read(scalar_reduction_box, local_gather_copy_instr);
	}

	// gather remote contributions

	const transfer_id trid(gather.consumer_tid, bid, gather.rid);
	const auto gather_instr = &create<gather_receive_instruction>(trid, host_memory_id, gather_aid, node_chunk_size);
	if(m_recorder != nullptr) { *m_recorder << gather_receive_instruction_record(*gather_instr, buffer.name, gather.gather_box, m_num_nodes); }
	add_dependency(*gather_instr, *fill_identity_instr, dependency_kind::true_dep);

	// perform the global reduction

	allocate_contiguously(bid, host_memory_id, bounding_box_set({scalar_reduction_box}));

	auto& host_memory = buffer.memories.at(host_memory_id);
	auto dest_alloc = host_memory.find_contiguous_allocation(scalar_reduction_box);
	assert(dest_alloc != nullptr);

	const auto reduce_instr = &create<reduce_instruction>(rid, host_memory_id, gather_aid, m_num_nodes, dest_alloc->aid);
	if(m_recorder != nullptr) {
		*m_recorder << reduce_instruction_record(
		    *reduce_instr, rcmd.get_cid(), bid, buffer.name, scalar_reduction_box, reduce_instruction_record::reduction_scope::global);
	}
	add_dependency(*reduce_instr, *gather_instr, dependency_kind::true_dep);
	if(local_gather_copy_instr != nullptr) { add_dependency(*reduce_instr, *local_gather_copy_instr, dependency_kind::true_dep); }
	for(const auto& [_, front] : dest_alloc->access_fronts.get_region_values(scalar_reduction_box)) {
		for(const auto access_instr : front.front) {
			add_dependency(*reduce_instr, *access_instr, dependency_kind::true_dep);
		}
	}
	dest_alloc->record_write(scalar_reduction_box, reduce_instr);
	buffer.original_writers.update_region(scalar_reduction_box, reduce_instr);
	buffer.original_write_memories.update_region(scalar_reduction_box, host_memory_id);
	buffer.newest_data_location.update_region(scalar_reduction_box, data_location().set(host_memory_id));

	// free the gather space

	const auto gather_free_instr = &create<free_instruction>(host_memory_id, gather_aid);
	if(m_recorder != nullptr) { *m_recorder << free_instruction_record(*gather_free_instr, m_num_nodes * node_chunk_size, std::nullopt); }
	add_dependency(*gather_free_instr, *reduce_instr, dependency_kind::true_dep);

	buffer.pending_gathers.clear();
}


void instruction_graph_generator::impl::compile_fence_command(const fence_command& fcmd) {
	const auto& tsk = *m_tm.get_task(fcmd.get_tid());

	const auto& bam = tsk.get_buffer_access_map();
	const auto& sem = tsk.get_side_effect_map();
	assert(bam.get_num_accesses() + sem.size() == 1);

	for(const auto bid : bam.get_accessed_buffers()) {
		// fences encode their buffer requirements through buffer_access_map with a fixed range mapper (this is rather ugly)
		const subrange<3> local_sr{};
		const std::vector chunks{localized_chunk{host_memory_id, local_sr}};
		assert(tsk.get_reductions().empty()); // it doesn't matter what we pass to is_reduction_initializer next
		satisfy_buffer_requirements(bid, tsk, local_sr, false /* is_reduction_initializer */, chunks);

		const auto region = bam.get_mode_requirements(bid, access_mode::read, 0, {}, zeros);
		assert(region.get_boxes().size() == 1);
		const auto box = region.get_boxes().front();
		// TODO explicitly verify support for empty-range buffer fences

		auto& buffer = m_buffers.at(bid);
		const auto allocation = buffer.memories.at(host_memory_id).find_contiguous_allocation(box);
		assert(allocation != nullptr);

		// TODO this should become copy_instruction as soon as IGGEN supports user_memory_id
		const auto export_instr = &create<export_instruction>(allocation->aid, buffer.dims, allocation->box.get_range(),
		    box.get_offset() - allocation->box.get_offset(), box.get_range(), buffer.elem_size, tsk.get_fence_promise()->get_snapshot_pointer());
		for(const auto& [_, dep_instr] : allocation->last_writers.get_region_values(box)) { // TODO copy-pasta
			assert(dep_instr != nullptr);
			add_dependency(*export_instr, *dep_instr, dependency_kind::true_dep);
		}
		allocation->record_read(box, export_instr);

		const auto fence_instr = &create<fence_instruction>(tsk.get_fence_promise());
		add_dependency(*fence_instr, *export_instr, dependency_kind::true_dep);

		if(m_recorder != nullptr) {
			*m_recorder << export_instruction_record(*export_instr, bid, buffer.name, box.get_offset());
			*m_recorder << fence_instruction_record(*fence_instr, tsk.get_id(), fcmd.get_cid(), bid, buffer.name, box.get_subrange());
		}
	}

	for(const auto [hoid, _] : sem) {
		auto& obj = m_host_objects.at(hoid);
		const auto fence_instr = &create<fence_instruction>(tsk.get_fence_promise());
		add_dependency(*fence_instr, *obj.last_side_effect, dependency_kind::true_dep);
		obj.last_side_effect = fence_instr;

		if(m_recorder != nullptr) { *m_recorder << fence_instruction_record(*fence_instr, tsk.get_id(), fcmd.get_cid(), hoid); }
	}
}


template <typename Iterator>
bool is_topologically_sorted(Iterator begin, Iterator end) {
	for(auto check = begin; check != end; ++check) {
		for(const auto& dep : (*check)->get_dependencies()) {
			if(std::find(std::next(check), end, dep.node) != end) return false;
		}
	}
	return true;
}

std::pair<std::vector<const instruction*>, std::vector<outbound_pilot>> instruction_graph_generator::impl::compile(const abstract_command& cmd) {
	matchbox::match(
	    cmd,                                                                         //
	    [&](const execution_command& ecmd) { compile_execution_command(ecmd); },     //
	    [&](const push_command& pcmd) { compile_push_command(pcmd); },               //
	    [&](const await_push_command& apcmd) { compile_await_push_command(apcmd); }, //
	    [&](const horizon_command& hcmd) {
		    m_idag.begin_epoch(hcmd.get_tid());
		    const auto horizon = &create<horizon_instruction>(hcmd.get_tid());
		    collapse_execution_front_to(horizon);
		    if(m_last_horizon != nullptr) { apply_epoch(m_last_horizon); }
		    m_last_horizon = horizon;
		    if(m_recorder != nullptr) { *m_recorder << horizon_instruction_record(*horizon, hcmd.get_cid()); }
	    },
	    [&](const epoch_command& ecmd) {
		    m_idag.begin_epoch(ecmd.get_tid());
		    const auto epoch = &create<epoch_instruction>(ecmd.get_tid(), ecmd.get_epoch_action());
		    collapse_execution_front_to(epoch);
		    apply_epoch(epoch);
		    m_last_horizon = nullptr;
		    if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*epoch, ecmd.get_cid()); }
	    },
	    [&](const reduction_command& rcmd) { compile_reduction_command(rcmd); }, //
	    [&](const fence_command& fcmd) { compile_fence_command(fcmd); }          //
	);

	if(m_recorder != nullptr) {
		for(const auto instr : m_current_batch) {
			m_recorder->record_dependencies(*instr);
		}
	}

	assert(is_topologically_sorted(m_current_batch.begin(), m_current_batch.end()));
	auto result = std::pair{std::move(m_current_batch), std::move(m_pending_pilots)};
	m_current_batch.clear();
	m_pending_pilots.clear();
	return result;
}

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, const size_t num_nodes, const node_id local_nid, system_info system,
    instruction_graph& idag, instruction_recorder* const recorder, const policy_set& policy)
    : m_impl(new impl(tm, num_nodes, local_nid, std::move(system), idag, recorder, policy)) {}

instruction_graph_generator::~instruction_graph_generator() = default;

void instruction_graph_generator::create_buffer(
    const buffer_id bid, const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const bool host_initialized) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("create buffer B{}", bid);
	m_impl->create_buffer(bid, dims, range, elem_size, elem_align, host_initialized);
}

void instruction_graph_generator::set_buffer_debug_name(const buffer_id bid, const std::string& name) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("set buffer name B{} \"{}\"", bid, name);
	m_impl->set_buffer_debug_name(bid, name);
}

void instruction_graph_generator::destroy_buffer(const buffer_id bid) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("destroy buffer B{}", bid);
	m_impl->destroy_buffer(bid);
}

void instruction_graph_generator::create_host_object(const host_object_id hoid, const bool owns_instance) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("create host object H{}", hoid);
	m_impl->create_host_object(hoid, owns_instance);
}

void instruction_graph_generator::destroy_host_object(const host_object_id hoid) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("destroy host object H{}", hoid);
	m_impl->destroy_host_object(hoid);
}

// Resulting instructions are in topological order of dependencies (i.e. sequential execution would fulfill all internal dependencies)
std::pair<std::vector<const instruction*>, std::vector<outbound_pilot>> instruction_graph_generator::compile(const abstract_command& cmd) {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE(NavyBlue, "IDAG");
	CELERITY_DETAIL_TRACY_ZONE_TEXT("compile C{}", cmd.get_cid());
	return m_impl->compile(cmd);
}

} // namespace celerity::detail
