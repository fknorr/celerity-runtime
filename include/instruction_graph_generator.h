#pragma once

#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace celerity::detail {

class abstract_command;
class task_manager;
class instruction_recorder;

class instruction_graph_generator {
  public:
	struct device_info {
		memory_id native_memory;
		// - which backends / features are supported?
	};

	// TODO should take unordered_map<device_id, device_info> (runtime is responsible for device id allocation, not IGGEN)
	explicit instruction_graph_generator(const task_manager& tm, std::vector<device_info> devices, instruction_recorder* recorder);

	void create_buffer(buffer_id bid, int dims, range<3> range, size_t elem_size, size_t elem_align, bool host_initialized);

	void set_buffer_debug_name(buffer_id bid, const std::string& name);

	void destroy_buffer(buffer_id bid);

	void create_host_object(host_object_id hoid, bool owns_instance);

	void destroy_host_object(host_object_id hoid);

	// Resulting instructions are in topological order of dependencies (i.e. sequential execution would fulfill all internal dependencies)
	std::pair<std::vector<const instruction*>, std::vector<outbound_pilot>> compile(const abstract_command& cmd);

	const instruction_graph& get_graph() const { return m_idag; }

  private:
	static constexpr size_t max_memories = 32; // The maximum number of distinct memories (RAM, GPU RAM) supported by the buffer manager
	using data_location = std::bitset<max_memories>;

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
					record.front = {instr};
				}
				assert(std::is_sorted(record.front.begin(), record.front.end(), instruction_id_less()));
				access_fronts.update_region(region, std::move(record));
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
		struct partial_inbound_transfer {
			transfer_id id;
			region<3> unconsumed_region;
			begin_receive_instruction *begin_receive;
			std::vector<await_receive_instruction*> await_receives;

			partial_inbound_transfer(const transfer_id id, region<3> region, begin_receive_instruction* const begin_receive)
			    : id(id), unconsumed_region(std::move(region)), begin_receive(begin_receive) {}
		};

		int dims;
		range<3> range;
		size_t elem_size;
		size_t elem_align;
		std::vector<buffer_per_memory_data> memories;
		region_map<data_location> newest_data_location;
		region_map<instruction*> original_writers;

		/// Since the CDAG only contains a single await-push per task and buffer, we track partial receives in a vector instead of a region_map - this allows us
		/// to easily find transfers with an empty unconsumed region for which we then emit end_receive_instruction.
		std::vector<partial_inbound_transfer> inbound_transfers;

		explicit per_buffer_data(int dims, const celerity::range<3>& range, const size_t elem_size, const size_t elem_align, const size_t n_memories)
		    : dims(dims), range(range), elem_size(elem_size), elem_align(elem_align), memories(n_memories), newest_data_location(range, dims),
		      original_writers(range, dims) {}

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
			assert(inbound_transfers.empty());
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

	instruction_graph m_idag;
	std::vector<outbound_pilot> m_pending_pilots;
	instruction_id m_next_iid = 0;
	allocation_id m_next_aid = 0;
	int m_next_p2p_tag = 10; // TODO
	const task_manager& m_tm;
	std::vector<device_info> m_devices;
	instruction* m_last_horizon = nullptr;
	instruction* m_last_epoch = nullptr;
	std::unordered_set<instruction*> m_execution_front;
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
		m_idag.insert(std::move(instr));
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
	void allocate_contiguously(const buffer_id bid, const memory_id mid, const box_vector<3>& boxes);

	// To avoid multi-hop copies, all read requirements for one buffer must be satisfied on all memories simultaneously. We deliberately allow multiple,
	// potentially-overlapping regions per memory to avoid aggregated copies introducing synchronization points between otherwise independent instructions.
	void satisfy_read_requirements(const buffer_id bid, const std::vector<std::pair<memory_id, region<3>>>& reads);

	std::vector<copy_instruction*> linearize_buffer_subrange(const buffer_id, const box<3>& box, const memory_id out_mid, alloc_instruction& ainstr);

	int create_pilot_message(node_id target, buffer_id bid, transfer_id trid, const box<3>& box);

	void compile_execution_command(const execution_command& ecmd);

	void compile_push_command(const push_command& pcmd);

	void compile_await_push_command(const await_push_command& apcmd);

	void compile_fence_command(const fence_command& fcmd);
};

} // namespace celerity::detail
