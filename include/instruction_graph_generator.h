#pragma once

#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

#include <bitset>
#include <unordered_set>

namespace celerity::detail {

class abstract_command;
class task_manager;

class instruction_graph_generator {
  public:
	explicit instruction_graph_generator(const task_manager& tm, size_t num_devices);

	void register_buffer(buffer_id bid, int dims, range<3> range, size_t elem_size, size_t elem_align);

	void unregister_buffer(buffer_id bid);

	void register_host_object(host_object_id hoid);

	void unregister_host_object(host_object_id hoid);

	void compile(const abstract_command& cmd);

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
			GridBox<3> box;

			friend bool operator==(const allocated_box& lhs, const allocated_box& rhs) { return lhs.aid == rhs.aid && lhs.box == rhs.box; }
			friend bool operator!=(const allocated_box& lhs, const allocated_box& rhs) { return !(lhs == rhs); }
		};

		allocation_id aid;
		GridBox<3> box;
		region_map<instruction*> last_writers;  // in virtual-buffer coordinates
		region_map<access_front> access_fronts; // in virtual-buffer coordinates

		explicit buffer_memory_per_allocation_data(const allocation_id aid, const GridBox<3>& allocated_box, const range<3>& buffer_range)
		    : aid(aid), box(allocated_box), last_writers(buffer_range), access_fronts(buffer_range) {}

		void record_read(const GridRegion<3>& region, instruction* const instr) {
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

		void record_write(const GridRegion<3>& region, instruction* const instr) {
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

		bool is_allocated_contiguously(const GridBox<3>& box) const {
			return std::any_of(allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& a) { return a.box.covers(box); });
		}

		void apply_epoch(instruction* const epoch) {
			for(auto& alloc : allocations) {
				alloc.apply_epoch(epoch);
			}
		}
	};

	struct per_buffer_data {
		int dims;
		range<3> range;
		size_t elem_size;
		size_t elem_align;
		std::vector<buffer_per_memory_data> memories;
		region_map<data_location> newest_data_location;
		region_map<instruction*> original_writers;
		region_map<transfer_id> pending_await_pushes;

		explicit per_buffer_data(int dims, const celerity::range<3>& range, const size_t elem_size, const size_t elem_align, const size_t n_memories)
		    : dims(dims), range(range), elem_size(elem_size), elem_align(elem_align), memories(n_memories), newest_data_location(range),
		      original_writers(range), pending_await_pushes(range) {}

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
		}
	};

	struct per_host_object_data {
		instruction* last_side_effect = nullptr;

		void apply_epoch(instruction* const epoch) {
			if(last_side_effect && last_side_effect->get_id() < epoch->get_id()) { last_side_effect = epoch; }
		}
	};

	struct per_collective_group_data {
		instruction* last_host_task = nullptr;

		void apply_epoch(instruction* const epoch) {
			if(last_host_task && last_host_task->get_id() < epoch->get_id()) { last_host_task = epoch; }
		}
	};

	instruction_graph m_idag;
	instruction_id m_next_iid = 0;
	allocation_id m_next_aid = 0;
	const task_manager& m_tm;
	size_t m_num_devices;
	instruction* m_last_horizon = nullptr;
	instruction* m_last_epoch = nullptr;
	std::unordered_set<instruction*> m_execution_front;
	std::unordered_map<buffer_id, per_buffer_data> m_buffers;
	std::unordered_map<host_object_id, per_host_object_data> m_host_objects;
	std::unordered_map<collective_group_id, per_collective_group_data> m_collective_groups;

	static memory_id next_location(const data_location& location, memory_id first);

	template <typename Instruction, typename... CtorParams>
	Instruction& create(CtorParams&&... ctor_args) {
		const auto id = m_next_iid++;
		auto instr = std::make_unique<Instruction>(id, std::forward<CtorParams>(ctor_args)...);
		const auto ptr = instr.get();
		m_idag.insert(std::move(instr));
		m_execution_front.insert(ptr);
		return *ptr;
	}

	void add_dependency(instruction& from, instruction& to, const dependency_kind kind, const dependency_origin origin) {
		from.add_dependency({&to, kind, origin});
		if(kind == dependency_kind::true_dep) { m_execution_front.erase(&to); }
	}

	// This mapping will differ for architectures that share memory between host and device or between devices.
	// TODO we want a class like detail::local_devices to do the conversion, but without the runtime dependency (i.e. host_queue).
	memory_id device_to_memory_id(const device_id did) const { return did + 1; }

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
			if(instr != horizon) { horizon->add_dependency({instr, dependency_kind::true_dep, dependency_origin::execution_front}); }
		}
		m_execution_front.clear();
		m_execution_front.insert(horizon);
	}

	// Re-allocation of one buffer on one memory never interacts with other buffers or other memories backing the same buffer, this function can be called
	// in any order of allocation requirements without generating additional dependencies.
	void allocate_contiguously(const buffer_id bid, const memory_id mid, const std::vector<GridBox<3>>& boxes);

	// To avoid multi-hop copies, all read requirements for one buffer must be satisfied on all memories simultaneously. We deliberately allow multiple,
	// potentially-overlapping regions per memory to avoid aggregated copies introducing synchronization points between otherwise independent instructions.
	void satisfy_read_requirements(const buffer_id bid, const std::vector<std::pair<memory_id, GridRegion<3>>>& reads);

	std::vector<copy_instruction*> linearize_buffer_subrange(const buffer_id, const GridBox<3>& box, const memory_id out_mid, const allocation_id out_aid);

	void compile_execution_command(const execution_command& ecmd);

	void compile_push_command(const push_command& pcmd);
};

} // namespace celerity::detail