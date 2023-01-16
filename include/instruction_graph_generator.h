#pragma once

#include "instruction_graph.h"
#include "region_map.h"

#include <bitset>
#include <unordered_set>

namespace celerity::detail {

class abstract_command;
class task_manager;

class instruction_graph_generator {
  public:
	explicit instruction_graph_generator(const task_manager& tm, size_t num_devices);

	void register_buffer(buffer_id bid, cl::sycl::range<3> range);

	void unregister_buffer(buffer_id bid);

	void register_host_object(host_object_id hoid);

	void unregister_host_object(host_object_id hoid);

	void compile(const abstract_command& cmd);

	const instruction_graph& get_graph() const { return m_idag; }

  private:
	static constexpr size_t max_memories = 32; // The maximum number of distinct memories (RAM, GPU RAM) supported by the buffer manager
	using data_location = std::bitset<max_memories>;

	struct per_buffer_data {
		struct per_memory_data {
			struct access_front {
				gch::small_vector<instruction*> front; // sorted by id to allow equality comparison
				enum { read, write } mode = write;

				friend bool operator==(const access_front& lhs, const access_front& rhs) { return lhs.front == rhs.front && lhs.mode == rhs.mode; }
				friend bool operator!=(const access_front& lhs, const access_front& rhs) { return !(lhs == rhs); }
			};

			GridRegion<3> allocation;
			region_map<instruction*> last_writers;
			region_map<access_front> access_fronts;

			explicit per_memory_data(const range<3>& range) : last_writers(range), access_fronts(range) {}

			void record_allocation(const GridRegion<3>& region, instruction* const insn) {
				allocation = GridRegion<3>::merge(allocation, region);
				// TODO this will generate antidependencies, but semantically, we want true dependencies.
				access_fronts.update_region(region, access_front{{insn}, access_front::write});
			}

			void record_read(const GridRegion<3>& region, instruction* const insn) {
				for(auto& [box, record] : access_fronts.get_region_values(region)) {
					if(record.mode == access_front::read) {
						// sorted insert
						const auto at = std::lower_bound(record.front.begin(), record.front.end(), insn, instruction_id_less());
						assert(at == record.front.end() || *at != insn);
						record.front.insert(at, insn);
					} else {
						record.front = {insn};
					}
					assert(std::is_sorted(record.front.begin(), record.front.end(), instruction_id_less()));
					access_fronts.update_region(region, std::move(record));
				}
			}

			void record_write(const GridRegion<3>& region, instruction* const insn) {
				last_writers.update_region(region, insn);
				access_fronts.update_region(region, access_front{{insn}, access_front::write});
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

		range<3> range;
		std::vector<per_memory_data> memories;
		region_map<data_location> newest_data_location;

		explicit per_buffer_data(const celerity::range<3>& range, const size_t n_memories) : range(range), newest_data_location(range) {
			memories.reserve(n_memories);
			for(size_t i = 0; i < n_memories; ++i) {
				memories.emplace_back(range);
			}
		}
	};

	struct per_memory_data {
		instruction* last_horizon = nullptr;
		instruction* last_epoch = nullptr;
		std::unordered_set<instruction*> execution_front;

		void collapse_execution_front_to(instruction* const horizon) {
			for(const auto insn : execution_front) {
				if(insn != horizon) { horizon->add_dependency({insn, dependency_kind::true_dep, dependency_origin::execution_front}); }
			}
			execution_front.clear();
			execution_front.insert(horizon);
		}
	};

	struct host_memory_data {
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

		std::unordered_map<host_object_id, per_host_object_data> host_objects;
		std::unordered_map<collective_group_id, per_collective_group_data> collective_groups;

		void apply_epoch(instruction* const epoch) {
			for(auto& [_, host_object] : host_objects) {
				host_object.apply_epoch(epoch);
			}
			for(auto& [_, collective_group] : collective_groups) {
				collective_group.apply_epoch(epoch);
			}
		}
	};

	instruction_graph m_idag;
	instruction_id m_next_iid = 0;
	const task_manager& m_tm;
	size_t m_num_devices;
	std::vector<per_memory_data> m_memories;
	std::unordered_map<buffer_id, per_buffer_data> m_buffers;
	host_memory_data m_host_memory;

	static memory_id next_location(const data_location& location, memory_id first);

	template <typename Instruction, typename... CtorParams>
	Instruction& create(CtorParams&&... ctor_args) {
		const auto id = m_next_iid++;
		auto insn = std::make_unique<Instruction>(id, std::forward<CtorParams>(ctor_args)...);
		const auto ptr = insn.get();
		m_idag.insert(std::move(insn));
		m_memories[ptr->get_memory_id()].execution_front.insert(ptr);
		return *ptr;
	}

	std::pair<copy_instruction&, copy_instruction&> create_copy(
	    const memory_id source_mid, const memory_id dest_mid, const buffer_id bid, GridRegion<3> region) {
		const auto source_id = m_next_iid++;
		const auto dest_id = m_next_iid++;
		auto [source, dest] = copy_instruction::make_pair(source_id, source_mid, dest_id, dest_mid, bid, std::move(region));
		const auto source_ptr = source.get(), dest_ptr = dest.get();
		m_idag.insert(std::move(source));
		m_idag.insert(std::move(dest));
		m_memories[source_ptr->get_memory_id()].execution_front.insert(source_ptr);
		m_memories[dest_ptr->get_memory_id()].execution_front.insert(dest_ptr);
		return {*source_ptr, *dest_ptr};
	}

	void add_dependency(instruction& from, instruction& to, const dependency_kind kind, const dependency_origin origin) {
		from.add_dependency({&to, kind, origin});
		if(kind == dependency_kind::true_dep) { m_memories[to.get_memory_id()].execution_front.erase(&to); }
	}
};

} // namespace celerity::detail
