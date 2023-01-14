#pragma once

#include "instruction_graph.h"
#include "region_map.h"

#include <bitset>

namespace celerity::detail {

class abstract_command;
class task_manager;

class instruction_graph_generator {
  public:
	explicit instruction_graph_generator(const task_manager& tm, size_t num_devices);

	void register_buffer(buffer_id bid, cl::sycl::range<3> range);

	void unregister_buffer(buffer_id bid);

	void compile(const abstract_command& cmd);

	const instruction_graph& get_graph() const { return m_idag; }

  private:
	static constexpr size_t max_memories = 32; // The maximum number of distinct memories (RAM, GPU RAM) supported by the buffer manager
	using data_location = std::bitset<max_memories>;

	struct per_buffer_data {
		struct per_memory_data {
			struct access_front {
				gch::small_vector<instruction*> front; // sorted by pointer value to allow equality comparison
				enum { read, write } mode;

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
						const auto at = std::lower_bound(record.front.begin(), record.front.end(), insn);
						assert(at == record.front.end() || *at != insn);
						record.front.insert(at, insn);
					} else {
						record.front = {insn};
					}
					access_fronts.update_region(region, std::move(record));
				}
			}

			void record_write(const GridRegion<3>& region, instruction* const insn) {
				last_writers.update_region(region, insn);
				access_fronts.update_region(region, access_front{{insn}, access_front::write});
			}
		};

		std::vector<per_memory_data> memories;
		region_map<data_location> newest_data_location;

		explicit per_buffer_data(const range<3>& range, const size_t n_memories) : newest_data_location(range) {
			memories.reserve(n_memories);
			for(size_t i = 0; i < n_memories; ++i) {
				memories.emplace_back(range);
			}
		}
	};

	struct per_memory_data {
		instruction* epoch = nullptr;
	};

	instruction_graph m_idag;
	const task_manager& m_tm;
	size_t m_num_devices;
	std::vector<per_memory_data> m_memories;
	std::unordered_map<buffer_id, per_buffer_data> m_buffers;

	static memory_id next_location(const data_location& location, memory_id first);
};

} // namespace celerity::detail
