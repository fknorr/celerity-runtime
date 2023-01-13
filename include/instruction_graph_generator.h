#pragma once

#include "instruction_graph.h"
#include "region_map.h"

#include <bitset>

namespace celerity::detail {

class abstract_command;
class task_ring_buffer;


class instruction_graph_generator {
  public:
	explicit instruction_graph_generator(const task_ring_buffer& task_buffer, size_t num_devices);

	void register_buffer(buffer_id bid, cl::sycl::range<3> range);

	void unregister_buffer(buffer_id bid);

	void compile(const abstract_command& cmd);

	const instruction_graph& get_graph() const { return m_idag; }

  private:
	static constexpr size_t max_memories = 32; // The maximum number of distinct memories (RAM, GPU RAM) supported by the buffer manager
	using data_location = std::bitset<max_memories>;

	struct per_buffer_data {
		struct per_memory_data {
			GridRegion<3> allocation;
			region_map<instruction*> last_accessors;
			region_map<instruction*> last_writers;

			explicit per_memory_data(const range<3>& range) : last_accessors(range), last_writers(range) {}

			void record_allocation(const GridRegion<3>& region, instruction* const instr) {
				allocation = GridRegion<3>::merge(allocation, region);
				// TODO this will generate antidependencies, but semantically, we want true dependencies.
				last_accessors.update_region(region, instr);
			}

			void record_read(const GridRegion<3>& region, instruction* const instr) { last_accessors.update_region(region, instr); }

			void record_write(const GridRegion<3>& region, instruction* const instr) {
				last_accessors.update_region(region, instr);
				last_writers.update_region(region, instr);
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
	const task_ring_buffer& m_task_buffer;
	size_t m_num_devices;
	std::vector<per_memory_data> m_memories;
	std::unordered_map<buffer_id, per_buffer_data> m_buffers;

	static memory_id next_location(const data_location& location, memory_id first);
};

} // namespace celerity::detail
