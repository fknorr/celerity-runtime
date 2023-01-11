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

  private:
	static constexpr size_t max_memories = 32; // The maximum number of distinct memories (RAM, GPU RAM) supported by the buffer manager
	using data_location = std::bitset<max_memories>;

	struct per_memory_data {
		std::unordered_map<buffer_id, region_map<const instruction*>> last_writers;
	};

	instruction_graph m_idag;
	const task_ring_buffer& m_task_buffer;
	size_t m_num_devices;
	std::unordered_map<buffer_id, region_map<data_location>> m_newest_data_location;
	std::vector<per_memory_data> m_memories;

	static memory_id next_location(const data_location& location, memory_id first);
};

} // namespace celerity::detail
