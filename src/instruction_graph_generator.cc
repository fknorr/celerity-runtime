#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "command.h"
#include "task_ring_buffer.h"

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_ring_buffer& task_buffer, size_t num_devices)
    : m_task_buffer(task_buffer), m_num_devices(num_devices), m_memories(1 + num_devices) {}

void instruction_graph_generator::register_buffer(buffer_id bid, cl::sycl::range<3> range) {
	{
		[[maybe_unused]] const auto [_, inserted] = m_newest_data_location.emplace(bid, region_map<data_location>(range, data_location{}));
		assert(inserted);
	}
	for(memory_id mid = 0; mid < m_num_devices + 1; ++mid) {
		[[maybe_unused]] const auto [_, inserted] = m_memories[mid].last_writers.emplace(bid, region_map<const instruction*>(range));
		assert(inserted);
	}
}

void instruction_graph_generator::unregister_buffer(buffer_id bid) {
	assert(m_newest_data_location.count(bid));
	m_newest_data_location.erase(bid);
	for(memory_id mid = 0; mid < m_num_devices + 1; ++mid) {
		assert(m_memories[mid].last_writers.count(bid));
		m_memories[mid].last_writers.erase(bid);
	}
}


memory_id instruction_graph_generator::next_location(const data_location& location, memory_id first) {
	for(size_t i = 0; i < max_memories; ++i) {
		const memory_id mem = (first + i) % max_memories;
		if(location[mem]) { return mem; }
	}
	assert(!"data is requested to be read, but not located in any memory");
	std::abort();
}


// TODO HACK we're just pulling in the splitting logic from distributed_graph_generator here
std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);


struct partial_instruction {
	subrange<3> execution_sr;
	memory_id memory = host_memory_id;
	std::unordered_map<buffer_id, GridRegion<3>> reads;
	std::unordered_map<buffer_id, GridRegion<3>> writes;
	const instruction* instruction = nullptr;
};

void instruction_graph_generator::compile(const abstract_command& cmd) {
	std::vector<partial_instruction> instrs;

	if(const auto* ecmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(ecmd->get_tid());
		const auto command_sr = ecmd->get_execution_range();
		if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device /* don't split host tasks, but TODO oversubscription */) {
			// TODO oversubscription, tiled split
			const auto device_chunks = split_1d(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_num_devices);
			instrs.resize(device_chunks.size());
			for(size_t i = 0; i < device_chunks.size(); ++i) {
				instrs[i].execution_sr = subrange<3>(device_chunks[i].offset, device_chunks[i].range);
				instrs[i].memory = memory_id(i + 1); // round-robin assignment
			}
		} else {
			instrs.resize(1);
			instrs[0].execution_sr = command_sr;
			// memory_id(1) is the first device - note this may lead to load imbalance if there's multiple independent unsplittale tasks.
			instrs[0].memory = tsk.get_execution_target() == execution_target::device ? memory_id(1) : host_memory_id;
		}

		const auto& bam = tsk.get_buffer_access_map();
		for(const auto bid : bam.get_accessed_buffers()) {
			for(auto& in : instrs) {
				GridRegion<3> b_reads;
				GridRegion<3> b_writes;
				for(const auto mode : bam.get_access_modes(bid)) {
					const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), in.execution_sr, tsk.get_global_size());
					if(access::mode_traits::is_consumer(mode)) { b_reads = GridRegion<3>::merge(b_reads, req); }
					if(access::mode_traits::is_producer(mode)) { b_writes = GridRegion<3>::merge(b_writes, req); }
				}
				if(!b_reads.empty()) { in.reads.emplace(bid, std::move(b_reads)); }
				if(!b_writes.empty()) { in.writes.emplace(bid, std::move(b_writes)); }
			}
		}
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		instrs.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		instrs[0].memory = host_memory_id;
		instrs[0].reads.emplace(pcmd->get_bid(), subrange_to_grid_box(pcmd->get_range()));
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		instrs.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		instrs[0].memory = host_memory_id;
		instrs[0].writes.emplace(apcmd->get_bid(), apcmd->get_region());
	} else {
		// TODO epochs and horizons
		std::abort();
	}

	for(const auto& in : instrs) {
		for(const auto& [bid, region] : in.reads) {
			auto& buffer_location = m_newest_data_location.at(bid);
			for(const auto& [box, location] : buffer_location.get_region_values(region)) {
				if(!location.test(in.memory)) {
					// TODO copy_from will currently create chains, ensure that if there are multiple locations, we don't use one that has been copied to
					//  while generating this instruction (unless we can avoid a second h2d by doing a single-link (!) h2d -> d2d chain).
					memory_id copy_from;
					if(in.memory == host_memory_id) {
						// device -> host
						copy_from = next_location(location, in.memory + 1);
					} else if(const auto device_sources = data_location(location).reset(host_memory_id); device_sources.any()) {
						// device -> device when possible (faster than host -> device)
						copy_from = next_location(device_sources, in.memory + 1);
					} else {
						// host -> device
						copy_from = host_memory_id;
					}
					assert(copy_from != in.memory);
					assert(location.test(copy_from));

					auto& copy_instr = m_idag.create<copy_instruction>(bid, grid_box_to_subrange(box), copy_from, in.memory);

					buffer_location.update_region(box, data_location(location).set(in.memory));
					m_memories[in.memory].last_writers.at(bid).update_region(box, &copy_instr);
				}
			}
		}
	}

	if(const auto* ecmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(ecmd->get_tid());
		for(auto& in : instrs) {
			assert(in.execution_sr.range.size() > 0);
			assert(in.memory > 0 && in.memory - 1 < m_num_devices);
			in.instruction = &m_idag.create<device_kernel_instruction>(tsk, device_id(in.memory - 1), in.execution_sr);
		}
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		assert(instrs.size() == 1);
		instrs.front().instruction = &m_idag.create<send_instruction>(pcmd->get_target(), pcmd->get_bid(), pcmd->get_range());
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		assert(instrs.size() == 1);
		instrs.front().instruction = &m_idag.create<recv_instruction>(apcmd->get_transfer_id(), apcmd->get_bid(), apcmd->get_region());
	} else {
		// TODO epochs and horizons
		std::abort();
	}

	for(const auto& in : instrs) {
		for(const auto& [bid, region] : in.writes) {
			assert(in.instruction != nullptr);
			m_newest_data_location.at(bid).update_region(region, data_location().set(in.memory));
			m_memories[in.memory].last_writers.at(bid).update_region(region, in.instruction);
		}
	}
}

} // namespace celerity::detail