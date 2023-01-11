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

void instruction_graph_generator::compile(const abstract_command& cmd) {
	memory_id memory;
	std::unordered_map<buffer_id, GridRegion<3>> reads;
	std::unordered_map<buffer_id, GridRegion<3>> writes;

	if(const auto* ecmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(ecmd->get_tid());
		const auto& bam = tsk.get_buffer_access_map();
		for(const auto bid : bam.get_accessed_buffers()) {
			GridRegion<3> b_reads;
			GridRegion<3> b_writes;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), ecmd->get_execution_range(), tsk.get_global_size());
				if(access::mode_traits::is_consumer(mode)) { b_reads = GridRegion<3>::merge(b_reads, req); }
				if(access::mode_traits::is_producer(mode)) { b_writes = GridRegion<3>::merge(b_writes, req); }
			}
			if(!b_reads.empty()) { reads.emplace(bid, std::move(b_reads)); }
			if(!b_writes.empty()) { writes.emplace(bid, std::move(b_writes)); }
		}

		// TODO do split and device assignment first!
		memory = tsk.get_execution_target() == execution_target::device ? memory_id(1) : host_memory_id;
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		reads.emplace(pcmd->get_bid(), subrange_to_grid_box(pcmd->get_range()));
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		memory = host_memory_id;
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		writes.emplace(apcmd->get_bid(), apcmd->get_region());
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		memory = host_memory_id;
	} else {
		// TODO epochs and horizons
		std::abort();
	}

	for(const auto& [bid, region] : reads) {
		auto& buffer_location = m_newest_data_location.at(bid);
		for(const auto& [box, location] : buffer_location.get_region_values(region)) {
			if(!location.test(memory)) {
				memory_id copy_from;
				if(memory == host_memory_id) {
					// device -> host
					copy_from = next_location(location, memory + 1);
				} else if(const auto device_sources = data_location(location).reset(host_memory_id); device_sources.any()) {
					// device -> device when possible (faster than host -> device)
					copy_from = next_location(device_sources, memory + 1);
				} else {
					// host -> device
					copy_from = host_memory_id;
				}
				assert(copy_from != memory);
				assert(location.test(copy_from));

				auto& copy_instr = m_idag.create<copy_instruction>(bid, grid_box_to_subrange(box), copy_from, memory);

				// TODO these updates need to be staged so we don't accidentally chain transfers when splitting a task.
				buffer_location.update_region(box, data_location(location).set(memory));
				m_memories[memory].last_writers.at(bid).update_region(box, &copy_instr);
			}
		}
	}

	instruction* the_instr = nullptr;
	if(const auto* ecmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(ecmd->get_tid());
		the_instr = &m_idag.create<device_kernel_instruction>(tsk, ecmd->get_device_id(), ecmd->get_execution_range());
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		the_instr = &m_idag.create<send_instruction>(pcmd->get_target(), pcmd->get_bid(), pcmd->get_range());
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		the_instr = &m_idag.create<recv_instruction>(apcmd->get_transfer_id(), apcmd->get_bid(), apcmd->get_region());
	} else {
		// TODO epochs and horizons
		std::abort();
	}

	for(const auto& [bid, region] : writes) {
		assert(the_instr != nullptr);
		m_newest_data_location.at(bid).update_region(region, data_location().set(memory));
		m_memories[memory].last_writers.at(bid).update_region(region, the_instr);
	}
}

} // namespace celerity::detail