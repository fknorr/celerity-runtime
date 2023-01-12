#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "command.h"
#include "task_ring_buffer.h"

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_ring_buffer& task_buffer, size_t num_devices)
    : m_task_buffer(task_buffer), m_num_devices(num_devices), m_memories(num_devices + 1) {
	assert(num_devices + 1 <= max_memories);
	for(auto& mem : m_memories) {
		mem.epoch = &m_idag.create<epoch_instruction>();
	}
}

void instruction_graph_generator::register_buffer(buffer_id bid, cl::sycl::range<3> range) {
	[[maybe_unused]] const auto [_, inserted] = m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(range, m_num_devices + 1));
	assert(inserted);
}

void instruction_graph_generator::unregister_buffer(buffer_id bid) { m_buffers.erase(bid); }


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
	instruction* instruction = nullptr;
};

void instruction_graph_generator::compile(const abstract_command& cmd) {
	std::vector<partial_instruction> cmd_insns;

	// 1) assign work, determine execution range and target memory of command instructions (and perform a local split where applicable)

	if(const auto* xcmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(xcmd->get_tid());
		const auto command_sr = xcmd->get_execution_range();
		if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device /* don't split host tasks, but TODO oversubscription */) {
			// TODO oversubscription, tiled split
			const auto device_chunks = split_1d(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_num_devices);
			cmd_insns.resize(device_chunks.size());
			for(size_t i = 0; i < device_chunks.size(); ++i) {
				cmd_insns[i].execution_sr = subrange<3>(device_chunks[i].offset, device_chunks[i].range);
				cmd_insns[i].memory = memory_id(i + 1); // round-robin assignment
			}
		} else {
			cmd_insns.resize(1);
			cmd_insns[0].execution_sr = command_sr;
			// memory_id(1) is the first device - note this may lead to load imbalance if there's multiple independent unsplittale tasks.
			cmd_insns[0].memory = tsk.get_execution_target() == execution_target::device ? memory_id(1) : host_memory_id;
		}

		const auto& bam = tsk.get_buffer_access_map();
		for(const auto bid : bam.get_accessed_buffers()) {
			for(auto& insn : cmd_insns) {
				GridRegion<3> b_reads;
				GridRegion<3> b_writes;
				for(const auto mode : bam.get_access_modes(bid)) {
					const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
					if(access::mode_traits::is_consumer(mode)) { b_reads = GridRegion<3>::merge(b_reads, req); }
					if(access::mode_traits::is_producer(mode)) { b_writes = GridRegion<3>::merge(b_writes, req); }
				}
				if(!b_reads.empty()) { insn.reads.emplace(bid, std::move(b_reads)); }
				if(!b_writes.empty()) { insn.writes.emplace(bid, std::move(b_writes)); }
			}
		}
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		cmd_insns.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		cmd_insns[0].memory = host_memory_id;
		cmd_insns[0].reads.emplace(pcmd->get_bid(), subrange_to_grid_box(pcmd->get_range()));
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		cmd_insns.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		cmd_insns[0].memory = host_memory_id;
		cmd_insns[0].writes.emplace(apcmd->get_bid(), apcmd->get_region());
	} else if(const auto* ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
		cmd_insns.resize(m_memories.size());
		for(memory_id mid = 0; mid < m_memories.size(); ++mid) {
			cmd_insns[mid].memory = mid;
		}
	} else {
		// TODO horizons
		std::abort();
	}

	// 2) create allocation instructions for any region unallocated in the memory read or written by a command instruction

	std::unordered_map<std::pair<buffer_id, memory_id>, GridRegion<3>, utils::pair_hash> unallocated_regions;
	for(const auto& insn : cmd_insns) {
		for(const auto& access : {insn.reads, insn.writes}) {
			for(const auto& [bid, region] : access) {
				const auto unallocated = GridRegion<3>::difference(region, m_buffers.at(bid).memories[insn.memory].allocation);
				if(!unallocated.empty()) {
					unallocated_regions[{bid, insn.memory}] = GridRegion<3>::merge(unallocated_regions[{bid, insn.memory}], unallocated);
				}
			}
		}
	}

	for(const auto& [bid_memory, region] : unallocated_regions) {
		const auto [bid, memory] = bid_memory;

		// TODO allocate a multiple of the allocator's page size (GridRegion might not be the right parameter to alloc_instruction)
		auto& alloc_instr = m_idag.create<alloc_instruction>(bid, memory, region);

		// TODO until we have ndvbuffers everywhere, alloc_instructions need forward and backward dependencies to reproduce buffer-locking semantics
		//	 we could solve this by having resize_instructions instead that "read" the entire previous and "write" the entire new allocation
		instruction_graph::add_dependency(alloc_instr, *m_memories[memory].epoch, dependency_kind::true_dep, dependency_origin::last_epoch);

		auto& metadata = m_buffers.at(bid).memories[memory];
		metadata.allocation = GridRegion<3>::merge(metadata.allocation, region);
		metadata.last_writers.update_region(region, &alloc_instr);
	}

	// 3) create device <-> host or device <-> device copy instructions to satisfy all command-instruction reads

	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, region] : insn.reads) {
			auto& buffer = m_buffers.at(bid);
			for(const auto& [box, location] : buffer.newest_data_location.get_region_values(region)) {
				if(location.test(insn.memory)) continue;

				// TODO copy_from will currently create chains, ensure that if there are multiple locations, we don't use one that has been copied to
				//  while generating this instruction (unless we can avoid a second h2d by doing a single-link (!) h2d -> d2d chain).
				memory_id copy_from;
				if(insn.memory == host_memory_id) {
					// device -> host
					copy_from = next_location(location, insn.memory + 1);
				} else if(const auto device_sources = data_location(location).reset(host_memory_id); device_sources.any()) {
					// device -> device when possible (faster than host -> device)
					copy_from = next_location(device_sources, insn.memory + 1);
				} else {
					// host -> device
					copy_from = host_memory_id;
				}
				assert(copy_from != insn.memory);
				assert(location.test(copy_from));

				auto& copy_instr = m_idag.create<copy_instruction>(bid, grid_box_to_subrange(box), copy_from, insn.memory);

				for(const auto& [_, last_writer_instr] : buffer.memories[copy_from].last_writers.get_region_values(box)) {
					instruction_graph::add_dependency(copy_instr, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
				}
				for(const auto& [_, last_writer_instr] : buffer.memories[insn.memory].last_writers.get_region_values(box)) {
					instruction_graph::add_dependency(copy_instr, *last_writer_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
				}
				instruction_graph::add_dependency(*insn.instruction, *m_memories[insn.memory].epoch, dependency_kind::true_dep, dependency_origin::last_epoch);

				buffer.newest_data_location.update_region(box, data_location(location).set(insn.memory));
				m_buffers.at(bid).memories[insn.memory].last_writers.update_region(box, &copy_instr);
			}
		}
	}

	// 4) create the actual command instructions

	if(const auto* xcmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_task_buffer.get_task(xcmd->get_tid());
		for(auto& in : cmd_insns) {
			assert(in.execution_sr.range.size() > 0);
			assert(in.memory > 0 && in.memory - 1 < m_num_devices);
			in.instruction = &m_idag.create<device_kernel_instruction>(tsk, device_id(in.memory - 1), in.execution_sr);
		}
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		assert(cmd_insns.size() == 1);
		cmd_insns.front().instruction = &m_idag.create<send_instruction>(pcmd->get_target(), pcmd->get_bid(), pcmd->get_range());
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		assert(cmd_insns.size() == 1);
		cmd_insns.front().instruction = &m_idag.create<recv_instruction>(apcmd->get_transfer_id(), apcmd->get_bid(), apcmd->get_region());
	} else if(const auto* ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
		for(auto& insn : cmd_insns) {
			m_idag.create<epoch_instruction>();
			// TODO reduce execution fronts
		}
	} else {
		// TODO horizons
		std::abort();
	}

	// 5) compute dependencies between command instructions and previous copy, allocation, and command (!) instructions
	// TODO side effects, collective-group serialization

	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, region] : insn.reads) {
			auto& buffer = m_buffers.at(bid);
			for(const auto& [_, last_writer_instr] : buffer.memories[insn.memory].last_writers.get_region_values(region)) {
				instruction_graph::add_dependency(*insn.instruction, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
			}
		}
		if(insn.reads.empty()) {
			// if there is no transitive dependency to the last epoch (because this is a pure producer), insert an explicit one
			instruction_graph::add_dependency(*insn.instruction, *m_memories[insn.memory].epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
		}
		for(const auto& [bid, region] : insn.writes) {
			auto& buffer = m_buffers.at(bid);
			for(const auto& [_, last_writer_instr] : buffer.memories[insn.memory].last_writers.get_region_values(region)) {
				instruction_graph::add_dependency(*insn.instruction, *last_writer_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
			}
		}
	}

	// 6) update data locations and last writers resulting from command instructions

	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, region] : insn.writes) {
			assert(insn.instruction != nullptr);
			m_buffers.at(bid).newest_data_location.update_region(region, data_location().set(insn.memory));
			m_buffers.at(bid).memories[insn.memory].last_writers.update_region(region, insn.instruction);
		}
	}

	// 7) apply epochs, where applicable

	if(const auto* ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
		for(auto& insn : cmd_insns) {
			m_memories[insn.memory].epoch = insn.instruction;
			// TODO update last writers
			// TODO collapse execution front
		}
		// TODO prune graph
	}
}

} // namespace celerity::detail