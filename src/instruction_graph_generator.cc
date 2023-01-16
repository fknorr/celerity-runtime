#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "command.h"
#include "task_manager.h"

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, size_t num_devices)
    : m_tm(tm), m_num_devices(num_devices), m_memories(num_devices + 1) {
	assert(num_devices + 1 <= max_memories);
}

void instruction_graph_generator::register_buffer(buffer_id bid, cl::sycl::range<3> range) {
	[[maybe_unused]] const auto [_, inserted] = m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(range, m_num_devices + 1));
	assert(inserted);
}

void instruction_graph_generator::unregister_buffer(buffer_id bid) { m_buffers.erase(bid); }

void instruction_graph_generator::register_host_object(host_object_id hoid) {
	[[maybe_unused]] const auto [_, inserted] = m_host_memory.host_objects.emplace(hoid, host_memory_data::per_host_object_data());
	assert(inserted);
}

void instruction_graph_generator::unregister_host_object(host_object_id hoid) { m_host_memory.host_objects.erase(hoid); }


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


void instruction_graph_generator::compile(const abstract_command& cmd) {
	assert(std::all_of(m_memories.begin(), m_memories.end(), [](const per_memory_data& memory) { return memory.last_epoch != nullptr; }) //
	       || isa<epoch_command>(&cmd));

	struct partial_instruction {
		subrange<3> execution_sr;
		memory_id mid = host_memory_id;
		std::unordered_map<buffer_id, GridRegion<3>> reads;
		std::unordered_map<buffer_id, GridRegion<3>> writes;
		std::unordered_set<host_object_id> side_effects;
		collective_group_id cgid = 0;
		instruction* instruction = nullptr;
	};
	std::vector<partial_instruction> cmd_insns;

	// 1) assign work, determine execution range and target memory of command instructions (and perform a local split where applicable)

	if(const auto* xcmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_tm.get_task(xcmd->get_tid());
		const auto command_sr = xcmd->get_execution_range();
		if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device /* don't split host tasks, but TODO oversubscription */) {
			// TODO oversubscription, tiled split
			const auto device_chunks = split_1d(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_num_devices);
			cmd_insns.resize(device_chunks.size());
			for(size_t i = 0; i < device_chunks.size(); ++i) {
				cmd_insns[i].execution_sr = subrange<3>(device_chunks[i].offset, device_chunks[i].range);
				cmd_insns[i].mid = memory_id(i + 1); // round-robin assignment
			}
		} else {
			cmd_insns.resize(1);
			cmd_insns[0].execution_sr = command_sr;
			// memory_id(1) is the first device - note this may lead to load imbalance if there's multiple independent unsplittale tasks.
			cmd_insns[0].mid = tsk.get_execution_target() == execution_target::device ? memory_id(1) : host_memory_id;
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

		for(const auto& [hoid, order] : tsk.get_side_effect_map()) {
			assert(cmd_insns[0].mid == host_memory_id);
			cmd_insns[0].side_effects.insert(hoid);
		}

		cmd_insns[0].cgid = tsk.get_collective_group_id();
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		cmd_insns.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		cmd_insns[0].mid = host_memory_id;
		cmd_insns[0].reads.emplace(pcmd->get_bid(), subrange_to_grid_box(pcmd->get_range()));
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		cmd_insns.resize(1);
		// This can eventually become a device memory id if we make use of CUDA-aware MPI
		cmd_insns[0].mid = host_memory_id;
		cmd_insns[0].writes.emplace(apcmd->get_bid(), apcmd->get_region());
	} else if(isa<horizon_command>(&cmd) || isa<epoch_command>(&cmd)) {
		cmd_insns.resize(m_memories.size());
		for(memory_id mid = 0; mid < m_memories.size(); ++mid) {
			cmd_insns[mid].mid = mid;
		}
	} else {
		assert(!"unhandled command type");
		std::abort();
	}

	// 2) create allocation instructions for any region unallocated in the memory read or written by a command instruction

	std::unordered_map<std::pair<buffer_id, memory_id>, GridRegion<3>, utils::pair_hash> unallocated_regions;
	for(const auto& insn : cmd_insns) {
		for(const auto& access : {insn.reads, insn.writes}) {
			for(const auto& [bid, region] : access) {
				const auto unallocated = GridRegion<3>::difference(region, m_buffers.at(bid).memories[insn.mid].allocation);
				if(!unallocated.empty()) { unallocated_regions[{bid, insn.mid}] = GridRegion<3>::merge(unallocated_regions[{bid, insn.mid}], unallocated); }
			}
		}
	}

	for(const auto& [bid_mid, region] : unallocated_regions) {
		const auto [bid, mid] = bid_mid;

		// TODO allocate a multiple of the allocator's page size (GridRegion might not be the right parameter to alloc_instruction)
		auto& alloc_instr = create<alloc_instruction>(mid, bid, region);

		// TODO until we have ndvbuffers everywhere, alloc_instructions need forward and backward dependencies to reproduce buffer-locking semantics
		//	 we could solve this by having resize_instructions instead that "read" the entire previous and "write" the entire new allocation
		add_dependency(alloc_instr, *m_memories[mid].last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);

		m_buffers.at(bid).memories[mid].record_allocation(region, &alloc_instr);
	}

	// 3) create device <-> host or device <-> device copy instructions to satisfy all command-instruction reads

	std::unordered_map<std::pair<buffer_id, memory_id>, std::vector<GridRegion<3>>, utils::pair_hash> unsatisfied_reads;
	for(size_t cmd_insn_idx = 0; cmd_insn_idx < cmd_insns.size(); ++cmd_insn_idx) {
		auto& insn = cmd_insns[cmd_insn_idx];
		for(const auto& [bid, region] : insn.reads) {
			auto& buffer = m_buffers.at(bid);
			GridRegion<3> unsatified_region;
			for(const auto& [box, location] : buffer.newest_data_location.get_region_values(region)) {
				if(!location.test(insn.mid)) { unsatified_region = GridRegion<3>::merge(unsatified_region, box); }
			}
			if(!unsatified_region.empty()) { unsatisfied_reads[{bid, insn.mid}].push_back(unsatified_region); }
		}
	}

	// transform vectors of potentially-overlapping unsatisfied regions into disjoint regions
	for(auto& [bid_mid, regions] : unsatisfied_reads) {
	restart:
		for(size_t i = 0; i < regions.size(); ++i) {
			for(size_t j = i + 1; j < regions.size(); ++j) {
				auto intersection = GridRegion<3>::intersect(regions[i], regions[j]);
				if(!intersection.empty()) {
					regions[i] = GridRegion<3>::difference(regions[i], intersection);
					regions[j] = GridRegion<3>::difference(regions[j], intersection);
					regions.push_back(std::move(intersection));
					// if intersections above are actually subsets, we will end up with empty regions
					regions.erase(std::remove_if(regions.begin(), regions.end(), std::mem_fn(&GridRegion<3>::empty)), regions.end());
					goto restart;
				}
			}
		}
	}

	struct copy_template {
		buffer_id bid;
		memory_id from;
		memory_id to;
		GridRegion<3> region;
	};

	std::vector<copy_template> pending_copies;
	for(auto& [bid_mid, disjoint_regions] : unsatisfied_reads) {
		const auto& [bid, mid] = bid_mid;
		auto& buffer = m_buffers.at(bid);
		for(auto& region : disjoint_regions) {
			const auto region_sources = buffer.newest_data_location.get_region_values(region);

			// try finding a common source for the entire region first to minimize instruction / synchronization complexity down the line
			const auto common_sources = std::accumulate(region_sources.begin(), region_sources.end(), data_location(),
			    [](const data_location common, const std::pair<GridBox<3>, data_location>& box_sources) { return common & box_sources.second; });

			if(const auto common_device_sources = data_location(common_sources).reset(host_memory_id); common_device_sources.any()) {
				// best case: we can copy all data from a single device
				const auto copy_from = next_location(common_device_sources, mid + 1);
				pending_copies.push_back({bid, copy_from, mid, std::move(region)});
				continue;
			}

			// see if we can find data exclusively on devices, or exclusively on the host
			const auto have_all_device_sources = std::all_of(region_sources.begin(), region_sources.end(),
			    [](const std::pair<GridBox<3>, data_location>& box_sources) { return data_location(box_sources.second).reset(host_memory_id).any(); });
			if(!have_all_device_sources && common_sources[host_memory_id]) {
				// prefer a single copy from the host to mixing and matching host and device sources
				pending_copies.push_back({bid, host_memory_id, mid, std::move(region)});
				continue;
			}

			// mix and match sources - there exists an optimal solution, but for now we just assemble source regions by picking the next device if any,
			// or the host as a fallback.
			std::unordered_map<memory_id, GridRegion<3>> combined_source_regions;
			for(const auto& [box, sources] : region_sources) {
				memory_id copy_from;
				if(const auto device_sources = data_location(sources).reset(host_memory_id); device_sources.any()) {
					copy_from = next_location(device_sources, mid + 1);
				} else {
					copy_from = host_memory_id;
				}
				auto& copy_region = combined_source_regions[copy_from];
				copy_region = GridRegion<3>::merge(copy_region, box);
			}
			for(auto& [copy_from, copy_region] : combined_source_regions) {
				pending_copies.push_back({bid, copy_from, mid, std::move(copy_region)});
			}
		}
	}

	for(auto& copy : pending_copies) {
		assert(copy.from != copy.to);
		auto& buffer = m_buffers.at(copy.bid);

		auto [source_instr, dest_instr] = create_copy(copy.from, copy.to, copy.bid, copy.region);
		for(const auto& [_, last_writer_instr] : buffer.memories[copy.from].last_writers.get_region_values(copy.region)) {
			add_dependency(source_instr, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
		}
		buffer.memories[copy.from].record_read(copy.region, &source_instr);

		for(const auto& [_, front] : buffer.memories[copy.to].access_fronts.get_region_values(copy.region)) {
			for(const auto dep_instr : front.front) {
				add_dependency(dest_instr, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
			}
		}
		add_dependency(dest_instr, *m_memories[copy.to].last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
		buffer.memories[copy.to].record_write(copy.region, &dest_instr);
		for(auto& [box, location] : buffer.newest_data_location.get_region_values(copy.region)) {
			buffer.newest_data_location.update_region(box, data_location(location).set(copy.to));
		}
	}

	// 4) create the actual command instructions

	if(const auto* xcmd = dynamic_cast<const execution_command*>(&cmd)) {
		const auto& tsk = *m_tm.get_task(xcmd->get_tid());
		for(auto& in : cmd_insns) {
			if(tsk.get_execution_target() == execution_target::device) {
				assert(in.execution_sr.range.size() > 0);
				assert(in.mid != host_memory_id && in.mid - 1 < m_num_devices);
				in.instruction = &create<device_kernel_instruction>(device_id(in.mid - 1), tsk, in.execution_sr);
			} else {
				assert(tsk.get_execution_target() == execution_target::host);
				assert(in.mid == host_memory_id);
				in.instruction = &create<host_kernel_instruction>(tsk, in.execution_sr);
			}
		}
	} else if(const auto* pcmd = dynamic_cast<const push_command*>(&cmd)) {
		assert(cmd_insns.size() == 1);
		cmd_insns.front().instruction = &create<send_instruction>(pcmd->get_target(), pcmd->get_bid(), pcmd->get_range());
	} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		assert(cmd_insns.size() == 1);
		cmd_insns.front().instruction = &create<recv_instruction>(apcmd->get_transfer_id(), apcmd->get_bid(), apcmd->get_region());
	} else if(const auto* hcmd = dynamic_cast<const horizon_command*>(&cmd)) {
		for(auto& insn : cmd_insns) {
			insn.instruction = &create<horizon_instruction>(insn.mid);
			// TODO reduce execution fronts
		}
	} else if(const auto* ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
		for(auto& insn : cmd_insns) {
			insn.instruction = &create<epoch_instruction>(insn.mid);
			// TODO reduce execution fronts
		}
	} else {
		assert(!"unhandled command type");
		std::abort();
	}

	// 5) compute dependencies between command instructions and previous copy, allocation, and command (!) instructions

	// TODO this will not work correctly for oversubscription
	//	 - read-all + write-1:1 cannot be oversubscribed at all chunks would need a global read->write barrier (how would the kernel even look like?)
	//	 - oversubscribed host tasks would need dependencies between their chunks based on side effects and collective groups
	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, region] : insn.reads) {
			auto& buffer = m_buffers.at(bid);
			for(const auto& [_, last_writer_instr] : buffer.memories[insn.mid].last_writers.get_region_values(region)) {
				add_dependency(*insn.instruction, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
			}
		}
		for(const auto& [bid, region] : insn.writes) {
			auto& buffer = m_buffers.at(bid);
			for(const auto& [_, front] : buffer.memories[insn.mid].access_fronts.get_region_values(region)) {
				for(const auto dep_instr : front.front) {
					add_dependency(*insn.instruction, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
				}
			}
		}
		for(const auto hoid : insn.side_effects) {
			assert(insn.mid == host_memory_id);
			if(const auto last_side_effect = m_host_memory.host_objects.at(hoid).last_side_effect) {
				add_dependency(*insn.instruction, *last_side_effect, dependency_kind::true_dep, dependency_origin::dataflow);
			}
		}
		if(insn.cgid != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(insn.mid == host_memory_id);
			auto& group = m_host_memory.collective_groups[insn.cgid]; // allow default-insertion since we do not register CGs explicitly
			if(group.last_host_task) {
				add_dependency(*insn.instruction, *group.last_host_task, dependency_kind::true_dep, dependency_origin::collective_group_serialization);
			}
		}
	}

	// 6) update data locations and last writers resulting from command instructions

	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, region] : insn.writes) {
			assert(insn.instruction != nullptr);
			m_buffers.at(bid).newest_data_location.update_region(region, data_location().set(insn.mid));
			m_buffers.at(bid).memories[insn.mid].record_write(region, insn.instruction);
		}
		for(const auto hoid : insn.side_effects) {
			assert(insn.mid == host_memory_id);
			m_host_memory.host_objects.at(hoid).last_side_effect = insn.instruction;
		}
		if(insn.cgid != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(insn.mid == host_memory_id);
			m_host_memory.collective_groups.at(insn.cgid).last_host_task = insn.instruction;
		}
	}

	// 7) insert epoch and horizon dependencies, apply epochs

	for(auto& insn : cmd_insns) {
		auto& memory = m_memories[insn.mid];

		if(isa<horizon_command>(&cmd) || isa<epoch_command>(&cmd)) {
			memory.collapse_execution_front_to(insn.instruction);

			instruction* new_epoch = nullptr;
			if(isa<epoch_command>(&cmd)) {
				new_epoch = insn.instruction;
			} else {
				new_epoch = memory.last_horizon; // can be null
			}

			if(new_epoch) {
				for(auto& [_, buffer] : m_buffers) {
					buffer.memories[insn.mid].apply_epoch(new_epoch);
				}
				if(insn.mid == host_memory_id) { m_host_memory.apply_epoch(new_epoch); }
				memory.last_epoch = new_epoch;

				// TODO prune graph. Should we re-write node dependencies?
				//	 - pro: No accidentally following stale pointers
				//   - con: Thread safety (but how would a consumer know which dependency edges can be followed)?
			}
		} else {
			// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
			// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
			if(memory.last_epoch != nullptr) {
				const auto deps = insn.instruction->get_dependencies();
				if(std::none_of(deps.begin(), deps.end(), [](const instruction::dependency& dep) { return dep.kind == dependency_kind::true_dep; })) {
					add_dependency(*insn.instruction, *memory.last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
				}
			}
		}
	}
}

} // namespace celerity::detail