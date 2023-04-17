#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "allscale/api/user/data/grid.h"
#include "command.h"
#include "grid.h"
#include "task.h"
#include "task_manager.h"

#include <numeric>

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, size_t num_devices) : m_tm(tm), m_num_devices(num_devices) {
	assert(num_devices + 1 <= max_memories);
	m_last_epoch = &create<epoch_instruction>(command_id(0 /* or so we assume */));
}

void instruction_graph_generator::register_buffer(buffer_id bid, int dims, range<3> range, const size_t elem_size, const size_t elem_align) {
	[[maybe_unused]] const auto [_, inserted] =
	    m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(dims, range, elem_size, elem_align, m_num_devices + 1));
	assert(inserted);
}

void instruction_graph_generator::unregister_buffer(buffer_id bid) { m_buffers.erase(bid); }

void instruction_graph_generator::register_host_object(host_object_id hoid) {
	[[maybe_unused]] const auto [_, inserted] = m_host_objects.emplace(hoid, per_host_object_data());
	assert(inserted);
}

void instruction_graph_generator::unregister_host_object(host_object_id hoid) { m_host_objects.erase(hoid); }


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
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);


void instruction_graph_generator::compile(const abstract_command& cmd) {
	// We do not generate instructions for await-push commands immediately upon receiving them; instead, we buffer them and generate recv-instructions
	// as soon as data is to be read by another instruction. This way, we can split the recv instructions and avoid unnecessary synchronization points
	// between chunks that can otherwise profit from a transfer-compute overlap.
	if(const auto* apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
		auto& buffer = m_buffers.at(apcmd->get_bid());
		auto region = apcmd->get_region();

#ifndef NDEBUG
		for(const auto& [box, trid] : buffer.pending_await_pushes.get_region_values(region)) {
			assert(trid == transfer_id() && "received an await-push command into a previously await-pushed region without an intermediate read");
		}
#endif
		buffer.newest_data_location.update_region(region, data_location()); // not present anywhere locally
		buffer.pending_await_pushes.update_region(region, apcmd->get_transfer_id());
		return;
	}

	struct partial_instruction {
		subrange<3> execution_sr;
		device_id did = -1;
		memory_id mid = host_memory_id;
		buffer_read_write_map rw_map;
		side_effect_map se_map;
		collective_group_id cgid = 0;
		instruction* instruction = nullptr;
	};
	std::vector<partial_instruction> cmd_insns;

	// 1) assign work, determine execution range and target memory of command instructions (and perform a local split where applicable)

	utils::match(
	    cmd,
	    [&](const execution_command& xcmd) {
		    const auto& tsk = dynamic_cast<const command_group_task&>(*m_tm.get_task(xcmd.get_tid()));
		    const auto command_sr = xcmd.get_execution_range();
		    if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device) {
			    const auto split = tsk.get_hint<experimental::hints::tiled_split>() ? split_2d : split_1d;
			    const auto oversubscribe = tsk.get_hint<experimental::hints::oversubscribe>();
			    const auto num_sub_chunks_per_device = oversubscribe ? oversubscribe->get_factor() : 1;

			    const auto device_chunks = split(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_num_devices);
			    for(device_id did = 0; did < m_num_devices && did < device_chunks.size(); ++did) {
				    // subdivide recursively so that in case of a 2D split, we still produce 2D tiles instead of a row-subset
				    const auto this_device_sub_chunks = split(device_chunks[did], tsk.get_granularity(), num_sub_chunks_per_device);
				    for(const auto& sub_chunk : this_device_sub_chunks) {
					    auto& insn = cmd_insns.emplace_back();
					    insn.execution_sr = subrange<3>(sub_chunk.offset, sub_chunk.range);
					    insn.did = did;
					    insn.mid = device_to_memory_id(did);
				    }
			    }
		    } else {
			    // TODO oversubscribe distributed host tasks (but not if they have side effects)
			    auto& insn = cmd_insns.emplace_back();
			    insn.execution_sr = command_sr;
			    if(tsk.get_execution_target() == execution_target::device) {
				    // Assign work to the first device - note this may lead to load imbalance if there's multiple independent unsplittable tasks.
				    //   - but at the same time, keeping work on one device minimizes data transfers => this can only truly be solved through profiling.
				    insn.did = 0;
				    insn.mid = device_to_memory_id(insn.did);
			    } else {
				    insn.mid = host_memory_id;
			    }
		    }

		    const auto& bam = tsk.get_buffer_access_map();
		    for(const auto bid : bam.get_accessed_buffers()) {
			    for(auto& insn : cmd_insns) {
				    reads_writes rw;
				    for(const auto mode : bam.get_access_modes(bid)) {
					    const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
					    if(access::mode_traits::is_consumer(mode)) { rw.reads = GridRegion<3>::merge(rw.reads, req); }
					    if(access::mode_traits::is_producer(mode)) { rw.writes = GridRegion<3>::merge(rw.writes, req); }
				    }
				    rw.contiguous_boxes = bam.get_required_contiguous_boxes(bid, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
				    if(!rw.empty()) { insn.rw_map.emplace(bid, std::move(rw)); }
			    }
		    }

		    if(!tsk.get_side_effect_map().empty()) {
			    assert(cmd_insns.size() == 1); // split instructions for host tasks with side effects would race
			    assert(cmd_insns[0].mid == host_memory_id);
			    cmd_insns[0].se_map = tsk.get_side_effect_map();
		    }

		    cmd_insns[0].cgid = tsk.get_collective_group_id();
	    },
	    [&](const push_command& pcmd) {
		    const auto bid = pcmd.get_bid();
		    auto& buffer = m_buffers.at(bid);
		    const auto push_box = subrange_to_grid_box(pcmd.get_range());

		    // We want to generate the fewest number of send instructions possible without introducing new synchronization points between chunks of the same
		    // command that generated the pushed data. This will allow compute-transfer overlap, especially in the case of oversubscribed splits.
		    std::unordered_map<instruction*, GridRegion<3>> writer_regions;
		    for(auto& [box, writer] : buffer.original_writers.get_region_values(push_box)) {
			    auto& region = writer_regions[writer]; // allow default-insert
			    region = GridRegion<3>::merge(region, box);
		    }
		    for(auto& [writer, region] : writer_regions) {
			    auto& insn = cmd_insns.emplace_back();
			    insn.mid = host_memory_id;
			    insn.rw_map.emplace(bid, reads_writes{std::move(region), {}, {}});
		    }
	    },
	    [&](const horizon_command& /* hcmd */) { cmd_insns.emplace_back(); }, //
	    [&](const epoch_command& /* ecmd */) { cmd_insns.emplace_back(); },
	    [](const auto& /* unhandled */) {
		    assert(!"unhandled command type");
		    std::abort();
	    });

	// 2) create allocation instructions for any region unallocated in the memory read or written by a command instruction

	// TODO this must
	// 	- be inherently conditional on the presence of virtual memory
	//  - for VM, keep page size in mind

	std::unordered_map<std::pair<buffer_id, memory_id>, std::vector<GridBox<3>>, utils::pair_hash> not_contiguously_allocated_boxes;
	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, rw] : insn.rw_map) {
			const auto& memory = m_buffers.at(bid).memories[insn.mid];
			for(auto& box : rw.contiguous_boxes) {
				if(!memory.is_allocated_contiguously(box)) { not_contiguously_allocated_boxes[{bid, insn.mid}].push_back(box); }
			}
		}
	}

	// TODO allocate for await-pushes!

	for(auto& [bid_mid, boxes] : not_contiguously_allocated_boxes) {
		const auto& [bid, mid] = bid_mid;

		auto& buffer = m_buffers.at(bid);
		auto& memory = buffer.memories[mid];

		contiguous_box_set contiguous_after_reallocation;
		for(auto& alloc : memory.allocations) {
			contiguous_after_reallocation.insert(alloc.box);
		}
		for(auto& box : boxes) {
			contiguous_after_reallocation.insert(box);
		}

		std::vector<allocation_id> free_after_reallocation;
		for(auto& alloc : memory.allocations) {
			if(std::none_of(contiguous_after_reallocation.begin(), contiguous_after_reallocation.end(), [&](auto& box) { return alloc.box == box; })) {
				free_after_reallocation.push_back(alloc.aid);
			}
		}

		for(const auto& dest_box : contiguous_after_reallocation) {
			if(std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == dest_box; })) continue;

			auto& dest = memory.allocations.emplace_back(m_next_aid++, dest_box);
			auto& alloc_instr = create<alloc_instruction>(dest.aid, mid, dest.box.area() * buffer.elem_size, buffer.elem_align);
			add_dependency(alloc_instr, *m_last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
			dest.record_allocation(dest.box, &alloc_instr); // TODO figure out how to make alloc_instr the "epoch" for any subsequent reads or writes.

			for(auto& source : memory.allocations) {
				// only copy those boxes to the new allocation that are still up-to-date in the old allocation
				// TODO investigate a garbage-collection heuristic that omits these copies if we do not expect them to be read from again on this memory
				const auto full_copy_box = GridBox<3>::intersect(dest.box, source.box);
				GridRegion<3> live_copy_region;
				for(const auto& [copy_box, location] : buffer.newest_data_location.get_region_values(full_copy_box)) {
					live_copy_region = GridRegion<3>::merge(live_copy_region, copy_box);
				}

				live_copy_region.scanByBoxes([&](const auto& copy_box) {
					assert(!copy_box.empty());

					const auto source_offset = grid_box_to_subrange(source.box).offset;
					const auto dest_offset = grid_box_to_subrange(source.box).offset;
					const auto [copy_offset, copy_range] = grid_box_to_subrange(copy_box);

					const auto copy_instr = &create<copy_instruction>(
					    source.aid, copy_offset - source_offset, dest.aid, copy_offset - dest_offset, buffer.dims, copy_range, buffer.elem_size);

					for(const auto& [_, dep_instr] : source.last_writers.get_region_values(copy_box)) { // TODO copy-pasta
						add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep, dependency_origin::dataflow);
					}
					source.record_read(copy_box, copy_instr);

					for(const auto& [_, front] : dest.access_fronts.get_region_values(copy_box)) { // TODO copy-pasta
						for(const auto dep_instr : front.front) {
							add_dependency(*copy_instr, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
						}
					}
					dest.record_write(copy_box, copy_instr);
				});
			}
		}

		// TODO consider keeping old allocations around until their box is written to in order to resolve "buffer-locking" anti-dependencies
		for(const auto free_aid : free_after_reallocation) {
			const auto& allocation = *std::find_if(memory.allocations.begin(), memory.allocations.end(), [&](const auto& a) { return a.aid == free_aid; });
			const auto free_instr = &create<free_instruction>(allocation.aid);
			for(const auto& [_, front] : allocation.access_fronts.get_region_values(allocation.box)) { // TODO copy-pasta
				for(const auto dep_instr : front.front) {
					add_dependency(*free_instr, *dep_instr, dependency_kind::true_dep, dependency_origin::dataflow);
				}
			}
		}
		// TODO garbage-collect allocations that are both stale and not written to?

		const auto end_retain_after_allocation = std::remove_if(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) {
			return std::any_of(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const auto aid) { return alloc.aid == aid; });
		});
		memory.allocations.erase(end_retain_after_allocation, memory.allocations.end());
	}

	// 3) create device <-> host or device <-> device copy instructions to satisfy all command-instruction reads

	std::unordered_map<std::pair<buffer_id, memory_id>, std::vector<GridRegion<3>>, utils::pair_hash> unsatisfied_reads;
	for(size_t cmd_insn_idx = 0; cmd_insn_idx < cmd_insns.size(); ++cmd_insn_idx) {
		auto& insn = cmd_insns[cmd_insn_idx];
		for(const auto& [bid, rw] : insn.rw_map) {
			auto& buffer = m_buffers.at(bid);
			GridRegion<3> unsatified_region;
			for(const auto& [box, location] : buffer.newest_data_location.get_region_values(rw.reads)) {
				if(!location.test(insn.mid)) { unsatified_region = GridRegion<3>::merge(unsatified_region, box); }
			}
			if(!unsatified_region.empty()) { unsatisfied_reads[{bid, insn.mid}].push_back(std::move(unsatified_region)); }
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

	// First, see if there are pending await-pushes for any of the unsatisfied read regions.
	for(auto& [bid_mid, disjoint_regions] : unsatisfied_reads) {
		const auto& [bid, mid] = bid_mid;
		auto& buffer = m_buffers.at(bid);
		for(auto& region : disjoint_regions) {
			// merge regions per transfer id to generate at most one instruction per host allocation and pending await-push command
			std::unordered_map<transfer_id, GridRegion<3>> transfer_regions;
			for(auto& [box, trid] : buffer.pending_await_pushes.get_region_values(region)) {
				if(trid != transfer_id()) {
					auto& tr_region = transfer_regions[trid]; // allow default-insert
					tr_region = GridRegion<3>::merge(tr_region, box);
				}
			}
			for(auto& [trid, tr_region] : transfer_regions) {
				auto& buffer_memory = buffer.memories[host_memory_id];

				for(auto& alloc : buffer_memory.allocations) {
					tr_region.scanByBoxes([&, trid = trid, bid = bid](const GridBox<3>& tr_box) {
						const auto recv_box = GridBox<3>::intersect(alloc.box, tr_box);
						if(recv_box.empty()) return;

						const auto [alloc_offset, alloc_range] = grid_box_to_subrange(alloc.box);
						const auto [recv_offset, recv_range] = grid_box_to_subrange(recv_box);
						const auto recv_instr = &create<recv_instruction>(
						    trid, bid, alloc.aid, buffer.dims, alloc_range, recv_offset - alloc_offset, recv_offset, recv_range, buffer.elem_size);

						// TODO the dependency logic here is duplicated from copy-instruction generation
						for(const auto& [_, front] : alloc.access_fronts.get_region_values(recv_box)) {
							for(const auto dep_instr : front.front) {
								add_dependency(*recv_instr, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
							}
						}
						add_dependency(*recv_instr, *m_last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
						alloc.record_write(recv_box, recv_instr);

						buffer.original_writers.update_region(recv_box, recv_instr);
					});
				}
				// TODO assert that the entire region has been consumed

				buffer.newest_data_location.update_region(tr_region, data_location().set(host_memory_id));
				buffer.pending_await_pushes.update_region(tr_region, transfer_id());
			}
		}
	}

	struct copy_template {
		buffer_id bid;
		memory_id source_mid;
		memory_id dest_mid;
		GridRegion<3> region;
	};

	std::vector<copy_template> pending_copies;
	for(auto& [bid_mid, disjoint_regions] : unsatisfied_reads) {
		const auto& [bid, dest_mid] = bid_mid;
		auto& buffer = m_buffers.at(bid);
		for(auto& region : disjoint_regions) {
			const auto region_sources = buffer.newest_data_location.get_region_values(region);
#ifndef NDEBUG
			for(const auto& [box, sources] : region_sources) {
				// TODO for convenience, we want to accept read-write access by the first kernel that ever touches a given buffer range (think a read-write
				//  kernel in a loop). However we still want to be able to detect the error condition of not having received a buffer region that was produced
				//  by some other kernel in the past.
				assert(sources.any() && "trying to read data that is neither found locally nor has been await-pushed before");
			}
#endif

			// try finding a common source for the entire region first to minimize instruction / synchronization complexity down the line
			const auto common_sources = std::accumulate(region_sources.begin(), region_sources.end(), data_location(),
			    [](const data_location common, const std::pair<GridBox<3>, data_location>& box_sources) { return common & box_sources.second; });

			if(common_sources.test(host_memory_id)) { // best case: we can copy all data from the host
				pending_copies.push_back({bid, host_memory_id, dest_mid, std::move(region)});
				continue;
			}

			// see if we can copy all data from a single device
			if(const auto common_device_sources = data_location(common_sources).reset(host_memory_id); common_device_sources.any()) {
				const auto copy_from = next_location(common_device_sources, dest_mid + 1);
				pending_copies.push_back({bid, copy_from, dest_mid, std::move(region)});
				continue;
			}

			// mix and match sources - there exists an optimal solution, but for now we just assemble source regions by picking the next device if any,
			// or the host as a fallback.
			std::unordered_map<memory_id, GridRegion<3>> combined_source_regions;
			for(const auto& [box, sources] : region_sources) {
				memory_id copy_from;
				if(const auto device_sources = data_location(sources).reset(host_memory_id); device_sources.any()) {
					copy_from = next_location(device_sources, dest_mid + 1);
				} else {
					copy_from = host_memory_id;
				}
				auto& copy_region = combined_source_regions[copy_from];
				copy_region = GridRegion<3>::merge(copy_region, box);
			}
			for(auto& [copy_from, copy_region] : combined_source_regions) {
				pending_copies.push_back({bid, copy_from, dest_mid, std::move(copy_region)});
			}
		}

		for(auto& copy : pending_copies) {
			assert(copy.dest_mid != copy.source_mid);
			auto& buffer = m_buffers.at(copy.bid);
			copy.region.scanByBoxes([&](const GridBox<3>& box) {
				for(auto& source : buffer.memories[copy.source_mid].allocations) {
					const auto read_box = GridBox<3>::intersect(box, source.box);
					if(read_box.empty()) continue;

					for(auto& dest : buffer.memories[copy.dest_mid].allocations) {
						const auto copy_box = GridBox<3>::intersect(read_box, dest.box);
						if(copy_box.empty()) continue;

						const auto [source_offset, source_range] = grid_box_to_subrange(source.box);
						const auto [dest_offset, dest_range] = grid_box_to_subrange(dest.box);
						const auto [copy_offset, copy_range] = grid_box_to_subrange(copy_box);

						const auto copy_instr = &create<copy_instruction>(
						    source.aid, copy_offset - source_offset, dest.aid, copy_offset - dest_offset, buffer.dims, copy_range, buffer.elem_size);

						for(const auto& [_, last_writer_instr] : source.last_writers.get_region_values(copy.region)) {
							add_dependency(*copy_instr, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
						}
						source.record_read(copy.region, copy_instr);

						for(const auto& [_, front] : dest.access_fronts.get_region_values(copy.region)) { // TODO copy-pasta
							for(const auto dep_instr : front.front) {
								add_dependency(*copy_instr, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
							}
						}
						dest.record_write(copy.region, copy_instr);
						for(auto& [box, location] : buffer.newest_data_location.get_region_values(copy.region)) {
							buffer.newest_data_location.update_region(box, data_location(location).set(copy.dest_mid));
						}
					}
				}
			});
		}
	}

	// 4) create the actual command instructions

	utils::match(
	    cmd,
	    [&](const execution_command& xcmd) {
		    const auto& tsk = *m_tm.get_task(xcmd.get_tid());
		    for(auto& in : cmd_insns) {
			    if(tsk.get_execution_target() == execution_target::device) {
				    assert(in.execution_sr.range.size() > 0);
				    assert(in.mid != host_memory_id);
				    in.instruction = &create<device_kernel_instruction>(in.did, xcmd.get_cid(), in.execution_sr, in.rw_map);
			    } else {
				    assert(tsk.get_execution_target() == execution_target::host);
				    assert(in.mid == host_memory_id);
				    in.instruction = &create<host_kernel_instruction>(xcmd.get_cid(), in.execution_sr, in.rw_map, in.se_map, in.cgid);
			    }
		    }
	    },
	    [&](const push_command& pcmd) {
		    for(auto& insn : cmd_insns) {
			    assert(insn.rw_map.size() == 1);
			    auto [bid, rw] = *insn.rw_map.begin();
			    assert(!rw.reads.empty());

			    const auto& buffer = m_buffers.at(bid);
			    for(auto& alloc : buffer.memories[host_memory_id].allocations) {
				    const auto send_box = GridBox<3>::intersect(alloc.box, subrange_to_grid_box(pcmd.get_range()));
				    if(!send_box.empty()) {
					    const auto [send_offset, send_range] = grid_box_to_subrange(send_box);
					    const auto [alloc_offset, alloc_range] = grid_box_to_subrange(alloc.box);
					    insn.instruction = &create<send_instruction>(pcmd.get_cid(), pcmd.get_target(), bid, alloc.aid, buffer.dims, alloc_range,
					        send_offset - alloc_offset, send_offset, send_range, buffer.elem_size);
					    // ^ TODO FIXME I'm generating (and overwriting) more than one instruction here! The "split" should probably be decided ahead of time?
				    }
			    }
		    }
	    },
	    [&](const horizon_command& hcmd) {
		    for(auto& insn : cmd_insns) {
			    insn.instruction = &create<horizon_instruction>(hcmd.get_cid());
		    }
	    },
	    [&](const epoch_command& ecmd) {
		    for(auto& insn : cmd_insns) {
			    insn.instruction = &create<epoch_instruction>(ecmd.get_cid());
		    }
	    },
	    [](const auto& /* unhandled */) {
		    assert(!"unhandled command type");
		    std::abort();
	    });

	// 5) compute dependencies between command instructions and previous copy, allocation, and command (!) instructions

	// TODO this will not work correctly for oversubscription
	//	 - read-all + write-1:1 cannot be oversubscribed at all, chunks would need a global read->write barrier (how would the kernel even look like?)
	//	 - oversubscribed host tasks would need dependencies between their chunks based on side effects and collective groups
	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, rw] : insn.rw_map) {
			for(auto& alloc : m_buffers.at(bid).memories[insn.mid].allocations) {
				for(const auto& [_, last_writer_instr] : alloc.last_writers.get_region_values(rw.reads)) {
					add_dependency(*insn.instruction, *last_writer_instr, dependency_kind::true_dep, dependency_origin::dataflow);
				}
			}
		}
		for(const auto& [bid, rw] : insn.rw_map) {
			for(auto& alloc : m_buffers.at(bid).memories[insn.mid].allocations) {
				for(const auto& [_, front] : alloc.access_fronts.get_region_values(rw.writes)) {
					for(const auto dep_instr : front.front) {
						add_dependency(*insn.instruction, *dep_instr, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}
			}
		}
		for(const auto& [hoid, order] : insn.se_map) {
			assert(insn.mid == host_memory_id);
			if(const auto last_side_effect = m_host_objects.at(hoid).last_side_effect) {
				add_dependency(*insn.instruction, *last_side_effect, dependency_kind::true_dep, dependency_origin::dataflow);
			}
		}
		if(insn.cgid != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(insn.mid == host_memory_id);
			auto& group = m_collective_groups[insn.cgid]; // allow default-insertion since we do not register CGs explicitly
			if(group.last_host_task) {
				add_dependency(*insn.instruction, *group.last_host_task, dependency_kind::true_dep, dependency_origin::collective_group_serialization);
			}
		}
	}

	// 6) update data locations and last writers resulting from command instructions

	for(const auto& insn : cmd_insns) {
		for(const auto& [bid, rw] : insn.rw_map) {
			assert(insn.instruction != nullptr);
			auto& buffer = m_buffers.at(bid);
			buffer.newest_data_location.update_region(rw.writes, data_location().set(insn.mid));
			buffer.original_writers.update_region(rw.writes, insn.instruction);

			for(auto& alloc : buffer.memories[insn.mid].allocations) {
				rw.writes.scanByBoxes([&](const auto& box) {
					const auto write_box = GridBox<3>::intersect(alloc.box, box);
					if(!write_box.empty()) { alloc.record_write(write_box, insn.instruction); }
				});
			}
		}
		for(const auto& [hoid, order] : insn.se_map) {
			assert(insn.mid == host_memory_id);
			m_host_objects.at(hoid).last_side_effect = insn.instruction;
		}
		if(insn.cgid != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(insn.mid == host_memory_id);
			m_collective_groups.at(insn.cgid).last_host_task = insn.instruction;
		}
	}

	// 7) insert epoch and horizon dependencies, apply epochs

	for(auto& insn : cmd_insns) {
		if(isa<horizon_command>(&cmd) || isa<epoch_command>(&cmd)) {
			collapse_execution_front_to(insn.instruction);

			if(isa<epoch_command>(&cmd)) {
				apply_epoch(insn.instruction);
				m_last_horizon = nullptr;
			} else {
				if(m_last_horizon) { apply_epoch(m_last_horizon); }
				m_last_horizon = insn.instruction;
			}
		} else {
			// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
			// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
			const auto deps = insn.instruction->get_dependencies();
			if(std::none_of(deps.begin(), deps.end(), [](const instruction::dependency& dep) { return dep.kind == dependency_kind::true_dep; })) {
				add_dependency(*insn.instruction, *m_last_epoch, dependency_kind::true_dep, dependency_origin::last_epoch);
			}
		}
	}
}

} // namespace celerity::detail