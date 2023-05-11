#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "allscale/api/user/data/grid.h"
#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <numeric>

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, std::vector<device_info> devices) : m_tm(tm), m_devices(std::move(devices)) {
	assert(devices.size() + 1 <= max_memories);
	const auto initial_epoch = &create<epoch_instruction>(task_id(0 /* or so we assume */));
	initial_epoch->set_debug_info(epoch_instruction_debug_info(command_id(0 /* or so we assume */)));
	m_last_epoch = initial_epoch;
}

void instruction_graph_generator::register_buffer(buffer_id bid, int dims, range<3> range, const size_t elem_size, const size_t elem_align) {
	[[maybe_unused]] const auto [_, inserted] =
	    m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(dims, range, elem_size, elem_align, m_devices.size() + 1));
	assert(inserted);
}

void instruction_graph_generator::set_buffer_debug_name(const buffer_id bid, std::string name) {
	if(const auto it = m_buffers.find(bid); it != m_buffers.end()) { it->second.debug_name = std::move(name); }
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


instruction_graph_generator::platform instruction_graph_generator::get_memory_platform(const memory_id mid) const {
	return mid == 0 ? platform::host : m_devices[mid - 1].platform;
}

instruction_port instruction_graph_generator::get_allocation_port(const memory_id mid) const {
	switch(get_memory_platform(mid)) {
	case platform::host: return instruction_port::host;
	case platform::sycl: return instruction_port::sycl_async;
	}
}

instruction_port instruction_graph_generator::get_copy_port(const memory_id from_mid, const memory_id to_mid) const {
	const auto from = get_memory_platform(from_mid);
	const auto to = get_memory_platform(to_mid);
	if(from == platform::host && to == platform::host) {
		return instruction_port::host;
	} else if((from == platform::sycl && to == platform::sycl) || (from == platform::host && to == platform::sycl)
	          || (from == platform::sycl && to == platform::host)) {
		return instruction_port::sycl_async;
	} else {
		assert(!"cannot copy between unrelated platforms");
		std::abort();
	}
}

instruction_port instruction_graph_generator::get_kernel_launch_port(const device_id did) const {
	switch(m_devices[did].platform) {
	case platform::host: assert(!"device cannot have host platform"); std::abort();
	case platform::sycl: return instruction_port::sycl_async;
	}
}

void instruction_graph_generator::allocate_contiguously(const buffer_id bid, const memory_id mid, const std::vector<GridBox<3>>& boxes) {
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

	auto new_allocations = std::move(contiguous_after_reallocation).into_vector();
	const auto last_new_allocation = std::remove_if(new_allocations.begin(), new_allocations.end(),
	    [&](auto& box) { return std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == box; }); });
	new_allocations.erase(last_new_allocation, new_allocations.end());

	// region-merge adjacent boxes that need to be allocated (usually for oversubscriptions). This should not introduce problematic synchronization points since


	// TODO but it does introduce synchronization between producers on the resize-copies, which we want to avoid. To resolve this, allocate the fused boxes as
	// before, but use the non-fused boxes as copy destinations.
	allscale::api::user::data::detail::box_fuser<3>().apply(new_allocations);

	// TODO don't copy data that will be overwritten (have an additional GridRegion<3> to_be_overwritten parameter)

	for(const auto& dest_box : new_allocations) {
		auto& dest = memory.allocations.emplace_back(m_next_aid++, dest_box, buffer.range);
		const auto alloc_instr = &create<alloc_instruction>(get_allocation_port(mid), dest.aid, mid, dest.box.area() * buffer.elem_size, buffer.elem_align);
		alloc_instr->set_debug_info(alloc_instruction_debug_info(bid, buffer.debug_name, dest_box));
		add_dependency(*alloc_instr, *m_last_epoch, dependency_kind::true_dep);
		dest.record_write(dest.box, alloc_instr); // TODO figure out how to make alloc_instr the "epoch" for any subsequent reads or writes.

		for(auto& source : memory.allocations) {
			if(std::find_if(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const allocation_id aid) { return aid == source.aid; })
			    == free_after_reallocation.end()) {
				// we modify memory.allocations in-place, so we need to be careful not to attempt copying from a new allocation to itself.
				// Since we don't have overlapping allocations, any copy source must currently be one that will be freed after reallocation.
				continue;
			}

			// only copy those boxes to the new allocation that are still up-to-date in the old allocation
			// TODO investigate a garbage-collection heuristic that omits these copies if we do not expect them to be read from again on this memory
			const auto full_copy_box = GridBox<3>::intersect(dest.box, source.box);
			GridRegion<3> live_copy_region;
			for(const auto& [copy_box, location] : buffer.newest_data_location.get_region_values(full_copy_box)) {
				if(location.test(mid)) { live_copy_region = GridRegion<3>::merge(live_copy_region, copy_box); }
			}

			live_copy_region.scanByBoxes([&](const auto& copy_box) {
				assert(!copy_box.empty());

				const auto [source_offset, source_range] = grid_box_to_subrange(source.box);
				const auto [dest_offset, dest_range] = grid_box_to_subrange(dest_box);
				const auto [copy_offset, copy_range] = grid_box_to_subrange(copy_box);

				// TODO to avoid introducing a synchronization point on oversubscription, split into multiple copies if that will allow unimpeded
				// oversubscribed-producer to oversubscribed-consumer data flow.

				const auto copy_instr = &create<copy_instruction>(get_allocation_port(mid), buffer.dims, mid, source.aid, source_range,
				    copy_offset - source_offset, mid, dest.aid, dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);
				copy_instr->set_debug_info(copy_instruction_debug_info(copy_instruction_debug_info::copy_origin::resize, bid, buffer.debug_name, copy_box));

				for(const auto& [_, dep_instr] : source.last_writers.get_region_values(copy_box)) { // TODO copy-pasta
					assert(dep_instr != nullptr);
					add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
				}
				source.record_read(copy_box, copy_instr);

				for(const auto& [_, front] : dest.access_fronts.get_region_values(copy_box)) { // TODO copy-pasta
					for(const auto dep_instr : front.front) {
						add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
					}
				}
				dest.record_write(copy_box, copy_instr);
			});
		}
	}

	// TODO consider keeping old allocations around until their box is written to in order to resolve "buffer-locking" anti-dependencies
	for(const auto free_aid : free_after_reallocation) {
		const auto& allocation = *std::find_if(memory.allocations.begin(), memory.allocations.end(), [&](const auto& a) { return a.aid == free_aid; });
		const auto free_instr = &create<free_instruction>(get_allocation_port(mid), allocation.aid);
		free_instr->set_debug_info(
		    free_instruction_debug_info(mid, allocation.box.area() * buffer.elem_size, buffer.elem_align, bid, buffer.debug_name, allocation.box));
		for(const auto& [_, front] : allocation.access_fronts.get_region_values(allocation.box)) { // TODO copy-pasta
			for(const auto dep_instr : front.front) {
				add_dependency(*free_instr, *dep_instr, dependency_kind::true_dep);
			}
		}
	}

	// TODO garbage-collect allocations that are both stale and not written to? We cannot re-fetch buffer subranges from their original producer without some
	// sort of inter-node pull semantics if the GC turned out to be a misprediction, but we can swap allocations to the host when we run out of device memory.
	// Basically we would annotate each allocation with an last-used value to implement LRU semantics.

	const auto end_retain_after_allocation = std::remove_if(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) {
		return std::any_of(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const auto aid) { return alloc.aid == aid; });
	});
	memory.allocations.erase(end_retain_after_allocation, memory.allocations.end());
}


void instruction_graph_generator::satisfy_read_requirements(const buffer_id bid, const std::vector<std::pair<memory_id, GridRegion<3>>>& reads) {
	auto& buffer = m_buffers.at(bid);

	std::unordered_map<memory_id, std::vector<GridRegion<3>>> unsatisfied_reads;
	for(const auto& [mid, read_region] : reads) {
		GridRegion<3> unsatified_region;
		for(const auto& [box, location] : buffer.newest_data_location.get_region_values(read_region)) {
			if(!location.test(mid)) { unsatified_region = GridRegion<3>::merge(unsatified_region, box); }
		}
		if(!unsatified_region.empty()) { unsatisfied_reads[mid].push_back(std::move(unsatified_region)); }
	}

	// transform vectors of potentially-overlapping unsatisfied regions into disjoint regions
	for(auto& [mid, regions] : unsatisfied_reads) {
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
	for(auto& [mid, disjoint_regions] : unsatisfied_reads) {
		auto& buffer = m_buffers.at(bid);
		auto& memory = buffer.memories[mid];

		for(auto& unsatisfied_region : disjoint_regions) {
			// merge regions per transfer id to generate at most one instruction per host allocation and pending await-push command
			std::unordered_map<transfer_id, GridRegion<3>> transfer_regions;
			for(auto& [box, trid] : buffer.pending_await_pushes.get_region_values(unsatisfied_region)) {
				if(trid == transfer_id()) continue;

				auto& tr_region = transfer_regions[trid]; // allow default-insert
				tr_region = GridRegion<3>::merge(tr_region, box);
			}
			for(auto& [trid, accepted_transfer_region] : transfer_regions) {
				for(auto& alloc : memory.allocations) {
					const auto accepted_alloc_region = GridRegion<3>::intersect(alloc.box, accepted_transfer_region);
					accepted_alloc_region.scanByBoxes([&, mid = mid, trid = trid](const GridBox<3>& tr_box) {
						const auto recv_box = GridBox<3>::intersect(alloc.box, tr_box);
						if(recv_box.empty()) return;

						const auto [alloc_offset, alloc_range] = grid_box_to_subrange(alloc.box);
						const auto [recv_offset, recv_range] = grid_box_to_subrange(recv_box);
						const auto recv_instr = &create<recv_instruction>(
						    trid, mid, alloc.aid, buffer.dims, alloc_range, recv_offset - alloc_offset, recv_offset, recv_range, buffer.elem_size);
						command_id await_push_cid = 1234; // TODO where do we get this from without changing non-debug code paths too much?
						recv_instr->set_debug_info(recv_instruction_debug_info(await_push_cid, bid, buffer.debug_name));

						// TODO the dependency logic here is duplicated from copy-instruction generation
						for(const auto& [_, front] : alloc.access_fronts.get_region_values(recv_box)) {
							for(const auto dep_instr : front.front) {
								add_dependency(*recv_instr, *dep_instr, dependency_kind::true_dep);
							}
						}
						add_dependency(*recv_instr, *m_last_epoch, dependency_kind::true_dep);
						alloc.record_write(recv_box, recv_instr);

						buffer.original_writers.update_region(recv_box, recv_instr);
					});
				}
				// TODO assert that the entire region is consumed (... eventually?)

				// TODO this always transfers to a single memory, which is suboptimal. Find a way to implement "device broadcast" from a single receive buffer.
				// This will probably require explicit handling of the receive buffer inside the IDAG.
				//
				// How to realize this in the presence of CUDA-aware MPI? We want to have RDMA MPI_Recv to device buffers as the happy path, and a copy from
				// receive buffer as fallback - but only in the absence of a broadcast condition.
				// 	 - annotate recv_instructions with the desired allocation to RDMA to (if any)
				//   - have a conditional copy_instruction following it, in case the RDMA fails (or should the buffer_transfer_manager dispatch that copy?)
				//   - in case of a RDMA failure, or when we want to broadcast, we need a release_transfer_instruction to hand the transfer_allocation_id back
				// 	   to the buffer_transfer_manager (again kinda conditional on whether we did an RDMA receive or not)
				buffer.newest_data_location.update_region(accepted_transfer_region, data_location().set(mid));
				buffer.pending_await_pushes.update_region(accepted_transfer_region, transfer_id());

				unsatisfied_region = GridRegion<3>::difference(unsatisfied_region, accepted_transfer_region);
			}
		}

		// remove regions that are fully satisfied by incoming transfers
		disjoint_regions.erase(std::remove_if(disjoint_regions.begin(), disjoint_regions.end(), std::mem_fn(&GridRegion<3>::empty)), disjoint_regions.end());
	}

	struct copy_template {
		memory_id source_mid;
		memory_id dest_mid;
		GridRegion<3> region;
	};

	std::vector<copy_template> pending_copies;
	for(auto& [dest_mid, disjoint_regions] : unsatisfied_reads) {
		if(disjoint_regions.empty()) continue; // if fully satisfied by incoming transfers

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
				pending_copies.push_back({host_memory_id, dest_mid, std::move(region)});
				continue;
			}

			// see if we can copy all data from a single device
			if(const auto common_device_sources = data_location(common_sources).reset(host_memory_id); common_device_sources.any()) {
				const auto copy_from = next_location(common_device_sources, dest_mid + 1);
				pending_copies.push_back({copy_from, dest_mid, std::move(region)});
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
					copy_from = host_memory_id; // heuristic: avoid copies from host because they can't be DMA'd
				}
				auto& copy_region = combined_source_regions[copy_from];
				copy_region = GridRegion<3>::merge(copy_region, box);
			}
			for(auto& [copy_from, copy_region] : combined_source_regions) {
				pending_copies.push_back({copy_from, dest_mid, std::move(copy_region)});
			}
		}
	}

	for(auto& copy : pending_copies) {
		assert(copy.dest_mid != copy.source_mid);
		const auto copy_port = get_copy_port(copy.source_mid, copy.dest_mid);
		auto& buffer = m_buffers.at(bid);
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

					const auto copy_instr = &create<copy_instruction>(copy_port, buffer.dims, copy.source_mid, source.aid, source_range,
					    copy_offset - source_offset, copy.dest_mid, dest.aid, dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);
					copy_instr->set_debug_info(
					    copy_instruction_debug_info(copy_instruction_debug_info::copy_origin::coherence, bid, buffer.debug_name, copy_box));

					for(const auto& [_, last_writer_instr] : source.last_writers.get_region_values(copy.region)) {
						assert(last_writer_instr != nullptr);
						add_dependency(*copy_instr, *last_writer_instr, dependency_kind::true_dep);
					}
					source.record_read(copy.region, copy_instr);

					for(const auto& [_, front] : dest.access_fronts.get_region_values(copy.region)) { // TODO copy-pasta
						for(const auto dep_instr : front.front) {
							add_dependency(*copy_instr, *dep_instr, dependency_kind::true_dep);
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


std::vector<copy_instruction*> instruction_graph_generator::linearize_buffer_subrange(
    const buffer_id bid, const GridBox<3>& box, const memory_id out_mid, const allocation_id out_aid) {
	auto& buffer = m_buffers.at(bid);

	const auto box_sources = buffer.newest_data_location.get_region_values(box);

#ifndef NDEBUG
	for(const auto& [box, sources] : box_sources) {
		// TODO for convenience, we want to accept read-write access by the first kernel that ever touches a given buffer range (think a read-write
		//  kernel in a loop). However we still want to be able to detect the error condition of not having received a buffer region that was produced
		//  by some other kernel in the past.
		assert(sources.any() && "trying to read data that is neither found locally nor has been await-pushed before");
	}
	for(auto& [box, await_push] : buffer.pending_await_pushes.get_region_values(box)) {
		assert(await_push == 0 && "attempting to linearize a subrange with uncommitted await-pushes");
	}
#endif

	// There's no danger of multi-hop copies here, but we still stage copies to potentially merge boxes present on multiple memories below
	std::unordered_map<memory_id, GridRegion<3>> pending_copies;

	// try finding a common source for the entire region first to minimize instruction / synchronization complexity down the line
	const auto common_sources = std::accumulate(box_sources.begin(), box_sources.end(), data_location(),
	    [](const data_location common, const std::pair<GridBox<3>, data_location>& box_sources) { return common & box_sources.second; });
	if(const auto common_device_sources = data_location(common_sources).reset(host_memory_id); common_device_sources.any()) {
		// best case: we can copy all data from a single device
		const auto source_mid = next_location(common_device_sources, host_memory_id + 1);
		auto& copy_region = pending_copies[source_mid];
		copy_region = GridRegion<3>::merge(copy_region, box);
	} else {
		// mix and match sources - there exists an optimal solution, but for now we just assemble source regions by picking the next device if any,
		// or the host as a fallback.
		for(const auto& [box, sources] : box_sources) {
			const memory_id source_mid = next_location(sources, host_memory_id + 1); // picks host_memory last (which usually can't leverage DMA)
			auto& copy_region = pending_copies[source_mid];
			copy_region = GridRegion<3>::merge(copy_region, box);
		}
	}

	std::vector<copy_instruction*> copy_instrs;
	for(const auto& [source_mid, region] : pending_copies) {
		const auto copy_port = get_copy_port(source_mid, out_mid);
		region.scanByBoxes([&, source_mid = source_mid](const GridBox<3>& source_box) {
			for(auto& source : buffer.memories[source_mid].allocations) {
				const auto copy_box = GridBox<3>::intersect(source.box, source_box);
				if(copy_box.empty()) continue;

				const auto [source_offset, source_range] = grid_box_to_subrange(source.box);
				const auto [copy_offset, copy_range] = grid_box_to_subrange(copy_box);
				const auto [dest_offset, dest_range] = grid_box_to_subrange(box);
				const auto copy_instr = &create<copy_instruction>(copy_port, buffer.dims, source_mid, source.aid, source_range, copy_offset - source_offset,
				    out_mid, out_aid, dest_range, dest_offset - source_offset, copy_range, buffer.elem_size);
				copy_instr->set_debug_info(copy_instruction_debug_info(copy_instruction_debug_info::copy_origin::linearize, bid, buffer.debug_name, copy_box));

				// TODO copy-pasta
				for(const auto& [_, last_writer_instr] : source.last_writers.get_region_values(copy_box)) {
					assert(last_writer_instr != nullptr);
					add_dependency(*copy_instr, *last_writer_instr, dependency_kind::true_dep);
				}
				source.record_read(copy_box, copy_instr);

				copy_instrs.push_back(copy_instr);
			}
		});
	}

	return copy_instrs;
}


int instruction_graph_generator::create_pilot_message(buffer_id bid, const GridBox<3>& box) {
	int tag = m_next_p2p_tag++;
	m_pilots.push_back(pilot_message{tag, bid, box});
	return tag;
}


// TODO HACK we're just pulling in the splitting logic from distributed_graph_generator here
std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);


void instruction_graph_generator::compile_execution_command(const execution_command& ecmd) {
	const auto& tsk = dynamic_cast<const command_group_task&>(*m_tm.get_task(ecmd.get_tid()));

	struct partial_instruction {
		subrange<3> execution_sr;
		device_id did = -1;
		memory_id mid = host_memory_id;
		buffer_read_write_map rw_map;
		side_effect_map se_map;
		instruction* instruction = nullptr;
	};
	std::vector<partial_instruction> cmd_instrs;

	const auto command_sr = ecmd.get_execution_range();
	if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device) {
		const auto split = tsk.get_hint<experimental::hints::tiled_split>() ? split_2d : split_1d;
		const auto oversubscribe = tsk.get_hint<experimental::hints::oversubscribe>();
		const auto num_sub_chunks_per_device = oversubscribe ? oversubscribe->get_factor() : 1;

		const auto device_chunks = split(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_devices.size());
		for(device_id did = 0; did < m_devices.size() && did < device_chunks.size(); ++did) {
			// subdivide recursively so that in case of a 2D split, we still produce 2D tiles instead of a row-subset
			const auto this_device_sub_chunks = split(device_chunks[did], tsk.get_granularity(), num_sub_chunks_per_device);
			for(const auto& sub_chunk : this_device_sub_chunks) {
				auto& insn = cmd_instrs.emplace_back();
				insn.execution_sr = subrange<3>(sub_chunk.offset, sub_chunk.range);
				insn.did = did;
				insn.mid = device_to_memory_id(did);
			}
		}
	} else {
		// TODO oversubscribe distributed host tasks (but not if they have side effects)
		auto& insn = cmd_instrs.emplace_back();
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

	std::unordered_map<std::pair<buffer_id, memory_id>, GridRegion<3>, utils::pair_hash> invalidations;
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(auto& insn : cmd_instrs) {
			reads_writes rw;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
				if(access::mode_traits::is_consumer(mode)) { rw.reads = GridRegion<3>::merge(rw.reads, req); }
				if(access::mode_traits::is_producer(mode)) { rw.writes = GridRegion<3>::merge(rw.writes, req); }
				if(!access::mode_traits::is_consumer(mode)) {
					auto& region = invalidations[{bid, insn.mid}]; // allow default-insert
					region = GridRegion<3>::merge(region, req);
				}
			}
			rw.contiguous_boxes = bam.get_required_contiguous_boxes(bid, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
			if(!rw.empty()) { insn.rw_map.emplace(bid, std::move(rw)); }
		}
	}

	if(!tsk.get_side_effect_map().empty()) {
		assert(cmd_instrs.size() == 1); // split instructions for host tasks with side effects would race
		assert(cmd_instrs[0].mid == host_memory_id);
		cmd_instrs[0].se_map = tsk.get_side_effect_map();
	}

	// Invalidate any buffer region that will immediately be overwritten (and not also read) to avoid preserving it across buffer resizes (and to catch
	// read-write access conflicts, TODO)
	for(auto& [bid_mid, region] : invalidations) {
		const auto& [bid, mid] = bid_mid;
		auto& buffer = m_buffers.at(bid);
		for(auto& [box, location] : buffer.newest_data_location.get_region_values(region)) {
			buffer.newest_data_location.update_region(box, data_location(location).reset(mid));
		}
	}

	// 2) allocate memory

	std::unordered_map<std::pair<buffer_id, memory_id>, std::vector<GridBox<3>>, utils::pair_hash> not_contiguously_allocated_boxes;
	for(const auto& insn : cmd_instrs) {
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
		allocate_contiguously(bid, mid, boxes);
	}

	// 3) create device <-> host or device <-> device copy instructions to satisfy all command-instruction reads

	std::unordered_map<buffer_id, std::vector<std::pair<memory_id, GridRegion<3>>>> buffer_reads;
	for(size_t cmd_insn_idx = 0; cmd_insn_idx < cmd_instrs.size(); ++cmd_insn_idx) {
		auto& insn = cmd_instrs[cmd_insn_idx];
		for(const auto& [bid, rw] : insn.rw_map) {
			if(!rw.reads.empty()) { buffer_reads[bid].emplace_back(insn.mid, rw.reads); }
		}
	}

	// 4) create the actual command instructions

	for(const auto& [bid, reads] : buffer_reads) {
		satisfy_read_requirements(bid, reads);
	}

	for(auto& instr : cmd_instrs) {
		access_allocation_map allocation_map(bam.get_num_accesses());
		std::vector<buffer_allocation_info> allocation_buffer_map(bam.get_num_accesses()); // TODO debug only

		for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
			const auto [bid, mode] = bam.get_nth_access(i);
			const auto accessed_box = bam.get_requirements_for_nth_access(i, tsk.get_dimensions(), instr.execution_sr, tsk.get_global_size());
			const auto& buffer = m_buffers.at(bid);
			const auto& allocations = buffer.memories[instr.mid].allocations;
			const auto allocation_it = std::find_if(
			    allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& alloc) { return alloc.box.covers(accessed_box); });
			assert(allocation_it != allocations.end());
			const auto& alloc = *allocation_it;
			const auto [access_offset, access_range] = grid_box_to_subrange(accessed_box);
			const auto [alloc_offset, alloc_range] = grid_box_to_subrange(alloc.box);
			allocation_map[i] = access_allocation{alloc.aid, alloc_range, access_offset - alloc_offset};
			allocation_buffer_map[i] = buffer_allocation_info{bid, buffer.debug_name, alloc.box};
		}

		kernel_instruction* kernel_instr;
		if(tsk.get_execution_target() == execution_target::device) {
			assert(instr.execution_sr.range.size() > 0);
			assert(instr.mid != host_memory_id);
			// TODO how do I know it's a SYCL kernel and not a CUDA kernel?
			kernel_instr = &create<sycl_kernel_instruction>(instr.did, tsk.get_launcher<sycl_kernel_launcher>(), instr.execution_sr, std::move(allocation_map));
		} else {
			assert(tsk.get_execution_target() == execution_target::host);
			assert(instr.mid == host_memory_id);
			kernel_instr =
			    &create<host_kernel_instruction>(tsk.get_launcher<host_task_launcher>(), instr.execution_sr, tsk.get_global_size(), std::move(allocation_map));
		}

		kernel_instr->set_debug_info(kernel_instruction_debug_info(ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), std::move(allocation_buffer_map)));
		instr.instruction = kernel_instr;
	}
	// 5) compute dependencies between command instructions and previous copy, allocation, and command (!) instructions

	// TODO this will not work correctly for oversubscription
	//	 - read-all + write-1:1 cannot be oversubscribed at all, chunks would need a global read->write barrier (how would the kernel even look like?)
	//	 - oversubscribed host tasks would need dependencies between their chunks based on side effects and collective groups
	for(const auto& instr : cmd_instrs) {
		for(const auto& [bid, rw] : instr.rw_map) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[instr.mid];
			for(auto& alloc : memory.allocations) {
				const auto reads_from_alloc = GridRegion<3>::intersect(rw.reads, alloc.box);
				for(const auto& [_, last_writer_instr] : alloc.last_writers.get_region_values(reads_from_alloc)) {
					assert(last_writer_instr != nullptr);
					add_dependency(*instr.instruction, *last_writer_instr, dependency_kind::true_dep);
				}
			}
		}
		for(const auto& [bid, rw] : instr.rw_map) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[instr.mid];
			for(auto& alloc : memory.allocations) {
				const auto writes_to_alloc = GridRegion<3>::intersect(rw.writes, alloc.box);
				for(const auto& [_, front] : alloc.access_fronts.get_region_values(writes_to_alloc)) {
					for(const auto dep_instr : front.front) {
						add_dependency(*instr.instruction, *dep_instr, dependency_kind::true_dep);
					}
				}
			}
		}
		for(const auto& [hoid, order] : instr.se_map) {
			assert(instr.mid == host_memory_id);
			if(const auto last_side_effect = m_host_objects.at(hoid).last_side_effect) {
				add_dependency(*instr.instruction, *last_side_effect, dependency_kind::true_dep);
			}
		}
		if(tsk.get_collective_group_id() != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(instr.mid == host_memory_id);
			auto& group = m_collective_groups[tsk.get_collective_group_id()]; // allow default-insertion since we do not register CGs explicitly
			if(group.last_host_task) { add_dependency(*instr.instruction, *group.last_host_task, dependency_kind::true_dep); }
		}
	}

	// 6) update data locations and last writers resulting from command instructions

	for(const auto& instr : cmd_instrs) {
		for(const auto& [bid, rw] : instr.rw_map) {
			assert(instr.instruction != nullptr);
			auto& buffer = m_buffers.at(bid);
			buffer.newest_data_location.update_region(rw.writes, data_location().set(instr.mid));
			buffer.original_writers.update_region(rw.writes, instr.instruction);

			for(auto& alloc : buffer.memories[instr.mid].allocations) {
				rw.writes.scanByBoxes([&](const auto& box) {
					const auto write_box = GridBox<3>::intersect(alloc.box, box);
					if(!write_box.empty()) { alloc.record_write(write_box, instr.instruction); }
				});
			}
		}
		for(const auto& [hoid, order] : instr.se_map) {
			assert(instr.mid == host_memory_id);
			m_host_objects.at(hoid).last_side_effect = instr.instruction;
		}
		if(tsk.get_collective_group_id() != collective_group_id(0) /* 0 means "no collective group association" */) {
			assert(instr.mid == host_memory_id);
			m_collective_groups.at(tsk.get_collective_group_id()).last_host_task = instr.instruction;
		}
	}

	// 7) insert epoch and horizon dependencies, apply epochs

	for(auto& instr : cmd_instrs) {
		// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
		// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
		const auto deps = instr.instruction->get_dependencies();
		if(std::none_of(deps.begin(), deps.end(), [](const instruction::dependency& dep) { return dep.kind == dependency_kind::true_dep; })) {
			add_dependency(*instr.instruction, *m_last_epoch, dependency_kind::true_dep);
		}
	}
}


void instruction_graph_generator::compile_push_command(const push_command& pcmd) {
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
		region.scanByBoxes([&](const GridBox<3>& box) {
			const auto bytes = box.area() * buffer.elem_size;

			// this allocation is not associated with a buffer (and thus not tracked in m_buffers), so we add all dependencies immediately
			const auto alloc_instr = &create<alloc_instruction>(instruction_port::host, m_next_aid++, host_memory_id, bytes, buffer.elem_align);
			alloc_instr->set_debug_info(alloc_instruction_debug_info(/* alloc_origin::send */));
			add_dependency(*alloc_instr, *m_last_epoch, dependency_kind::true_dep);

			const auto copy_instrs = linearize_buffer_subrange(bid, box, host_memory_id, alloc_instr->get_allocation_id());
			for(const auto copy_instr : copy_instrs) {
				add_dependency(*copy_instr, *alloc_instr, dependency_kind::true_dep);
			}
			const int tag = create_pilot_message(bid, box);
			const auto send_instr = &create<send_instruction>(pcmd.get_target(), tag, alloc_instr->get_allocation_id(), bytes);
			send_instr->set_debug_info(send_instruction_debug_info(pcmd.get_cid(), bid, buffer.debug_name, box));
			for(const auto copy_instr : copy_instrs) {
				add_dependency(*send_instr, *copy_instr, dependency_kind::true_dep);
			}
			const auto free_instr = &create<free_instruction>(instruction_port::host, alloc_instr->get_allocation_id());
			free_instr->set_debug_info(free_instruction_debug_info(host_memory_id, bytes, buffer.elem_align));
			add_dependency(*free_instr, *send_instr, dependency_kind::true_dep);
		});
	}
}

void instruction_graph_generator::compile(const abstract_command& cmd) {
	utils::match(
	    cmd,                                                                     //
	    [&](const execution_command& ecmd) { compile_execution_command(ecmd); }, //
	    [&](const push_command& pcmd) { compile_push_command(pcmd); },
	    [&](const await_push_command& apcmd) {
		    // We do not generate instructions for await-push commands immediately upon receiving them; instead, we buffer them and generate
		    // recv-instructions as soon as data is to be read by another instruction. This way, we can split the recv instructions and avoid
		    // unnecessary synchronization points between chunks that can otherwise profit from a transfer-compute overlap.
		    auto& buffer = m_buffers.at(apcmd.get_bid());
		    auto region = apcmd.get_region();

#ifndef NDEBUG
		    for(const auto& [box, trid] : buffer.pending_await_pushes.get_region_values(region)) {
			    assert(trid == transfer_id() && "received an await-push command into a previously await-pushed region without an intermediate read");
		    }
#endif
		    buffer.newest_data_location.update_region(region, data_location()); // not present anywhere locally
		    buffer.pending_await_pushes.update_region(region, apcmd.get_transfer_id());
	    },
	    [&](const horizon_command& hcmd) {
		    const auto horizon = &create<horizon_instruction>(hcmd.get_tid());
		    horizon->set_debug_info(horizon_instruction_debug_info(hcmd.get_cid()));
		    collapse_execution_front_to(horizon);
		    if(m_last_horizon) { apply_epoch(m_last_horizon); }
		    m_last_horizon = horizon;
	    },
	    [&](const epoch_command& ecmd) {
		    const auto epoch = &create<epoch_instruction>(ecmd.get_tid());
		    epoch->set_debug_info(epoch_instruction_debug_info(ecmd.get_cid()));
		    collapse_execution_front_to(epoch);
		    apply_epoch(epoch);
		    m_last_horizon = nullptr;
	    },
	    [](const auto& /* unhandled */) {
		    assert(!"unhandled command type");
		    std::abort();
	    });
}

} // namespace celerity::detail
