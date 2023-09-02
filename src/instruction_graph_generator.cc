#include "instruction_graph_generator.h"
#include "access_modes.h"
#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
#include "recorders.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <numeric>

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, std::map<device_id, device_info> devices, instruction_recorder* const recorder)
    : m_tm(tm), m_devices(std::move(devices)), m_recorder(recorder) {
	assert(std::all_of(m_devices.begin(), m_devices.end(), [](const auto& kv) { return memory_id(std::get<0>(kv) + 1) < max_memories; }));
	const auto initial_epoch = &create<epoch_instruction>(task_id(0 /* or so we assume */), epoch_action::none);
	if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*initial_epoch, command_id(0 /* or so we assume */)); }
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
	utils::panic("data is requested to be read, but not located in any memory");
}

instruction_backend instruction_graph_generator::get_allocation_backend(const memory_id mid) const {
	if(mid == host_memory_id) return instruction_backend::host;

	const device_id did = mid - 1;
	const auto& backends = m_devices.at(did).backends;
	for(const auto preferred_backend : {instruction_backend::cuda, instruction_backend::sycl}) {
		if(backends.count(preferred_backend)) return preferred_backend;
	}
	utils::panic("no backend to allocate on D{}", did);
}

instruction_backend instruction_graph_generator::get_copy_backend(const memory_id from_mid, const memory_id to_mid) const {
	if(from_mid == host_memory_id && to_mid == host_memory_id) return instruction_backend::host;

	for(const auto preferred_backend : {instruction_backend::cuda, instruction_backend::sycl}) {
		bool supported_by_both = true;
		for(const auto mid : {from_mid, to_mid}) {
			if(mid == host_memory_id) continue; // assume that any (device) backend can copy from and to host
			const device_id did = mid - 1;
			supported_by_both &= m_devices.at(did).backends.count(preferred_backend);
		}
		if(supported_by_both) return preferred_backend;
	}
	utils::panic("no backend to copy between M{} and M{}", from_mid, to_mid);
}

instruction_backend instruction_graph_generator::get_kernel_launch_backend(const device_id did) const {
	const auto& backends = m_devices.at(did).backends;
	if(backends.count(instruction_backend::sycl) == 0) utils::panic("cannot launch kernels on D{} which does not support SYCL", did);
	return instruction_backend::sycl;
}

void instruction_graph_generator::allocate_contiguously(const buffer_id bid, const memory_id mid, const std::vector<box<3>>& boxes) {
	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories[mid];

	bounding_box_set contiguous_after_reallocation;
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

	auto unmerged_new_allocation = std::move(contiguous_after_reallocation).into_vector();
	const auto last_new_allocation = std::remove_if(unmerged_new_allocation.begin(), unmerged_new_allocation.end(),
	    [&](auto& box) { return std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == box; }); });
	unmerged_new_allocation.erase(last_new_allocation, unmerged_new_allocation.end());

	// region-merge adjacent boxes that need to be allocated (usually for oversubscriptions). This should not introduce problematic synchronization points since


	// TODO but it does introduce synchronization between producers on the resize-copies, which we want to avoid. To resolve this, allocate the fused boxes as
	// before, but use the non-fused boxes as copy destinations.
	region new_allocations(std::move(unmerged_new_allocation));

	// TODO don't copy data that will be overwritten (have an additional region<3> to_be_overwritten parameter)

	for(const auto& dest_box : new_allocations.get_boxes()) {
		auto& dest = memory.allocations.emplace_back(buffer.dims, m_next_aid++, dest_box, buffer.range);
		const auto alloc_instr =
		    &create<alloc_instruction>(get_allocation_backend(mid), dest.aid, mid, dest.box.get_area() * buffer.elem_size, buffer.elem_align);
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
			const auto full_copy_box = box_intersection(dest.box, source.box);
			box_vector<3> live_copy_boxes;
			for(const auto& [copy_box, location] : buffer.newest_data_location.get_region_values(full_copy_box)) {
				if(location.test(mid)) { live_copy_boxes.push_back(copy_box); }
			}
			region<3> live_copy_region(std::move(live_copy_boxes));

			for(const auto& copy_box : live_copy_region.get_boxes()) {
				assert(!copy_box.empty());

				const auto [source_offset, source_range] = source.box.get_subrange();
				const auto [dest_offset, dest_range] = dest_box.get_subrange();
				const auto [copy_offset, copy_range] = copy_box.get_subrange();

				// TODO to avoid introducing a synchronization point on oversubscription, split into multiple copies if that will allow unimpeded
				// oversubscribed-producer to oversubscribed-consumer data flow.

				const auto copy_instr = &create<copy_instruction>(get_allocation_backend(mid), buffer.dims, mid, source.aid, source_range,
				    copy_offset - source_offset, mid, dest.aid, dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);

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

				if(m_recorder != nullptr) {
					*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::resize, bid, buffer.debug_name, copy_box);
				}
			}
		}

		if(m_recorder != nullptr) {
			*m_recorder << alloc_instruction_record(
			    *alloc_instr, alloc_instruction_record::alloc_origin::buffer, buffer_allocation_record{bid, buffer.debug_name, dest_box});
		}
	}

	// TODO consider keeping old allocations around until their box is written to in order to resolve "buffer-locking" anti-dependencies
	for(const auto free_aid : free_after_reallocation) {
		const auto& allocation = *std::find_if(memory.allocations.begin(), memory.allocations.end(), [&](const auto& a) { return a.aid == free_aid; });
		const auto free_instr = &create<free_instruction>(get_allocation_backend(mid), allocation.aid);
		for(const auto& [_, front] : allocation.access_fronts.get_region_values(allocation.box)) { // TODO copy-pasta
			for(const auto dep_instr : front.front) {
				add_dependency(*free_instr, *dep_instr, dependency_kind::true_dep);
			}
		}
		if(m_recorder != nullptr) {
			*m_recorder << free_instruction_record(*free_instr, mid, allocation.box.get_area() * buffer.elem_size, buffer.elem_align,
			    buffer_allocation_record{bid, buffer.debug_name, allocation.box});
		}
	}

	// TODO garbage-collect allocations that are both stale and not written to? We cannot re-fetch buffer subranges from their original producer without
	// some sort of inter-node pull semantics if the GC turned out to be a misprediction, but we can swap allocations to the host when we run out of device
	// memory. Basically we would annotate each allocation with an last-used value to implement LRU semantics.

	const auto end_retain_after_allocation = std::remove_if(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) {
		return std::any_of(free_after_reallocation.begin(), free_after_reallocation.end(), [&](const auto aid) { return alloc.aid == aid; });
	});
	memory.allocations.erase(end_retain_after_allocation, memory.allocations.end());
}


void instruction_graph_generator::satisfy_read_requirements(const buffer_id bid, const std::vector<std::pair<memory_id, region<3>>>& reads) {
	auto& buffer = m_buffers.at(bid);

	std::unordered_map<memory_id, std::vector<region<3>>> unsatisfied_reads;
	for(const auto& [mid, read_region] : reads) {
		box_vector<3> unsatified_boxes;
		for(const auto& [box, location] : buffer.newest_data_location.get_region_values(read_region)) {
			if(!location.test(mid)) { unsatified_boxes.push_back(box); }
		}
		region<3> unsatisfied_region(std::move(unsatified_boxes));
		if(!unsatisfied_region.empty()) { unsatisfied_reads[mid].push_back(std::move(unsatisfied_region)); }
	}

	// transform vectors of potentially-overlapping unsatisfied regions into disjoint regions
	for(auto& [mid, regions] : unsatisfied_reads) {
	restart:
		for(size_t i = 0; i < regions.size(); ++i) {
			for(size_t j = i + 1; j < regions.size(); ++j) {
				auto intersection = region_intersection(regions[i], regions[j]);
				if(!intersection.empty()) {
					regions[i] = region_difference(regions[i], intersection);
					regions[j] = region_difference(regions[j], intersection);
					regions.push_back(std::move(intersection));
					// if intersections above are actually subsets, we will end up with empty regions
					regions.erase(std::remove_if(regions.begin(), regions.end(), std::mem_fn(&region<3>::empty)), regions.end());
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
			std::unordered_map<transfer_id, region<3>> transfer_regions;
			for(auto& [box, trid] : buffer.pending_await_pushes.get_region_values(unsatisfied_region)) {
				if(trid == transfer_id()) continue;

				auto& tr_region = transfer_regions[trid]; // allow default-insert
				tr_region = region_union(tr_region, box);
			}
			for(auto& [trid, accepted_transfer_region] : transfer_regions) {
				for(auto& alloc : memory.allocations) {
					const auto accepted_alloc_region = region_intersection(alloc.box, accepted_transfer_region);
					for(const box<3>& tr_box : accepted_alloc_region.get_boxes()) {
						const auto recv_box = box_intersection(alloc.box, tr_box);
						if(recv_box.empty()) return;

						const auto [alloc_offset, alloc_range] = alloc.box.get_subrange();
						const auto [recv_offset, recv_range] = recv_box.get_subrange();
						const auto recv_instr = &create<recv_instruction>(
						    bid, trid, mid, alloc.aid, buffer.dims, alloc_range, recv_offset - alloc_offset, recv_offset, recv_range, buffer.elem_size);
						command_id await_push_cid = 1234; // TODO where do we get this from without changing non-debug code paths too much?

						// TODO the dependency logic here is duplicated from copy-instruction generation
						for(const auto& [_, front] : alloc.access_fronts.get_region_values(recv_box)) {
							for(const auto dep_instr : front.front) {
								add_dependency(*recv_instr, *dep_instr, dependency_kind::true_dep);
							}
						}
						add_dependency(*recv_instr, *m_last_epoch, dependency_kind::true_dep);
						alloc.record_write(recv_box, recv_instr);

						buffer.original_writers.update_region(recv_box, recv_instr);

						if(m_recorder != nullptr) { *m_recorder << recv_instruction_record(*recv_instr, await_push_cid, bid, buffer.debug_name); }
					}
				}
				// TODO assert that the entire region is consumed (... eventually?)

				// TODO this always transfers to a single memory, which is suboptimal. Find a way to implement "device broadcast" from a single receive
				// buffer. This will probably require explicit handling of the receive buffer inside the IDAG.
				//
				// How to realize this in the presence of CUDA-aware MPI? We want to have RDMA MPI_Recv to device buffers as the happy path, and a copy from
				// receive buffer as fallback - but only in the absence of a broadcast condition.
				// 	 - annotate recv_instructions with the desired allocation to RDMA to (if any)
				//   - have a conditional copy_instruction following it, in case the RDMA fails (or should the buffer_transfer_manager dispatch that copy?)
				//   - in case of a RDMA failure, or when we want to broadcast, we need a release_transfer_instruction to hand the transfer_allocation_id
				//   back
				// 	   to the buffer_transfer_manager (again kinda conditional on whether we did an RDMA receive or not)
				buffer.newest_data_location.update_region(accepted_transfer_region, data_location().set(mid));
				buffer.pending_await_pushes.update_region(accepted_transfer_region, transfer_id());

				unsatisfied_region = region_difference(unsatisfied_region, accepted_transfer_region);
			}
		}

		// remove regions that are fully satisfied by incoming transfers
		disjoint_regions.erase(std::remove_if(disjoint_regions.begin(), disjoint_regions.end(), std::mem_fn(&region<3>::empty)), disjoint_regions.end());
	}

	struct copy_template {
		memory_id source_mid;
		memory_id dest_mid;
		region<3> region;
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
				//  kernel in a loop). However we still want to be able to detect the error condition of not having received a buffer region that was
				//  produced by some other kernel in the past.
				assert(sources.any() && "trying to read data that is neither found locally nor has been await-pushed before");
			}
#endif

			// try finding a common source for the entire region first to minimize instruction / synchronization complexity down the line
			const auto common_sources = std::accumulate(region_sources.begin(), region_sources.end(), data_location(),
			    [](const data_location common, const std::pair<box<3>, data_location>& box_sources) { return common & box_sources.second; });

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
			std::unordered_map<memory_id, detail::region<3>> combined_source_regions;
			for(const auto& [box, sources] : region_sources) {
				memory_id copy_from;
				if(const auto device_sources = data_location(sources).reset(host_memory_id); device_sources.any()) {
					copy_from = next_location(device_sources, dest_mid + 1);
				} else {
					copy_from = host_memory_id; // heuristic: avoid copies from host because they can't be DMA'd
				}
				auto& copy_region = combined_source_regions[copy_from];
				copy_region = region_union(copy_region, box);
			}
			for(auto& [copy_from, copy_region] : combined_source_regions) {
				pending_copies.push_back({copy_from, dest_mid, std::move(copy_region)});
			}
		}
	}

	for(auto& copy : pending_copies) {
		assert(copy.dest_mid != copy.source_mid);
		const auto copy_backend = get_copy_backend(copy.source_mid, copy.dest_mid);
		auto& buffer = m_buffers.at(bid);
		for(const box<3>& box : copy.region.get_boxes()) {
			for(auto& source : buffer.memories[copy.source_mid].allocations) {
				const auto read_box = box_intersection(box, source.box);
				if(read_box.empty()) continue;

				for(auto& dest : buffer.memories[copy.dest_mid].allocations) {
					const auto copy_box = box_intersection(read_box, dest.box);
					if(copy_box.empty()) continue;

					const auto [source_offset, source_range] = source.box.get_subrange();
					const auto [dest_offset, dest_range] = dest.box.get_subrange();
					const auto [copy_offset, copy_range] = copy_box.get_subrange();

					const auto copy_instr = &create<copy_instruction>(copy_backend, buffer.dims, copy.source_mid, source.aid, source_range,
					    copy_offset - source_offset, copy.dest_mid, dest.aid, dest_range, copy_offset - dest_offset, copy_range, buffer.elem_size);

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

					if(m_recorder != nullptr) {
						*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::coherence, bid, buffer.debug_name, copy_box);
					}
				}
			}
		}
	}
}


std::vector<copy_instruction*> instruction_graph_generator::linearize_buffer_subrange(
    const buffer_id bid, const box<3>& box, const memory_id out_mid, alloc_instruction& alloc_instr) {
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
	std::unordered_map<memory_id, region<3>> pending_copies;

	// try finding a common source for the entire region first to minimize instruction / synchronization complexity down the line
	const auto common_sources = std::accumulate(box_sources.begin(), box_sources.end(), data_location(),
	    [](const data_location common, const std::pair<detail::box<3>, data_location>& box_sources) { return common & box_sources.second; });
	if(const auto common_device_sources = data_location(common_sources).reset(host_memory_id); common_device_sources.any()) {
		// best case: we can copy all data from a single device
		const auto source_mid = next_location(common_device_sources, host_memory_id + 1);
		auto& copy_region = pending_copies[source_mid];
		copy_region = region_union(copy_region, box);
	} else {
		// mix and match sources - there exists an optimal solution, but for now we just assemble source regions by picking the next device if any,
		// or the host as a fallback.
		for(const auto& [box, sources] : box_sources) {
			const memory_id source_mid = next_location(sources, host_memory_id + 1); // picks host_memory last (which usually can't leverage DMA)
			auto& copy_region = pending_copies[source_mid];
			copy_region = region_union(copy_region, box);
		}
	}

	std::vector<copy_instruction*> copy_instrs;
	for(const auto& [source_mid, region] : pending_copies) {
		const auto copy_backend = get_copy_backend(source_mid, out_mid);
		for(const auto& source_box : region.get_boxes()) {
			for(auto& source : buffer.memories[source_mid].allocations) {
				const auto copy_box = box_intersection(source.box, source_box);
				if(copy_box.empty()) continue;

				const auto [source_offset, source_range] = source.box.get_subrange();
				const auto [copy_offset, copy_range] = copy_box.get_subrange();
				const auto [dest_offset, dest_range] = box.get_subrange();
				const auto copy_instr = &create<copy_instruction>(copy_backend, buffer.dims, source_mid, source.aid, source_range, copy_offset - source_offset,
				    out_mid, alloc_instr.get_allocation_id(), dest_range, dest_offset - source_offset, copy_range, buffer.elem_size);

				add_dependency(*copy_instr, alloc_instr, dependency_kind::true_dep);
				// TODO copy-pasta
				for(const auto& [_, last_writer_instr] : source.last_writers.get_region_values(copy_box)) {
					assert(last_writer_instr != nullptr);
					add_dependency(*copy_instr, *last_writer_instr, dependency_kind::true_dep);
				}
				source.record_read(copy_box, copy_instr);

				copy_instrs.push_back(copy_instr);

				if(m_recorder != nullptr) {
					*m_recorder << copy_instruction_record(*copy_instr, copy_instruction_record::copy_origin::linearize, bid, buffer.debug_name, copy_box);
				}
			}
		}
	}

	return copy_instrs;
}


int instruction_graph_generator::create_pilot_message(const buffer_id bid, const transfer_id trid, const box<3>& box) {
	int tag = m_next_p2p_tag++;
	m_pilots.push_back(pilot_message{tag, bid, trid, box});
	if(m_recorder != nullptr) { *m_recorder << m_pilots.back(); }
	return tag;
}


// TODO HACK we're just pulling in the splitting logic from distributed_graph_generator here
std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks, const int dims);


void instruction_graph_generator::compile_execution_command(const execution_command& ecmd) {
	const auto& tsk = *m_tm.get_task(ecmd.get_tid());

	struct partial_instruction {
		subrange<3> execution_sr;
		device_id did = -1;
		memory_id mid = host_memory_id;
		buffer_read_write_map rw_map;
		side_effect_map se_map;
		instruction* instruction = nullptr;
		std::vector<buffer_allocation_record> allocation_buffer_map; // for kernel_instructions, if (m_recorder)
	};
	std::vector<partial_instruction> cmd_instrs;

	const auto command_sr = ecmd.get_execution_range();
	if(tsk.has_variable_split() && tsk.get_execution_target() == execution_target::device) {
		const auto device_chunks =
		    split_equal(chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size()), tsk.get_granularity(), m_devices.size(), tsk.get_dimensions());
		for(device_id did = 0; did < m_devices.size() && did < device_chunks.size(); ++did) {
			const auto& chunk = device_chunks[did];
			auto& insn = cmd_instrs.emplace_back();
			insn.execution_sr = subrange<3>(chunk.offset, chunk.range);
			insn.did = did;
			insn.mid = device_to_memory_id(did);
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

	std::unordered_map<std::pair<buffer_id, memory_id>, region<3>, utils::pair_hash> invalidations;
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(auto& insn : cmd_instrs) {
			reads_writes rw;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), insn.execution_sr, tsk.get_global_size());
				if(access::mode_traits::is_consumer(mode)) { rw.reads = region_union(rw.reads, req); }
				if(access::mode_traits::is_producer(mode)) { rw.writes = region_union(rw.writes, req); }
				if(!access::mode_traits::is_consumer(mode)) {
					auto& region = invalidations[{bid, insn.mid}]; // allow default-insert
					region = region_union(region, req);
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

	std::unordered_map<std::pair<buffer_id, memory_id>, std::vector<box<3>>, utils::pair_hash> not_contiguously_allocated_boxes;
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

	std::unordered_map<buffer_id, std::vector<std::pair<memory_id, region<3>>>> buffer_reads;
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
		if(m_recorder != nullptr) { instr.allocation_buffer_map.resize(bam.get_num_accesses()); }

		for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
			const auto [bid, mode] = bam.get_nth_access(i);
			const auto accessed_box = bam.get_requirements_for_nth_access(i, tsk.get_dimensions(), instr.execution_sr, tsk.get_global_size());
			const auto& buffer = m_buffers.at(bid);
			const auto& allocations = buffer.memories[instr.mid].allocations;
			const auto allocation_it = std::find_if(
			    allocations.begin(), allocations.end(), [&](const buffer_memory_per_allocation_data& alloc) { return alloc.box.covers(accessed_box); });
			assert(allocation_it != allocations.end());
			const auto& alloc = *allocation_it;
			const auto [access_offset, access_range] = accessed_box.get_subrange();
			const auto [alloc_offset, alloc_range] = alloc.box.get_subrange();
			allocation_map[i] = access_allocation{alloc.aid, alloc_range, access_offset - alloc_offset, {access_offset, access_range}};
			if(m_recorder != nullptr) { instr.allocation_buffer_map[i] = buffer_allocation_record{bid, buffer.debug_name, alloc.box}; }
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
				const auto reads_from_alloc = region_intersection(rw.reads, alloc.box);
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
				const auto writes_to_alloc = region_intersection(rw.writes, alloc.box);
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
				for(const auto& box : rw.writes.get_boxes()) {
					const auto write_box = box_intersection(alloc.box, box);
					if(!write_box.empty()) { alloc.record_write(write_box, instr.instruction); }
				}
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

	// 7) insert epoch and horizon dependencies, apply epochs, optionally record the instruction

	for(auto& instr : cmd_instrs) {
		// if there is no transitive dependency to the last epoch, insert one explicitly to enforce ordering.
		// this is never necessary for horizon and epoch commands, since they always have dependencies to the previous execution front.
		const auto deps = instr.instruction->get_dependencies();
		if(std::none_of(deps.begin(), deps.end(), [](const instruction::dependency& dep) { return dep.kind == dependency_kind::true_dep; })) {
			add_dependency(*instr.instruction, *m_last_epoch, dependency_kind::true_dep);
		}

		if(m_recorder != nullptr) {
			if(const auto kernel_instr = dynamic_cast<kernel_instruction*>(instr.instruction)) {
				*m_recorder << kernel_instruction_record(
				    *kernel_instr, ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), std::move(instr.allocation_buffer_map));
			}
		}
	}
}


void instruction_graph_generator::compile_push_command(const push_command& pcmd) {
	const auto bid = pcmd.get_bid();
	auto& buffer = m_buffers.at(bid);
	const auto push_box = box(pcmd.get_range());

	// We want to generate the fewest number of send instructions possible without introducing new synchronization points between chunks of the same
	// command that generated the pushed data. This will allow compute-transfer overlap, especially in the case of oversubscribed splits.
	std::unordered_map<instruction*, region<3>> writer_regions;
	for(auto& [box, writer] : buffer.original_writers.get_region_values(push_box)) {
		auto& region = writer_regions[writer]; // allow default-insert
		region = region_union(region, box);
	}

	for(auto& [writer, region] : writer_regions) {
		for(const auto& box : region.get_boxes()) {
			const auto bytes = box.get_area() * buffer.elem_size;

			// this allocation is not associated with a buffer (and thus not tracked in m_buffers), so we add all dependencies immediately
			const auto alloc_instr = &create<alloc_instruction>(instruction_backend::host, m_next_aid++, host_memory_id, bytes, buffer.elem_align);
			add_dependency(*alloc_instr, *m_last_epoch, dependency_kind::true_dep);

			const auto copy_instrs = linearize_buffer_subrange(bid, box, host_memory_id, *alloc_instr);

			const int tag = create_pilot_message(bid, pcmd.get_transfer_id(), box);
			const auto send_instr = &create<send_instruction>(pcmd.get_transfer_id(), pcmd.get_target(), tag, alloc_instr->get_allocation_id(), bytes);
			for(const auto copy_instr : copy_instrs) {
				add_dependency(*send_instr, *copy_instr, dependency_kind::true_dep);
			}
			const auto free_instr = &create<free_instruction>(instruction_backend::host, alloc_instr->get_allocation_id());
			add_dependency(*free_instr, *send_instr, dependency_kind::true_dep);

			if(m_recorder != nullptr) {
				*m_recorder << alloc_instruction_record(*alloc_instr, alloc_instruction_record::alloc_origin::send, std::nullopt);
				*m_recorder << send_instruction_record(*send_instr, pcmd.get_cid(), bid, buffer.debug_name, box);
				*m_recorder << free_instruction_record(*free_instr, host_memory_id, bytes, buffer.elem_align, std::nullopt);
			}
		}
	}
}

std::vector<const instruction*> instruction_graph_generator::compile(const abstract_command& cmd) {
	m_current_batch.clear();

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
		    collapse_execution_front_to(horizon);
		    if(m_last_horizon) { apply_epoch(m_last_horizon); }
		    m_last_horizon = horizon;
		    if(m_recorder != nullptr) { *m_recorder << horizon_instruction_record(*horizon, hcmd.get_cid()); }
	    },
	    [&](const epoch_command& ecmd) {
		    const auto epoch = &create<epoch_instruction>(ecmd.get_tid(), ecmd.get_epoch_action());
		    collapse_execution_front_to(epoch);
		    apply_epoch(epoch);
		    m_last_horizon = nullptr;
		    if(m_recorder != nullptr) { *m_recorder << epoch_instruction_record(*epoch, ecmd.get_cid()); }
	    },
	    [](const auto& /* unhandled */) {
		    assert(!"unhandled command type");
		    std::abort();
	    });

	return m_current_batch;
}

} // namespace celerity::detail
