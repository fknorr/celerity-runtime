#include "distributed_graph_generator.h"

#include <tuple>

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "task.h"
#include "task_manager.h"

#include "print_utils.h" // NOCOMMIT

namespace celerity::detail {

distributed_graph_generator::distributed_graph_generator(
    const size_t num_nodes, const size_t num_local_devices, const node_id local_nid, command_graph& cdag, const task_manager& tm)
    : m_num_nodes(num_nodes), m_num_local_devices(num_local_devices), m_local_nid(local_nid), m_cdag(cdag), m_task_mngr(tm) {
	if(m_num_nodes > max_num_nodes) {
		throw std::runtime_error(fmt::format("Number of nodes requested ({}) exceeds compile-time maximum of {}", m_num_nodes, max_num_nodes));
	}

	// Build initial epoch command (this is required to properly handle anti-dependencies on host-initialized buffers).
	// We manually generate the first command, this will be replaced by applied horizons or explicit epochs down the line (see
	// set_epoch_for_new_commands).
	const auto epoch_cmd = cdag.create<epoch_command>(m_local_nid, task_manager::initial_epoch_task, epoch_action::none);
	epoch_cmd->mark_as_flushed(); // there is no point in flushing the initial epoch command
	m_epoch_for_new_commands = epoch_cmd->get_cid();
}

void distributed_graph_generator::add_buffer(const buffer_id bid, const range<3>& range, int dims) {
#if USE_COOL_REGION_MAP
	m_buffer_states.emplace(
	    std::piecewise_construct, std::tuple{bid}, std::tuple{region_map_t<write_command_state>{range, dims}, region_map_t<node_bitset>{range, dims}});
#else
	m_buffer_states.try_emplace(bid, buffer_state{range, range});
#endif
	// Mark contents as available locally (= don't generate await push commands) and fully replicated (= don't generate push commands).
	// This is required when tasks access host-initialized or uninitialized buffers.
	m_buffer_states.at(bid).local_last_writer.update_region(subrange_to_grid_box({id<3>(), range}), m_epoch_for_new_commands);
	m_buffer_states.at(bid).replicated_regions.update_region(subrange_to_grid_box({id<3>(), range}), node_bitset{}.set());
}

std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks) {
#ifndef NDEBUG
	assert(num_chunks > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_chunk.range[d] % granularity[d] == 0);
	}
#endif

	// Due to split granularity requirements or if num_workers > global_size[0],
	// we may not be able to create the requested number of chunks.
	const auto actual_num_chunks = std::min(num_chunks, full_chunk.range[0] / granularity[0]);

	// If global range is not divisible by (actual_num_chunks * granularity),
	// assign ceil(quotient) to the first few chunks and floor(quotient) to the remaining
	const auto small_chunk_size_dim0 = full_chunk.range[0] / (actual_num_chunks * granularity[0]) * granularity[0];
	const auto large_chunk_size_dim0 = small_chunk_size_dim0 + granularity[0];
	const auto num_large_chunks = (full_chunk.range[0] - small_chunk_size_dim0 * actual_num_chunks) / granularity[0];
	assert(num_large_chunks * large_chunk_size_dim0 + (actual_num_chunks - num_large_chunks) * small_chunk_size_dim0 == full_chunk.range[0]);

	std::vector<chunk<3>> result(actual_num_chunks, {full_chunk.offset, full_chunk.range, full_chunk.global_size});
	for(auto i = 0u; i < num_large_chunks; ++i) {
		result[i].range[0] = large_chunk_size_dim0;
		result[i].offset[0] += i * large_chunk_size_dim0;
	}
	for(auto i = num_large_chunks; i < actual_num_chunks; ++i) {
		result[i].range[0] = small_chunk_size_dim0;
		result[i].offset[0] += num_large_chunks * large_chunk_size_dim0 + (i - num_large_chunks) * small_chunk_size_dim0;
	}

#ifndef NDEBUG
	size_t total_range_dim0 = 0;
	for(size_t i = 0; i < result.size(); ++i) {
		total_range_dim0 += result[i].range[0];
		if(i == 0) {
			assert(result[i].offset[0] == full_chunk.offset[0]);
		} else {
			assert(result[i].offset[0] == result[i - 1].offset[0] + result[i - 1].range[0]);
		}
	}
	assert(total_range_dim0 == full_chunk.range[0]);
#endif

	return result;
}

// TODO: Make the split dimensions configurable for 3D chunks?
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks) {
#ifndef NDEBUG
	assert(num_chunks > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_chunk.range[d] % granularity[d] == 0);
	}
#endif

	const auto assign_factors = [&full_chunk, &granularity, &num_chunks](const size_t factor) {
		assert(num_chunks % factor == 0);
		const size_t max_chunks[2] = {full_chunk.range[0] / granularity[0], full_chunk.range[1] / granularity[1]};
		const size_t f0 = factor;
		const size_t f1 = num_chunks / factor;

		// Decide in which direction to split by first checking which
		// factor assignment produces more chunks under the given constraints.
		const std::array<size_t, 2> split0 = {std::min(f0, max_chunks[0]), std::min(f1, max_chunks[1])};
		const std::array<size_t, 2> split1 = {std::min(f1, max_chunks[0]), std::min(f0, max_chunks[1])};
		const auto count0 = split0[0] * split0[1];
		const auto count1 = split1[0] * split1[1];

		if(count0 == count1) {
			// If we're tied for the number of chunks we can create, try some heuristics to decide.

			// TODO: Yet another heuristic we should consider is how even chunk sizes are,
			// i.e., how balanced the workload is.

			// If domain is square(-ish), prefer splitting along slower dimension.
			// (These bounds have been chosen arbitrarily!)
			const float squareishness = std::sqrt(full_chunk.range.size()) / full_chunk.range[0];
			if(squareishness > 0.95f && squareishness < 1.05f) { return (f0 >= f1) ? split0 : split1; }

			// For non-square domains, prefer split that produces shorter edges (compare sum of circumferences)
			const auto circ0 = full_chunk.range[0] / split0[0] + full_chunk.range[1] / split0[1];
			const auto circ1 = full_chunk.range[0] / split1[0] + full_chunk.range[1] / split1[1];
			return circ0 < circ1 ? split0 : split1;

		} else if(count0 > count1) {
			return split0;
		} else {
			return split1;
		}
	};

	// Factorize num_chunks
	// Try to find factors as close to the square root as possible, that also produce
	// (or come close to) the requested number of chunks (under the given constraints).
	size_t f = std::floor(std::sqrt(num_chunks));
	std::array<size_t, 2> best_f_counts = {0, 0};
	while(f >= 1) {
		while(f > 1 && num_chunks % f != 0) {
			f--;
		}
		const auto counts = assign_factors(f);
		if(counts[0] * counts[1] > best_f_counts[0] * best_f_counts[1]) { best_f_counts = counts; }
		if(counts[0] * counts[1] == num_chunks) { break; }
		f--;
	}

	const auto actual_num_chunks = best_f_counts;
	if(actual_num_chunks[0] * actual_num_chunks[1] != num_chunks) {
		// TODO: Don't always warn here, do maybe once per task
		CELERITY_WARN("Unable to create {} chunks, created {} instead.", num_chunks, actual_num_chunks[0] * actual_num_chunks[1]);
	}

	// TODO: Move to generic utility, can share between 1D and 2D split
	const std::array<size_t, 2> ideal_chunk_size = {full_chunk.range[0] / actual_num_chunks[0], full_chunk.range[1] / actual_num_chunks[1]};
	const std::array<size_t, 2> small_chunk_size = {
	    (ideal_chunk_size[0] / granularity[0]) * granularity[0], (ideal_chunk_size[1] / granularity[1]) * granularity[1]};
	const std::array<size_t, 2> large_chunk_size = {small_chunk_size[0] + granularity[0], small_chunk_size[1] + granularity[1]};
	const std::array<size_t, 2> num_large_chunks = {(full_chunk.range[0] - small_chunk_size[0] * actual_num_chunks[0]) / granularity[0],
	    (full_chunk.range[1] - small_chunk_size[1] * actual_num_chunks[1]) / granularity[1]};

	std::vector<chunk<3>> result(actual_num_chunks[0] * actual_num_chunks[1], {full_chunk.offset, full_chunk.range, full_chunk.global_size});
	id<3> offset = full_chunk.offset;

	for(size_t j = 0; j < actual_num_chunks[0]; ++j) {
		range<2> chunk_size = {(j < num_large_chunks[0]) ? large_chunk_size[0] : small_chunk_size[0], 0};
		for(size_t i = 0; i < actual_num_chunks[1]; ++i) {
			chunk_size[1] = (i < num_large_chunks[1]) ? large_chunk_size[1] : small_chunk_size[1];
			auto& chnk = result[j * actual_num_chunks[1] + i];
			chnk.offset = offset;
			chnk.range[0] = chunk_size[0];
			chnk.range[1] = chunk_size[1];
			offset[1] += chunk_size[1];
		}
		offset[0] += chunk_size[0];
		offset[1] = full_chunk.offset[1];
	}

// Sanity check
// TODO: Can we keep DRY with 1D variant?
#ifndef NDEBUG
	GridRegion<3> reconstructed_chunk;
	for(auto& chnk : result) {
		assert(GridRegion<3>::intersect(reconstructed_chunk, subrange_to_grid_box(chnk)).empty());
		reconstructed_chunk = GridRegion<3>::merge(subrange_to_grid_box(chnk), reconstructed_chunk);
	}
	assert(GridRegion<3>::difference(reconstructed_chunk, subrange_to_grid_box(full_chunk)).empty());
#endif
	return result;
}

using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<access_mode, GridRegion<3>>>;

static buffer_requirements_map get_buffer_requirements_for_mapped_access(const command_group_task& tsk, subrange<3> sr, const range<3> global_size) {
	buffer_requirements_map result;
	const auto& access_map = tsk.get_buffer_access_map();
	const auto buffers = access_map.get_accessed_buffers();
	for(const buffer_id bid : buffers) {
		const auto modes = access_map.get_access_modes(bid);
		for(auto m : modes) {
			result[bid][m] = access_map.get_mode_requirements(bid, m, tsk.get_dimensions(), sr, global_size);
		}
	}
	return result;
}

// Steps:
// 1. Compute local chunk(s)
// 2. Compute data sources
// 3. Resolve data dependencies
//    => Avoid generating the same transfers twice (both across chunks and tasks)
//    => Could become tricky in conjunction with anti-dependency use-counters (= how to properly anticipate number of requests?)
// ?. Generate anti-dependencies ("use-counters"? "semaphores"?)
//    => Consider these additional use cases:
//       - Facilitate conflict resolution through copying
//       - Facilitate partial execution ("sub splits")
std::unordered_set<abstract_command*> distributed_graph_generator::build_task(const task& tsk) {
	assert(m_current_cmd_batch.empty());
	[[maybe_unused]] const auto cmd_count_before = m_cdag.command_count();

	const auto epoch_to_prune_before = m_epoch_for_new_commands;

	if(tsk.get_type() == task_type::epoch) {
		generate_epoch_command(dynamic_cast<const epoch_task&>(tsk));
	} else if(tsk.get_type() == task_type::horizon) {
		generate_horizon_command(dynamic_cast<const horizon_task&>(tsk));
	} else if(tsk.get_type() == task_type::device_compute || tsk.get_type() == task_type::host_compute || tsk.get_type() == task_type::master_node) {
		generate_execution_commands(dynamic_cast<const command_group_task&>(tsk));
	} else {
		throw std::runtime_error("Task type NYI");
	}

	// Commands without any other true-dependency must depend on the active epoch command to ensure they cannot be re-ordered before the epoch.
	// Need to check count b/c for some tasks we may not have generated any commands locally.
	if(m_cdag.task_command_count(tsk.get_id()) > 0) {
		for(const auto cmd : m_cdag.task_commands(tsk.get_id())) {
			generate_epoch_dependencies(cmd);
		}
	}

	assert(m_cdag.command_count() - cmd_count_before == m_current_cmd_batch.size());

	// If a new epoch was completed in the CDAG before the current task, prune all predecessor commands of that epoch.
	prune_commands_before(epoch_to_prune_before);

	// TODO assert(!m_current_cmd_batch.empty() || (!tsk.has_variable_split() && m_local_nid != 0));
	return std::move(m_current_cmd_batch);
}


std::vector<chunk<3>> distributed_graph_generator::split_task(const command_group_task& tsk) const {
	// TODO: Pieced together from naive_split_transformer. We can probably do without creating all chunks and discarding everything except our own.
	// TODO: Or - maybe - we actually want to store all chunks somewhere b/c we'll probably need them frequently for lookups later on?
	chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
	const size_t num_chunks = m_num_nodes * 1; // TODO Make configurable (oversubscription - although we probably only want to do this for local chunks)

	std::vector<chunk<3>> chunks;
	if(tsk.has_variable_split()) {
		const auto split = tsk.get_hint<experimental::hints::tiled_split>() != nullptr ? split_2d : split_1d;
		chunks = split(full_chunk, tsk.get_granularity(), num_chunks);
	} else {
		chunks = {full_chunk};
	}

	assert(chunks.size() <= num_chunks); // We may have created less than requested
	assert(!chunks.empty());
	return chunks;
}


void distributed_graph_generator::generate_execution_commands(const command_group_task& tsk) {
	const auto distributed_chunks = split_task(tsk);

#if 0
	for(const auto& dep : tsk.get_dependencies()) {
		for(const simple_dataflow_annotation& ann : dep.annotations) {
			const auto& producer = *dep.node;
			assert(producer.get_type() != task_type::epoch && producer.get_type() != task_type::horizon);

			// Since splits are reproducible, we simply re-compute the producer split here. If we ever introduce a non-reproducible split/scheduling behavior,
			// replace this by a std::unordered_map<task_id, std::vector<chunk<3>>> member (which can be pruned-on-epoch).
			const auto producer_chunks = split_task(producer);
			const auto& consumer_chunks = distributed_chunks;
			const auto [bid, box, producer_rm, consumer_rm] = ann;

			if((producer_chunks.size() == 1 && consumer_chunks.size() == 1)
			    || (producer_chunks.size() == consumer_chunks.size() && producer_rm->is_identity() && consumer_rm->is_identity() /* TODO relax */)) {
				// no data transfers necessary to fulfill this dependency
			} else if(producer_chunks.size() > 1 && (consumer_chunks.size() == 1 || consumer_rm->is_constant())) {
				// Gather or Allgather
				std::vector<subrange<3>> source_srs(m_num_nodes);
				for(node_id nid = 0; nid < std::min(m_num_nodes, producer_chunks.size()); ++nid) {
					auto producer_box = subrange_to_grid_box(apply_range_mapper(producer_rm, producer_chunks[nid], producer.get_dimensions()));
					source_srs[nid] = grid_box_to_subrange(GridBox<3>::intersect(box, producer_box));
				}
				const auto local_input_sr = source_srs[m_local_nid];
				const auto dest_sr = grid_box_to_subrange(box);
				const auto single_dest_nid = consumer_chunks.size() == 1 ? std::optional{node_id{0}} : std::nullopt;
				const auto gather_cmd = create_command<gather_command>(m_local_nid, bid, std::move(source_srs), dest_sr, single_dest_nid);

				auto& buffer_state = m_buffer_states.at(bid);
				CELERITY_TRACE(
				    "is_writer on N{} C{} = {}", gather_cmd->get_nid(), gather_cmd->get_cid(), !single_dest_nid.has_value() || *single_dest_nid == m_local_nid);
				if(!single_dest_nid.has_value() || *single_dest_nid == m_local_nid) {
					CELERITY_TRACE("generate_anti_dependencies for N{} C{}", gather_cmd->get_nid(), gather_cmd->get_cid());
					generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, box, gather_cmd);
				}

				if(local_input_sr.range.size() > 0) {
					// Store the read access for determining anti-dependencies later on
					m_command_buffer_reads[gather_cmd->get_cid()][bid] = subrange_to_grid_box(local_input_sr);

					for(const auto& [_, wcs] : buffer_state.local_last_writer.get_region_values(subrange_to_grid_box(local_input_sr))) {
						assert(wcs.is_fresh());
						m_cdag.add_dependency(gather_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
					}
				}

				if(m_local_nid == 0) {
					// we treat an Allgather result has having been produced by node 0 and replicated to all other nodes
					const auto replicated_to = single_dest_nid ? node_bitset().set(*single_dest_nid) : node_bitset().set(/* all */);
					buffer_state.replicated_regions.update_region(box, replicated_to);
				}

				if(!single_dest_nid.has_value() || *single_dest_nid == m_local_nid) {
					const auto is_replicated = m_local_nid != 0;
					buffer_state.local_last_writer.update_region(box, write_command_state(gather_cmd->get_cid(), is_replicated));
				} else {
					// TODO: We need a way of updating regions in place!
					for(auto& [last_box, wcs] : buffer_state.local_last_writer.get_region_values(box)) {
						if(wcs.is_fresh()) {
							wcs.mark_as_stale();
							buffer_state.local_last_writer.update_region(last_box, wcs);
						}
					}
				}

				if(const auto cgit = m_last_collective_commands.find(implicit_collective); cgit != m_last_collective_commands.end()) {
					m_cdag.add_dependency(gather_cmd, m_cdag.get(cgit->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				}
				m_last_collective_commands[implicit_collective] = gather_cmd->get_cid();

				// TODO epoch dependencies for receive-only gather cmds
			} else if(producer_chunks.size() == 1 && consumer_chunks.size() > 1 && consumer_rm->is_constant()) {
				// Broadcast
				const node_id source_nid = 0; // follows from producer_chunks.size() == 1, since we always assign chunks beginning at node 0
				const auto broadcast_cmd = create_command<broadcast_command>(m_local_nid, bid, source_nid, grid_box_to_subrange(box));

				auto& buffer_state = m_buffer_states.at(bid);

				generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, box, broadcast_cmd);

				if(m_local_nid == source_nid) {
					// Store the read access for determining anti-dependencies later on
					m_command_buffer_reads[broadcast_cmd->get_cid()][bid] = box;

					for(const auto& [_, wcs] : buffer_state.local_last_writer.get_region_values(box)) {
						assert(wcs.is_fresh());
						m_cdag.add_dependency(broadcast_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
					}
				}

				buffer_state.local_last_writer.update_region(box, write_command_state(broadcast_cmd->get_cid(), m_local_nid != source_nid /* is_replicated */));
				if(m_local_nid == source_nid) { buffer_state.replicated_regions.update_region(box, node_bitset().set(/* all nodes */)); }

				if(const auto cgit = m_last_collective_commands.find(implicit_collective); cgit != m_last_collective_commands.end()) {
					m_cdag.add_dependency(
					    broadcast_cmd, m_cdag.get(cgit->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				}
				m_last_collective_commands[implicit_collective] = broadcast_cmd->get_cid();

				// TODO epoch dependencies for receive-only broadcast cmds
			} else if(producer_chunks.size() == 1 && consumer_chunks.size() > 1 && consumer_rm->is_identity() /* TODO can this be relaxed? */) {
				// Scatter
			} else if(producer_chunks.size() > 1 && consumer_chunks.size() > 1 && false /* TODO producer_rm is transposed consumer_rm */) {
				// Alltoall
			}
		}
	}
#endif

	const auto* oversub_hint = tsk.get_hint<experimental::hints::oversubscribe>();
	const size_t oversub_factor = oversub_hint != nullptr ? oversub_hint->get_factor() : 1;

	// Assign each chunk to a node
	// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
	// transfers between tasks than a round-robin assignment (for typical stencil codes).
	// FIXME: This only works if the number of chunks is an integer multiple of the number of workers, e.g. 3 chunks for 2 workers degrades to RR.
	const auto chunks_per_node = std::max<size_t>(1, distributed_chunks.size() / m_num_nodes);

	// Distributed push model:
	// - Iterate over all remote chunks and find read requirements intersecting with owned buffer regions.
	// 	 Generate push commands for those regions.
	// - Iterate over all local chunks and find read requirements on remote data.
	//   Generate single await push command for each buffer that contains the entire region (will be fulfilled by one or more pushes).

	std::unordered_map<buffer_id, GridRegion<3>> per_buffer_local_writes;
	std::vector<std::tuple<buffer_id, GridRegion<3>, command_id>> local_last_writer_update_list;
	for(size_t i = 0; i < distributed_chunks.size(); ++i) {
		const node_id nid = (i / chunks_per_node) % m_num_nodes;
		const bool is_local_chunk = nid == m_local_nid;

		// Depending on whether this is a local chunk or not we may have to process a different set of "effective" chunks:
		// Local chunks may be split up again to create enough work for all local devices.
		// Processing remote chunks on the other hand doesn't require knowledge of the devices available on that particular node.
		// The same push commands generated for a single remote chunk also apply to the effective chunks generated on that node.
		std::vector<chunk<3>> effective_chunks;
		std::vector<device_id> device_assignments;
		if(is_local_chunk && m_num_local_devices > 1 && tsk.has_variable_split()) {
			const bool tiled_split = (tsk.get_hint<experimental::hints::tiled_split>() != nullptr);
			const auto split = tiled_split ? split_2d : split_1d;
			const auto per_device_chunks = split(distributed_chunks[i], tsk.get_granularity(), m_num_local_devices);
			if(oversub_factor == 1) {
				effective_chunks = per_device_chunks;
				device_assignments.resize(per_device_chunks.size());
				std::iota(device_assignments.begin(), device_assignments.end(), 0);
			} else {
				effective_chunks.reserve(per_device_chunks.size() * oversub_factor); // Assuming we can do the full split
				device_id did = 0;
				for(auto& dc : per_device_chunks) {
					auto oversub_chunks =
					    tiled_split ? split_2d(dc, tsk.get_granularity(), oversub_factor) : split_1d(dc, tsk.get_granularity(), oversub_factor);
					effective_chunks.insert(
					    effective_chunks.end(), std::make_move_iterator(oversub_chunks.begin()), std::make_move_iterator(oversub_chunks.end()));
					device_assignments.resize(device_assignments.size() + oversub_chunks.size());
					std::fill(device_assignments.end() - oversub_chunks.size(), device_assignments.end(), did);
					did++;
				}
			}
		} else {
			effective_chunks.push_back(distributed_chunks[i]);
			device_assignments.push_back(0);
		}

		assert(!is_local_chunk || !tsk.has_variable_split() || effective_chunks.size() == device_assignments.size());

		// CELERITY_CRITICAL("Chunk {}: Processing {} effective chunks", i, effective_chunks.size()); // NOCOMMIT

		for(size_t j = 0; j < effective_chunks.size(); ++j) {
			const auto& chnk = effective_chunks[j];
			auto requirements = get_buffer_requirements_for_mapped_access(tsk, chnk, tsk.get_global_size());

			execution_command* cmd = nullptr;
			if(is_local_chunk) {
				cmd = create_command<execution_command>(nid, tsk.get_id(), subrange{chnk});
				cmd->set_device_id(device_assignments[j]);
			}

			// We use the task id, together with the "chunk id" and the buffer id (stored separately) to match pushes against their corresponding await
			// pushes
			const transfer_id trid = static_cast<transfer_id>((tsk.get_id() << 32) | i);
			for(auto& [bid, reqs_by_mode] : requirements) {
				auto& buffer_state = m_buffer_states.at(bid);
				// For "true writes" (= not replicated) we have to wait with updating the last writer
				// map until all modes have been processed, as we'll otherwise end up with a cycle in
				// the graph if a command both writes and reads the same buffer region.
				GridRegion<3> written_region;

				std::vector<access_mode> required_modes;
				for(const auto mode : detail::access::all_modes) {
					if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						if(!req_it->second.empty()) { required_modes.push_back(mode); }
					}
				}

				for(const auto mode : required_modes) {
					const auto& req = reqs_by_mode.at(mode);
					if(detail::access::mode_traits::is_consumer(mode)) {
						if(is_local_chunk) {
							// Store the read access for determining anti-dependencies later on
							m_command_buffer_reads[cmd->get_cid()][bid] = GridRegion<3>::merge(m_command_buffer_reads[cmd->get_cid()][bid], req);

							const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
							GridRegion<3> missing_parts;
							for(const auto& [box, wcs] : local_sources) {
								if(!wcs.is_fresh()) {
									missing_parts = GridRegion<3>::merge(missing_parts, box);
									continue;
								}
								// NEXT STEP: It seems like we are adding a dependency on the write access in same task.
								// => We'll probably need an update list after all...
								// Q: Does this even make sense? Why is the read access overlapping with the write? Check range mappers
								m_cdag.add_dependency(cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
							}

							// There is data we don't yet have locally. Generate an await push command for it.
							if(!missing_parts.empty()) {
								assert(m_num_nodes > 1);
								auto ap_cmd = create_command<await_push_command>(m_local_nid, bid, trid, missing_parts);
								m_cdag.add_dependency(cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
								generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, missing_parts, ap_cmd);
								generate_epoch_dependencies(ap_cmd);
								// Remember that we have this data now
								buffer_state.local_last_writer.update_region(missing_parts, {ap_cmd->get_cid(), true});
							}
						} else {
							const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
							for(const auto& [local_box, wcs] : local_sources) {
								if(!wcs.is_fresh() || wcs.is_replicated()) { continue; }

								// Check if we've already pushed this box
								const auto replicated_boxes = buffer_state.replicated_regions.get_region_values(local_box);
								for(const auto& [replicated_box, nodes] : replicated_boxes) {
									if(nodes.test(nid)) continue;

									// Generate separate PUSH command for each last writer command for now,
									// possibly even multiple for partially already-replicated data
									// TODO: Can we consolidate?
									auto push_cmd = create_command<push_command>(m_local_nid, bid, 0, nid, trid, grid_box_to_subrange(replicated_box));
									m_cdag.add_dependency(push_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);

									// Store the read access for determining anti-dependencies later on
									m_command_buffer_reads[push_cmd->get_cid()][bid] = replicated_box;

									// Remember that we've replicated this region
									buffer_state.replicated_regions.update_box(replicated_box, node_bitset{nodes}.set(nid));
								}
							}
						}
					}

					if(is_local_chunk && detail::access::mode_traits::is_producer(mode)) {
						generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, req, cmd);

						// NOCOMMIT Remember to not create intra-task anti-dependencies onto data requests for RW accesses
						written_region = GridRegion<3>::merge(written_region, req);
						per_buffer_local_writes[bid] = GridRegion<3>::merge(per_buffer_local_writes[bid], req);
					}
				}

				if(!written_region.empty()) {
					// Even after all modes have been processed we can't do the update right away,
					// as this could otherwise result in faulty intra-task dependencies between chunks.
					local_last_writer_update_list.push_back(std::tuple{bid, written_region, cmd->get_cid()});
				}
			}
		}
	}

	for(auto& [bid, region, cid] : local_last_writer_update_list) {
		auto& buffer_state = m_buffer_states.at(bid);
		buffer_state.local_last_writer.update_region(region, cid);
		buffer_state.replicated_regions.update_region(region, node_bitset{});
	}

	// Update task-level buffer states
	auto requirements = get_buffer_requirements_for_mapped_access(tsk, subrange<3>(tsk.get_global_offset(), tsk.get_global_size()), tsk.get_global_size());
	for(auto& [bid, reqs_by_mode] : requirements) {
		GridRegion<3> global_writes;
		for(const auto mode : access::producer_modes) {
			if(reqs_by_mode.count(mode) == 0) continue;
			global_writes = GridRegion<3>::merge(global_writes, reqs_by_mode.at(mode));
		}
		const auto& local_writes = per_buffer_local_writes[bid];
		const auto remote_writes = GridRegion<3>::difference(global_writes, local_writes);
		auto& buffer_state = m_buffer_states.at(bid);

		// TODO: We need a way of updating regions in place!
		auto boxes_and_cids = buffer_state.local_last_writer.get_region_values(remote_writes);
		for(auto& [box, wcs] : boxes_and_cids) {
			if(wcs.is_fresh()) {
				wcs.mark_as_stale();
				buffer_state.local_last_writer.update_region(box, wcs);
			}
		}
	}

	process_task_side_effect_requirements(tsk);
}

void distributed_graph_generator::generate_anti_dependencies(
    task_id tid, buffer_id bid, const region_map_t<write_command_state>& last_writers_map, const GridRegion<3>& write_req, abstract_command* write_cmd) {
	const auto last_writers = last_writers_map.get_region_values(write_req);
	for(auto& [box, wcs] : last_writers) {
		const auto last_writer_cmd = m_cdag.get(static_cast<command_id>(wcs));
		assert(!isa<task_command>(last_writer_cmd) || static_cast<task_command*>(last_writer_cmd)->get_tid() != tid);

		CELERITY_TRACE("C{} a", write_cmd->get_nid(), write_cmd->get_cid());
		// Add anti-dependencies onto all successors of the writer
		bool has_successors = false;
		for(const auto cmd : last_writer_cmd->get_dependent_nodes()) {
			// Only consider true dependencies
			if(cmd->get_dependency(last_writer_cmd).kind != dependency_kind::true_dep) continue;

			// We might have already generated new commands within the same task that also depend on this; in that case, skip it
			if(isa<task_command>(cmd) && static_cast<task_command*>(cmd)->get_tid() == tid) continue;

			CELERITY_TRACE("N{} C{} b", write_cmd->get_nid(), write_cmd->get_cid());
			// So far we don't know whether the dependent actually intersects with the subrange we're writing
			if(const auto command_reads_it = m_command_buffer_reads.find(cmd->get_cid()); command_reads_it != m_command_buffer_reads.end()) {
				const auto& command_reads = command_reads_it->second;
				CELERITY_TRACE("N{} C{} c", write_cmd->get_nid(), write_cmd->get_cid());
				// The task might be a dependent because of another buffer
				if(const auto buffer_reads_it = command_reads.find(bid); buffer_reads_it != command_reads.end()) {
					CELERITY_TRACE("N{} C{} d", write_cmd->get_nid(), write_cmd->get_cid());
					if(!GridRegion<3>::intersect(write_req, buffer_reads_it->second).empty()) {
						CELERITY_TRACE("N{} C{} add anti-dep to C{}", write_cmd->get_nid(), write_cmd->get_cid(), cmd->get_cid());
						has_successors = true;
						m_cdag.add_dependency(write_cmd, cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}
			}
		}

		// In some cases (horizons, master node host task, weird discard_* constructs...)
		// the last writer might not have any successors. Just add the anti-dependency onto the writer itself then.
		if(!has_successors) { m_cdag.add_dependency(write_cmd, last_writer_cmd, dependency_kind::anti_dep, dependency_origin::dataflow); }
	}
}

void distributed_graph_generator::process_task_side_effect_requirements(const command_group_task& tsk) {
	const task_id tid = tsk.get_id();
	if(tsk.get_side_effect_map().empty()) return; // skip the loop in the common case
	if(m_cdag.task_command_count(tid) == 0) return;

	for(const auto cmd : m_cdag.task_commands(tid)) {
		for(const auto& side_effect : tsk.get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			if(const auto last_effect = m_host_object_last_effects.find(hoid); last_effect != m_host_object_last_effects.end()) {
				// TODO once we have different side_effect_orders, their interaction will determine the dependency kind
				m_cdag.add_dependency(cmd, m_cdag.get(last_effect->second), dependency_kind::true_dep, dependency_origin::dataflow);
			}

			// Simplification: If there are multiple chunks per node, we generate true-dependencies between them in an arbitrary order, when all we really
			// need is mutual exclusion (i.e. a bi-directional pseudo-dependency).
			m_host_object_last_effects.insert_or_assign(hoid, cmd->get_cid());
		}
	}
}

void distributed_graph_generator::set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon) {
	// both an explicit epoch command and an applied horizon can be effective epochs
	assert(isa<epoch_command>(epoch_or_horizon) || isa<horizon_command>(epoch_or_horizon));

	for(auto& [bid, bs] : m_buffer_states) {
		// TODO this could be optimized to something like cdag.apply_horizon(node_id, horizon_cmd) with much fewer internal operations
		bs.local_last_writer.apply_to_values([epoch_or_horizon](const write_command_state& wcs) {
			auto new_wcs = write_command_state(std::max(epoch_or_horizon->get_cid(), static_cast<command_id>(wcs)), wcs.is_replicated());
			if(!wcs.is_fresh()) new_wcs.mark_as_stale();
			return new_wcs;
		});
	}
	for(auto& [hoid, cid] : m_host_object_last_effects) {
		cid = std::max(epoch_or_horizon->get_cid(), cid);
	}
	for(auto& [cgid, cid] : m_last_collective_commands) {
		cid = std::max(epoch_or_horizon->get_cid(), cid);
	}

	m_epoch_for_new_commands = epoch_or_horizon->get_cid();
}

void distributed_graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
	const auto nid = new_front->get_nid();
	const auto previous_execution_front = m_cdag.get_execution_front(nid);
	for(const auto front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { m_cdag.add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_cdag.get_execution_front(nid).size() == 1 && *m_cdag.get_execution_front(nid).begin() == new_front);
}

// NOCOMMIT TODO: Apply epoch to data structures
void distributed_graph_generator::generate_epoch_command(const epoch_task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	const auto epoch = create_command<epoch_command>(m_local_nid, tsk.get_id(), tsk.get_epoch_action());
	set_epoch_for_new_commands(epoch);
	m_current_horizon = no_command;
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

// NOCOMMIT TODO: Apply previous horizon to data structures
void distributed_graph_generator::generate_horizon_command(const horizon_task& tsk) {
	assert(tsk.get_type() == task_type::horizon);
	const auto horizon = create_command<horizon_command>(m_local_nid, tsk.get_id());

	if(m_current_horizon != static_cast<command_id>(no_command)) {
		// Apply the previous horizon
		set_epoch_for_new_commands(m_cdag.get(m_current_horizon));
	}
	m_current_horizon = horizon->get_cid();

	// Make the horizon depend on the previous execution front
	reduce_execution_front_to(horizon);
}

void distributed_graph_generator::generate_epoch_dependencies(abstract_command* cmd) {
	// No command must be re-ordered before its last preceding epoch to enforce the barrier semantics of epochs.
	// To guarantee that each node has a transitive true dependency (=temporal dependency) on the epoch, it is enough to add an epoch -> command dependency
	// to any command that has no other true dependencies itself and no graph traversal is necessary. This can be verified by a simple induction proof.

	// As long the first epoch is present in the graph, all transitive dependencies will be visible and the initial epoch commands (tid 0) are the only
	// commands with no true predecessor. As soon as the first epoch is pruned through the horizon mechanism however, more than one node with no true
	// predecessor can appear (when visualizing the graph). This does not violate the ordering constraint however, because all "free-floating" nodes
	// in that snapshot had a true-dependency chain to their predecessor epoch at the point they were flushed, which is sufficient for following the
	// dependency chain from the executor perspective.

	if(const auto deps = cmd->get_dependencies();
	    std::none_of(deps.begin(), deps.end(), [](const abstract_command::dependency d) { return d.kind == dependency_kind::true_dep; })) {
		assert(cmd->get_cid() != m_epoch_for_new_commands);
		m_cdag.add_dependency(cmd, m_cdag.get(m_epoch_for_new_commands), dependency_kind::true_dep, dependency_origin::last_epoch);
	}
}

void distributed_graph_generator::prune_commands_before(const command_id epoch) {
	return;
	if(epoch > m_epoch_last_pruned_before) {
		m_cdag.erase_if([&](abstract_command* cmd) {
			if(cmd->get_cid() < epoch) {
				m_command_buffer_reads.erase(cmd->get_cid());
				return true;
			}
			return false;
		});
		m_epoch_last_pruned_before = epoch;
	}
}

} // namespace celerity::detail