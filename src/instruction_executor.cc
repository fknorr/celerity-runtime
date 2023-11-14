#include "instruction_executor.h"
#include "buffer_storage.h" // for memcpy_strided_host
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "mpi_communicator.h" // TODO
#include "receive_arbiter.h"
#include "types.h"

#include <matchbox.hh>

namespace celerity::detail {

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, std::unique_ptr<communicator> comm, delegate* dlg)
    : m_delegate(dlg), m_communicator(std::move(comm)), m_backend_queue(std::move(backend_queue)), m_recv_arbiter(*m_communicator),
      m_thread(&instruction_executor::loop, this) {
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

instruction_executor::~instruction_executor() { wait(); }

void instruction_executor::wait() {
	if(m_thread.joinable()) { m_thread.join(); }
}

void instruction_executor::submit_instruction(const instruction& instr) { m_submission_queue.push_back(&instr); }

void instruction_executor::submit_pilot(const outbound_pilot& pilot) { m_submission_queue.push_back(pilot); }

void instruction_executor::announce_buffer_user_pointer(const buffer_id bid, const void* const ptr) {
	m_submission_queue.push_back(buffer_user_pointer_announcement{bid, ptr});
}

void instruction_executor::announce_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push_back(host_object_instance_announcement{hoid, std::move(instance)});
}

void instruction_executor::announce_reduction(const reduction_id rid, std::unique_ptr<runtime_reduction> reduction) {
	assert(reduction != nullptr);
	m_submission_queue.push_back(reduction_announcement{rid, std::move(reduction)});
}

void instruction_executor::thread_main() {
	try {
		loop();
	} catch(const std::exception& e) {
		CELERITY_CRITICAL("[executor] {}", e.what());
		std::abort();
	}
}

void instruction_executor::loop() {
	closure_hydrator::make_available();

	struct pending_instruction_info {
		size_t n_unmet_dependencies;
	};
	struct active_instruction_info {
		event completion_event;

		bool is_complete() const {
			return matchbox::match(
			    completion_event, //
			    [](const std::unique_ptr<backend::event>& evt) { return evt->is_complete(); },
			    CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK([](const boundary_checked_event& e) { return e.is_complete(); }, )[](
			        const std::unique_ptr<communicator::event>& evt) { return evt->is_complete(); },
			    [](const receive_arbiter::event& evt) { return evt.is_complete(); },
			    [](const std::future<host_queue::execution_info>& future) { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; },
			    [](const completed_synchronous) { return true; });
		};
	};

	m_allocations.emplace(null_allocation_id, nullptr);
	m_collective_groups.emplace(root_collective_group_id, m_communicator->get_collective_root());
	m_host_queue.require_collective_group(root_collective_group_id);

	std::vector<submission> loop_submission_queue;
	std::unordered_map<const instruction*, pending_instruction_info> pending_instructions;
	std::vector<const instruction*> ready_instructions;
	std::unordered_map<const instruction*, active_instruction_info> active_instructions;
	std::optional<std::chrono::steady_clock::time_point> last_progress_timestamp;
	bool progress_warning_emitted = false;
	while(m_expecting_more_submissions || !pending_instructions.empty() || !active_instructions.empty()) {
		m_recv_arbiter.poll_communicator();

		bool made_progress = false;

		for(auto active_it = active_instructions.begin(); active_it != active_instructions.end();) {
			auto& [active_instr, active_info] = *active_it;
			if(active_info.is_complete()) {
				CELERITY_DEBUG("[executor] completed I{}", active_instr->get_id());
				made_progress = true;
				for(const auto& dep : active_instr->get_dependents()) {
					if(const auto pending_it = pending_instructions.find(dep.node); pending_it != pending_instructions.end()) {
						auto& [pending_instr, pending_info] = *pending_it;
						assert(pending_info.n_unmet_dependencies > 0);
						pending_info.n_unmet_dependencies -= 1;
						if(pending_info.n_unmet_dependencies == 0) {
							ready_instructions.push_back(pending_instr);
							pending_instructions.erase(pending_it);
						}
					}
				}
				active_it = active_instructions.erase(active_it);
			} else {
				++active_it;
			}
		}

		if(m_submission_queue.swap_if_nonempty(loop_submission_queue)) {
			for(auto& submission : loop_submission_queue) {
				matchbox::match(
				    submission,
				    [&](const instruction* incoming_instr) {
					    const auto n_unmet_dependencies = static_cast<size_t>(std::count_if(
					        incoming_instr->get_dependencies().begin(), incoming_instr->get_dependencies().end(), [&](const instruction::dependency& dep) {
						        // TODO we really should have another unordered_set for the union of these
						        return pending_instructions.count(dep.node) != 0
						               || std::find(ready_instructions.begin(), ready_instructions.end(), dep.node) != ready_instructions.end()
						               || active_instructions.count(dep.node) != 0;
					        }));
					    if(n_unmet_dependencies > 0) {
						    pending_instructions.emplace(incoming_instr, pending_instruction_info{n_unmet_dependencies});
					    } else {
						    ready_instructions.push_back(incoming_instr);
					    }
				    },
				    [&](const outbound_pilot& pilot) { //
					    m_communicator->send_outbound_pilot(pilot);
				    },
				    [&](const buffer_user_pointer_announcement& ann) {
					    assert(m_buffer_user_pointers.count(ann.bid) == 0);
					    m_buffer_user_pointers.emplace(ann.bid, ann.ptr);
				    },
				    [&](host_object_instance_announcement& ann) {
					    assert(m_host_object_instances.count(ann.hoid) == 0);
					    m_host_object_instances.emplace(ann.hoid, std::move(ann.instance));
				    },
				    [&](reduction_announcement& ann) {
					    assert(m_reductions.count(ann.rid) == 0);
					    m_reductions.emplace(ann.rid, std::move(ann.reduction));
				    });
			}
			loop_submission_queue.clear();
		}

		for(const auto ready_instr : ready_instructions) {
			active_instructions.emplace(ready_instr, active_instruction_info{begin_executing(*ready_instr)});
			made_progress = true;
		}
		ready_instructions.clear();

		// TODO consider rate-limiting this (e.g. with an overflow counter) if steady_clock::now() turns out to have measurable latency
		if(made_progress) {
			last_progress_timestamp = std::chrono::steady_clock::now();
			progress_warning_emitted = false;
		} else if(last_progress_timestamp.has_value()) {
			const auto assume_stuck_after = std::chrono::seconds(3);
			const auto elapsed_since_last_progress = std::chrono::steady_clock::now() - *last_progress_timestamp;
			if(elapsed_since_last_progress > assume_stuck_after && !progress_warning_emitted) {
				std::string instr_list;
				for(auto& [instr, _] : active_instructions) {
					if(!instr_list.empty()) instr_list += ", ";
					fmt::format_to(std::back_inserter(instr_list), "I{}", instr->get_id());
				}
				CELERITY_WARN("[executor] no progress for {:.3f} seconds, potentially stuck. Active instructions: {}",
				    std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_since_last_progress).count(),
				    active_instructions.empty() ? "none" : instr_list);
				progress_warning_emitted = true;
			}
		}
	}

	assert(m_allocations.size() == 1); // null allocation
	assert(m_host_object_instances.empty());
}

instruction_executor::accessor_aux_info instruction_executor::prepare_accessor_hydration(target target, const access_allocation_map& amap) {
	accessor_aux_info aux_info;
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	if(!amap.empty()) {
		aux_info.out_of_bounds_box_per_accessor =
		    static_cast<oob_bounding_box*>(m_backend_queue->malloc(host_memory_id, amap.size() * sizeof(oob_bounding_box), alignof(oob_bounding_box)));
		std::uninitialized_default_construct_n(aux_info.out_of_bounds_box_per_accessor, amap.size());
	}
#endif

	std::vector<closure_hydrator::accessor_info> accessor_infos;
	accessor_infos.reserve(amap.size());
	for(size_t i = 0; i < amap.size(); ++i) {
		const auto ptr = m_allocations.at(amap[i].allocation_id);

		accessor_infos.push_back(closure_hydrator::accessor_info{ptr, amap[i].allocated_box_in_buffer,
		    amap[i].accessed_box_in_buffer CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, &aux_info.out_of_bounds_box_per_accessor[i])});
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		aux_info.declared_box_per_accessor.push_back(amap[i].accessed_box_in_buffer);
#endif
	}

	closure_hydrator::get_instance().arm(target, std::move(accessor_infos));

	return aux_info;
}

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
bool instruction_executor::boundary_checked_event::is_complete() const {
	if(!state.has_value()) return true; // we clear `state` completion to make is_complete() idempotent

	const bool is_complete = matchbox::match(
	    state->event, //
	    [](const std::unique_ptr<backend::event>& evt) { return evt->is_complete(); },
	    [](const std::future<host_queue::execution_info>& future) { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
	if(!is_complete) return false;

	for(size_t i = 0; i < state->aux_info.declared_box_per_accessor.size(); ++i) {
		if(const auto oob_box = state->aux_info.out_of_bounds_box_per_accessor[i].into_box(); !oob_box.empty()) {
			// This does not print task and buffer debug names as we used to prior to the IDAG transition. Including this info would require us to conditionally
			// integrate debug info into kernel / host task instructions. TODO
			CELERITY_ERROR("Out-of-bounds access detected: accessor {} attempted to access indices between {} which are outside of the mapped range {}", i,
			    oob_box, state->aux_info.declared_box_per_accessor[i]);
		}
	}

	if(state->aux_info.out_of_bounds_box_per_accessor != nullptr /* i.e. there is at least one accessor */) {
		state->executor->m_backend_queue->free(host_memory_id, state->aux_info.out_of_bounds_box_per_accessor);
	}

	state = {}; // make is_complete() idempotent
	return true;
}
#endif

instruction_executor::event instruction_executor::begin_executing(const instruction& instr) {
	static constexpr auto log_accesses = [](const access_allocation_map& map) {
		std::string acc_log;
		for(size_t i = 0; i < map.size(); ++i) {
			auto& aa = map[i];
			const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
			    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
			fmt::format_to(std::back_inserter(acc_log), "{} A{} {}", i == 0 ? ", accessing" : ",", aa.allocation_id, accessed_box_in_allocation);
		}
		return acc_log;
	};

	// TODO submit synchronous operations to thread pool
	return matchbox::match<event>(
	    instr,
	    [&](const clone_collective_group_instruction& ccginstr) {
		    const auto new_cgid = ccginstr.get_new_collective_group_id();
		    const auto origin_cgid = ccginstr.get_origin_collective_group_id();
		    CELERITY_DEBUG("[executor] I{}: clone collective group CG{} -> CG{}", ccginstr.get_id(), origin_cgid, new_cgid);

		    assert(m_collective_groups.count(new_cgid) == 0);
		    const auto new_group = m_collective_groups.at(origin_cgid)->clone();
		    m_collective_groups.emplace(new_cgid, new_group);
		    m_host_queue.require_collective_group(new_cgid);
		    return completed_synchronous();
	    },
	    [&](const alloc_instruction& ainstr) {
		    CELERITY_DEBUG("[executor] I{}: alloc M{}.A{}, {}%{} bytes", ainstr.get_id(), ainstr.get_memory_id(), ainstr.get_allocation_id(), ainstr.get_size(),
		        ainstr.get_alignment());

		    auto ptr = m_backend_queue->malloc(ainstr.get_memory_id(), ainstr.get_size(), ainstr.get_alignment());
		    m_allocations.emplace(ainstr.get_allocation_id(), ptr);

		    CELERITY_DEBUG("[executor] M{}.A{} allocated as {}", ainstr.get_memory_id(), ainstr.get_allocation_id(), ptr);
		    return completed_synchronous();
	    },
	    [&](const free_instruction& finstr) {
		    const auto it = m_allocations.find(finstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto ptr = it->second;

		    CELERITY_DEBUG("[executor] I{}: free M{}.A{}", finstr.get_id(), finstr.get_memory_id(), finstr.get_allocation_id());

		    m_allocations.erase(it);
		    m_backend_queue->free(finstr.get_memory_id(), ptr);
		    return completed_synchronous();
	    },
	    [&](const init_buffer_instruction& ibinstr) {
		    CELERITY_DEBUG("[executor] I{}: init B{} as M0.A{}, {} bytes", ibinstr.get_id(), ibinstr.get_buffer_id(), ibinstr.get_host_allocation_id(),
		        ibinstr.get_size());

		    const auto user_ptr = m_buffer_user_pointers.at(ibinstr.get_buffer_id());
		    const auto host_ptr = m_allocations.at(ibinstr.get_host_allocation_id());
		    memcpy(host_ptr, user_ptr, ibinstr.get_size());
		    return completed_synchronous();
	    },
	    [&](const export_instruction& einstr) {
		    CELERITY_DEBUG("[executor] I{}: export M0.A{} ({})+{}, {}x{} bytes", einstr.get_id(), einstr.get_host_allocation_id(),
		        einstr.get_allocation_range(), einstr.get_offset_in_allocation(), einstr.get_copy_range(), einstr.get_element_size());

		    const auto dest_ptr = einstr.get_out_pointer(); // TODO very naughty
		    const auto source_base_ptr = m_allocations.at(einstr.get_host_allocation_id());

		    // TODO copy-pasted,but export_instruction will be removed anyway
		    const auto dispatch_copy = [&](const auto dims) {
			    memcpy_strided_host(source_base_ptr, dest_ptr, einstr.get_element_size(), range_cast<dims.value>(einstr.get_allocation_range()),
			        id_cast<dims.value>(einstr.get_offset_in_allocation()), range_cast<dims.value>(einstr.get_copy_range()), id<dims.value>(),
			        range_cast<dims.value>(einstr.get_copy_range()));
		    };
		    switch(einstr.get_dimensions()) {
		    case 0: dispatch_copy(std::integral_constant<int, 0>()); break;
		    case 1: dispatch_copy(std::integral_constant<int, 1>()); break;
		    case 2: dispatch_copy(std::integral_constant<int, 2>()); break;
		    case 3: dispatch_copy(std::integral_constant<int, 3>()); break;
		    default: abort();
		    }
		    return completed_synchronous();
	    },
	    [&](const copy_instruction& cinstr) -> event {
		    CELERITY_DEBUG("[executor] I{}: copy M{}.A{}+{} -> M{}.A{}+{}, {}x{} bytes", cinstr.get_id(), cinstr.get_source_memory(),
		        cinstr.get_source_allocation(), cinstr.get_offset_in_source(), cinstr.get_dest_memory(), cinstr.get_dest_allocation(),
		        cinstr.get_offset_in_dest(), cinstr.get_copy_range(), cinstr.get_element_size());

		    const auto source_base_ptr = m_allocations.at(cinstr.get_source_allocation());
		    const auto dest_base_ptr = m_allocations.at(cinstr.get_dest_allocation());
		    if(cinstr.get_source_memory() == host_memory_id && cinstr.get_dest_memory() == host_memory_id) {
			    const auto dispatch_copy = [&](const auto dims) {
				    memcpy_strided_host(source_base_ptr, dest_base_ptr, cinstr.get_element_size(), range_cast<dims.value>(cinstr.get_source_range()),
				        id_cast<dims.value>(cinstr.get_offset_in_source()), range_cast<dims.value>(cinstr.get_dest_range()),
				        id_cast<dims.value>(cinstr.get_offset_in_dest()), range_cast<dims.value>(cinstr.get_copy_range()));
			    };
			    switch(cinstr.get_dimensions()) {
			    case 0: dispatch_copy(std::integral_constant<int, 0>()); break;
			    case 1: dispatch_copy(std::integral_constant<int, 1>()); break;
			    case 2: dispatch_copy(std::integral_constant<int, 2>()); break;
			    case 3: dispatch_copy(std::integral_constant<int, 3>()); break;
			    default: abort();
			    }
			    return completed_synchronous();
		    } else {
			    return m_backend_queue->memcpy_strided_device(cinstr.get_dimensions(), cinstr.get_source_memory(), cinstr.get_dest_memory(), source_base_ptr,
			        dest_base_ptr, cinstr.get_element_size(), cinstr.get_source_range(), cinstr.get_offset_in_source(), cinstr.get_dest_range(),
			        cinstr.get_offset_in_dest(), cinstr.get_copy_range());
		    }
	    },
	    [&](const device_kernel_instruction& dkinstr) {
		    CELERITY_DEBUG("[executor] I{}: launch device kernel on D{}, {}{}", dkinstr.get_id(), dkinstr.get_device_id(), dkinstr.get_execution_range(),
		        log_accesses(dkinstr.get_access_allocations()));

		    [[maybe_unused]] auto aux_info = prepare_accessor_hydration(target::device, dkinstr.get_access_allocations());

		    std::vector<void*> reduction_ptrs;
		    reduction_ptrs.reserve(dkinstr.get_reduction_allocations().size());
		    for(const auto& ra : dkinstr.get_reduction_allocations()) {
			    reduction_ptrs.push_back(m_allocations.at(ra.allocation_id));
		    }

		    auto evt = m_backend_queue->launch_kernel(dkinstr.get_device_id(), dkinstr.get_launcher(), dkinstr.get_execution_range(), reduction_ptrs);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return boundary_checked_event{boundary_checked_event::incomplete{this, std::move(evt), std::move(aux_info)}};
#else
		    return evt;
#endif
	    },
	    [&](const host_task_instruction& htinstr) {
		    CELERITY_DEBUG(
		        "[executor] I{}: launch host task, {}{}", htinstr.get_id(), htinstr.get_execution_range(), log_accesses(htinstr.get_access_allocations()));

		    [[maybe_unused]] auto aux_info = prepare_accessor_hydration(target::host_task, htinstr.get_access_allocations());

		    // TODO executor must not have any direct dependency on MPI!
		    MPI_Comm mpi_comm = MPI_COMM_NULL;
		    if(const auto cgid = htinstr.get_collective_group_id(); cgid != non_collective_group_id) {
			    const auto cg = m_collective_groups.at(htinstr.get_collective_group_id());
			    mpi_comm = dynamic_cast<mpi_communicator::collective_group&>(*cg).get_mpi_comm();
		    }

		    const auto& launch = htinstr.get_launcher();
		    auto evt = launch(m_host_queue, htinstr.get_execution_range(), mpi_comm);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return boundary_checked_event{boundary_checked_event::incomplete{this, std::move(evt), std::move(aux_info)}};
#else
		    return evt;
#endif
	    },
	    [&](const send_instruction& sinstr) {
		    CELERITY_DEBUG("[executor] I{}: send M{}.A{}+{}, {}x{} bytes to N{} (tag {})", sinstr.get_id(), host_memory_id /* TODO RDMA */,
		        sinstr.get_source_allocation_id(), sinstr.get_offset_in_allocation(), sinstr.get_send_range(), sinstr.get_element_size(),
		        sinstr.get_dest_node_id(), sinstr.get_tag());

		    const auto allocation_base = m_allocations.at(sinstr.get_source_allocation_id());
		    const communicator::stride stride{
		        sinstr.get_allocation_range(),
		        subrange<3>{sinstr.get_offset_in_allocation(), sinstr.get_send_range()},
		        sinstr.get_element_size(),
		    };
		    return m_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_tag(), allocation_base, stride);
	    },
	    [&](const receive_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: receive {} {} into M{}.A{} ({}), x{} bytes", rinstr.get_id(), rinstr.get_transfer_id(),
		        rinstr.get_requested_region(), rinstr.get_dest_memory(), rinstr.get_dest_allocation(), rinstr.get_allocated_box(), rinstr.get_element_size());

		    const auto allocation = m_allocations.at(rinstr.get_dest_allocation());
		    return m_recv_arbiter.receive(
		        rinstr.get_transfer_id(), rinstr.get_requested_region(), allocation, rinstr.get_allocated_box(), rinstr.get_element_size());
	    },
	    [&](const split_receive_instruction& srinstr) {
		    CELERITY_DEBUG("[executor] I{}: split receive {} {} into M{}.A{} ({}), x{} bytes", srinstr.get_id(), srinstr.get_transfer_id(),
		        srinstr.get_requested_region(), srinstr.get_dest_memory(), srinstr.get_dest_allocation(), srinstr.get_allocated_box(),
		        srinstr.get_element_size());

		    const auto allocation = m_allocations.at(srinstr.get_dest_allocation());
		    m_recv_arbiter.begin_split_receive(
		        srinstr.get_transfer_id(), srinstr.get_requested_region(), allocation, srinstr.get_allocated_box(), srinstr.get_element_size());
		    return completed_synchronous();
	    },
	    [&](const await_receive_instruction& arinstr) {
		    CELERITY_DEBUG("[executor] I{}: await receive {} {}", arinstr.get_id(), arinstr.get_transfer_id(), arinstr.get_received_region());

		    return m_recv_arbiter.await_split_receive_subregion(arinstr.get_transfer_id(), arinstr.get_received_region());
	    },
	    [&](const gather_receive_instruction& grinstr) {
		    CELERITY_DEBUG("[executor] I{}: gather receive {} into M{}.A{}, {} bytes per node", grinstr.get_id(), grinstr.get_transfer_id(),
		        grinstr.get_memory_id(), grinstr.get_allocation_id(), grinstr.get_node_chunk_size());

		    const auto allocation = m_allocations.at(grinstr.get_allocation_id());
		    return m_recv_arbiter.gather_receive(grinstr.get_transfer_id(), allocation, grinstr.get_node_chunk_size());
	    },
	    [&](const fill_identity_instruction& fiinstr) {
		    CELERITY_DEBUG("[executor] I{}: fill identity M{}.A{} x{} for R{}", fiinstr.get_id(), fiinstr.get_memory_id(), fiinstr.get_allocation_id(),
		        fiinstr.get_num_values(), fiinstr.get_reduction_id());

		    const auto allocation = m_allocations.at(fiinstr.get_allocation_id());
		    const auto& reduction = *m_reductions.at(fiinstr.get_reduction_id());
		    reduction.fill_identity(allocation, fiinstr.get_num_values());
		    return completed_synchronous();
	    },
	    [&](const reduce_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: reduce M{}.A{} x{} into M{}.A{} as R{}", rinstr.get_id(), rinstr.get_memory_id(), rinstr.get_source_allocation_id(),
		        rinstr.get_num_source_values(), rinstr.get_memory_id(), rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());

		    const auto gather_allocation = m_allocations.at(rinstr.get_source_allocation_id());
		    const auto dest_allocation = m_allocations.at(rinstr.get_dest_allocation_id());
		    const bool include_dest = false; // TODO
		    const auto& reduction = *m_reductions.at(rinstr.get_reduction_id());
		    reduction.reduce(dest_allocation, gather_allocation, rinstr.get_num_source_values(), include_dest);
		    // TODO GC runtime_reduction at some point
		    return completed_synchronous();
	    },
	    [&](const fence_instruction& finstr) {
		    CELERITY_DEBUG("[executor] I{}: fence", finstr.get_id());
		    finstr.get_promise()->fulfill();
		    return completed_synchronous();
	    },
	    [&](const destroy_host_object_instruction& dhoinstr) {
		    assert(m_host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
		    CELERITY_DEBUG("[executor] I{}: destroy H{}", dhoinstr.get_id(), dhoinstr.get_host_object_id());
		    m_host_object_instances.erase(dhoinstr.get_host_object_id());
		    return completed_synchronous();
	    },
	    [&](const horizon_instruction& hinstr) {
		    CELERITY_DEBUG("[executor] I{}: horizon", hinstr.get_id());

		    if(m_delegate != nullptr) { m_delegate->horizon_reached(hinstr.get_horizon_task_id()); }
		    return completed_synchronous();
	    },
	    [&](const epoch_instruction& einstr) {
		    switch(einstr.get_epoch_action()) {
		    case epoch_action::none: CELERITY_DEBUG("[executor] I{}: epoch", einstr.get_id()); break;
		    case epoch_action::barrier:
			    CELERITY_DEBUG("[executor] I{}: epoch (barrier)", einstr.get_id());
			    m_communicator->get_collective_root()->barrier();
			    break;
		    case epoch_action::shutdown:
			    CELERITY_DEBUG("[executor] I{}: epoch (shutdown)", einstr.get_id());
			    m_expecting_more_submissions = false;
			    break;
		    }
		    if(m_delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
			    m_delegate->epoch_reached(einstr.get_epoch_task_id());
		    }
		    return completed_synchronous();
	    });
}

} // namespace celerity::detail
