#include "instruction_executor.h"
#include "buffer_storage.h" // for memcpy_strided_host
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "recv_arbiter.h"
#include "types.h"

#include <atomic>
#include <mutex>

#include <matchbox.hh>

namespace celerity::detail {

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, const communicator_factory& comm_factory, delegate* dlg)
    : m_delegate(dlg), m_backend_queue(std::move(backend_queue)), m_communicator(comm_factory.make_communicator(this)), m_recv_arbiter(*m_communicator),
      m_thread(&instruction_executor::loop, this) {
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

instruction_executor::~instruction_executor() { m_thread.join(); }

void instruction_executor::submit_instruction(const instruction& instr) { m_submission_queue.push_back(&instr); }

void instruction_executor::submit_pilot(const outbound_pilot& pilot) { m_submission_queue.push_back(pilot); }

void instruction_executor::announce_buffer_user_pointer(const buffer_id bid, const void* const ptr) {
	m_submission_queue.push_back(buffer_user_pointer_announcement{bid, ptr});
}

void instruction_executor::announce_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push_back(host_object_instance_announcement{hoid, std::move(instance)});
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
			    [](const std::unique_ptr<communicator::event>& evt) { return evt->is_complete(); },
			    [](const recv_arbiter::event& evt) { return evt.is_complete(); },
			    [](const std::future<host_queue::execution_info>& future) { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; },
			    [](const completed_synchronous) { return true; });
		};
	};

	std::vector<submission> loop_submission_queue;
	std::vector<inbound_pilot> loop_inbound_pilot_queue;
	std::unordered_map<const instruction*, pending_instruction_info> pending_instructions;
	std::vector<const instruction*> ready_instructions;
	std::unordered_map<const instruction*, active_instruction_info> active_instructions;
	while(m_expecting_more_submissions || !pending_instructions.empty() || !active_instructions.empty()) {
		for(auto active_it = active_instructions.begin(); active_it != active_instructions.end();) {
			auto& [active_instr, active_info] = *active_it;
			if(active_info.is_complete()) {
				CELERITY_DEBUG("[executor] completed I{}", active_instr->get_id());
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
				    });
			}
			loop_submission_queue.clear();
		}

		if(m_inbound_pilot_queue.swap_if_nonempty(loop_inbound_pilot_queue)) {
			for(auto& pilot : loop_inbound_pilot_queue) {
				m_recv_arbiter.push_inbound_pilot(pilot);
			}
			loop_inbound_pilot_queue.clear();
		}

		for(const auto ready_instr : ready_instructions) {
			active_instructions.emplace(ready_instr, active_instruction_info{begin_executing(*ready_instr)});
		}
		ready_instructions.clear();
	}
}

instruction_executor::event instruction_executor::begin_executing(const instruction& instr) {
	static constexpr auto log_accesses = [](const access_allocation_map& map) {
		std::string acc_log;
		for(size_t i = 0; i < map.size(); ++i) {
			auto& aa = map[i];
			const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
			    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
			fmt::format_to(std::back_inserter(acc_log), "{} A{} {}", i == 0 ? ", accessing" : ",", aa.aid, accessed_box_in_allocation);
		}
		return acc_log;
	};

	// TODO submit synchronous operations to thread pool
	return matchbox::match<event>(
	    instr,
	    [&](const alloc_instruction& ainstr) {
		    CELERITY_DEBUG("[executor] I{}: alloc M{}.A{} with {}%{} bytes", ainstr.get_id(), ainstr.get_memory_id(), ainstr.get_allocation_id(),
		        ainstr.get_size(), ainstr.get_alignment());

		    auto [ptr, event] = m_backend_queue->malloc(ainstr.get_memory_id(), ainstr.get_size(), ainstr.get_alignment());
		    m_allocations.emplace(ainstr.get_allocation_id(), allocation{ainstr.get_memory_id(), ptr});
		    return std::move(event);
	    },
	    [&](const free_instruction& finstr) {
		    const auto it = m_allocations.find(finstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto [memory, ptr] = it->second;

		    CELERITY_DEBUG("[executor] I{}: free M{}.A{}", finstr.get_id(), memory, finstr.get_allocation_id());

		    m_allocations.erase(it);
		    return m_backend_queue->free(memory, ptr);
	    },
	    [&](const init_buffer_instruction& ibinstr) {
		    CELERITY_DEBUG("[executor] I{}: init B{} as M0.A{}, {} bytes", ibinstr.get_id(), ibinstr.get_buffer_id(), ibinstr.get_host_allocation_id(),
		        ibinstr.get_size());

		    const auto user_ptr = m_buffer_user_pointers.at(ibinstr.get_buffer_id());
		    const auto host_ptr = m_allocations.at(ibinstr.get_host_allocation_id()).pointer;
		    memcpy(host_ptr, user_ptr, ibinstr.get_size());
		    return completed_synchronous();
	    },
	    [&](const export_instruction& einstr) {
		    CELERITY_DEBUG("[executor] I{}: export M0.A{} ({})+{}, {}x{} bytes", einstr.get_id(), einstr.get_host_allocation_id(),
		        einstr.get_allocation_range(), einstr.get_offset_in_allocation(), einstr.get_copy_range(), einstr.get_element_size());

		    const auto dest_ptr = einstr.get_out_pointer(); // TODO very naughty
		    const auto source_base_ptr = m_allocations.at(einstr.get_host_allocation_id()).pointer;

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

		    const auto source_base_ptr = m_allocations.at(cinstr.get_source_allocation()).pointer;
		    const auto dest_base_ptr = m_allocations.at(cinstr.get_dest_allocation()).pointer;
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
	    [&](const sycl_kernel_instruction& skinstr) {
		    CELERITY_DEBUG("[executor] I{}: launch SYCL kernel on D{}, {}{}", skinstr.get_id(), skinstr.get_device_id(), skinstr.get_execution_range(),
		        log_accesses(skinstr.get_allocation_map()));

		    std::vector<closure_hydrator::accessor_info> accessor_infos;
		    accessor_infos.reserve(skinstr.get_allocation_map().size());
		    for(const auto& aa : skinstr.get_allocation_map()) {
			    const auto ptr = m_allocations.at(aa.aid).pointer;
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
			    accessor_infos.push_back(closure_hydrator::accessor_info{ptr, aa.allocated_box_in_buffer, aa.accessed_box_in_buffer, nullptr /* TODO */});
#else
			    accessor_infos.push_back(closure_hydrator::accessor_info{ptr, aa.allocated_box_in_buffer, aa.accessed_box_in_buffer, aa.buffer_subrange});
#endif
		    }

		    std::vector<void*> reduction_ptrs;     // TODO
		    bool is_reduction_initializer = false; // TODO

		    closure_hydrator::get_instance().arm(target::device, std::move(accessor_infos));

		    return m_backend_queue->launch_kernel(
		        skinstr.get_device_id(), skinstr.get_launcher(), skinstr.get_execution_range(), reduction_ptrs, is_reduction_initializer);
	    },
	    [&](const host_kernel_instruction& hkinstr) {
		    CELERITY_DEBUG(
		        "[executor] I{}: launch host task, {}{}", hkinstr.get_id(), hkinstr.get_execution_range(), log_accesses(hkinstr.get_allocation_map()));

		    std::vector<closure_hydrator::accessor_info> accessor_infos;
		    accessor_infos.reserve(hkinstr.get_allocation_map().size());
		    for(const auto& aa : hkinstr.get_allocation_map()) {
			    const auto ptr = m_allocations.at(aa.aid).pointer;
			    accessor_infos.push_back(closure_hydrator::accessor_info{ptr, aa.allocated_box_in_buffer, aa.accessed_box_in_buffer});
		    }

		    closure_hydrator::get_instance().arm(target::host_task, std::move(accessor_infos));

		    const auto& launch = hkinstr.get_launcher();
		    return launch(m_host_queue, hkinstr.get_execution_range());
	    },
	    [&](const send_instruction& sinstr) {
		    CELERITY_DEBUG("[executor] I{}: send M{}.A{}+{}, {}x{} bytes to N{} (tag {})", sinstr.get_id(), host_memory_id /* TODO RDMA */,
		        sinstr.get_source_allocation_id(), sinstr.get_offset_in_allocation(), sinstr.get_send_range(), sinstr.get_element_size(),
		        sinstr.get_dest_node_id(), sinstr.get_tag());

		    const auto allocation_base = m_allocations.at(sinstr.get_source_allocation_id()).pointer;
		    const communicator::stride stride{
		        sinstr.get_allocation_range(),
		        subrange<3>{sinstr.get_offset_in_allocation(), sinstr.get_send_range()},
		        sinstr.get_element_size(),
		    };
		    return m_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_tag(), allocation_base, stride);
	    },
	    [&](const recv_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: recv M{}.A{}+{}, {}x{} bytes (transfer id {})", rinstr.get_id(), rinstr.get_dest_memory_id(),
		        rinstr.get_dest_allocation_id(), rinstr.get_offset_in_allocation(), rinstr.get_recv_range(), rinstr.get_element_size(),
		        rinstr.get_transfer_id());

		    const auto allocation_base = m_allocations.at(rinstr.get_dest_allocation_id()).pointer;
		    return m_recv_arbiter.begin_aggregated_recv(rinstr.get_transfer_id(), rinstr.get_buffer_id(), allocation_base, rinstr.get_allocation_range(),
		        rinstr.get_offset_in_buffer(), rinstr.get_offset_in_allocation(), rinstr.get_recv_range(), rinstr.get_element_size());
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
			    MPI_Barrier(MPI_COMM_WORLD); // TODO this should not be in executor
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

void instruction_executor::inbound_pilot_received(const inbound_pilot& pilot) { m_inbound_pilot_queue.push_back(pilot); }

} // namespace celerity::detail
