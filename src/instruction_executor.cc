#include "instruction_executor.h"
#include "buffer_storage.h" // for memcpy_strided_host
#include "closure_hydrator.h"
#include "communicator.h"
#include "recv_arbiter.h"

#include <atomic>
#include <mutex>

namespace celerity::detail {

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, const communicator_factory& comm_factory, delegate* dlg)
    : m_delegate(dlg), m_backend_queue(std::move(backend_queue)), m_communicator(comm_factory.make_communicator(this)), m_recv_arbiter(*m_communicator),
      m_thread(&instruction_executor::loop, this) {}

instruction_executor::~instruction_executor() { m_thread.join(); }

void instruction_executor::submit(const instruction& instr) { m_submission_queue.push_back(&instr); }

void instruction_executor::announce_buffer_user_pointer(const buffer_id bid, const void* const ptr) { m_submission_queue.push_back(std::pair(bid, ptr)); }

void instruction_executor::loop() {
	set_thread_name(get_current_thread_handle(), "cy-executor");
	closure_hydrator::make_available();

	struct pending_instruction_info {
		size_t n_unmet_dependencies;
	};
	struct active_instruction_info {
		event completion_event;

		bool is_complete() const {
			return utils::match(
			    completion_event, //
			    [](const std::unique_ptr<backend::event>& evt) { return evt->is_complete(); },
			    [](const std::unique_ptr<communicator::event>& evt) { return evt->is_complete(); },
			    [](const recv_arbiter::event& evt) { return evt.is_complete(); },
			    [](const std::future<host_queue::execution_info>& future) { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; },
			    [](const completed_synchronous) { return true; });
		};
	};

	std::vector<submission> loop_submission_queue;
	std::vector<std::pair<node_id, pilot_message>> loop_pilot_queue;
	std::unordered_map<const instruction*, pending_instruction_info> pending_instructions;
	std::vector<const instruction*> ready_instructions;
	std::unordered_map<const instruction*, active_instruction_info> active_instructions;
	while(m_expecting_more_submissions || !pending_instructions.empty() || !active_instructions.empty()) {
		for(auto active_it = active_instructions.begin(); active_it != active_instructions.end();) {
			auto& [active_instr, active_info] = *active_it;
			if(active_info.is_complete()) {
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
			for(const auto& submission : loop_submission_queue) {
				utils::match(
				    submission,
				    [&](const instruction* incoming_instr) {
					    const auto n_unmet_dependencies = static_cast<size_t>(std::count_if(
					        incoming_instr->get_dependencies().begin(), incoming_instr->get_dependencies().end(), [&](const instruction::dependency& dep) {
						        return pending_instructions.count(dep.node) != 0 || active_instructions.count(dep.node) != 0;
					        }));
					    if(n_unmet_dependencies > 0) {
						    pending_instructions.emplace(incoming_instr, pending_instruction_info{n_unmet_dependencies});
					    } else {
						    ready_instructions.push_back(incoming_instr);
					    }
				    },
				    [&](const std::pair<buffer_id, const void*> buffer_host_ptr) {
					    const auto [bid, ptr] = buffer_host_ptr;
					    assert(m_buffer_user_pointers.count(bid) == 0);
					    m_buffer_user_pointers.emplace(bid, ptr);
				    });
			}
			loop_submission_queue.clear();
		}

		if(m_pilot_queue.swap_if_nonempty(loop_pilot_queue)) {
			for(auto& [source, pilot] : loop_pilot_queue) {
				m_recv_arbiter.push_pilot_message(source, pilot);
			}
			loop_pilot_queue.clear();
		}

		for(const auto ready_instr : ready_instructions) {
			active_instructions.emplace(ready_instr, active_instruction_info{begin_executing(*ready_instr)});
		}
		ready_instructions.clear();
	}
}

instruction_executor::event instruction_executor::begin_executing(const instruction& instr) {
	// TODO submit synchronous operations to thread pool
	return utils::match(
	    instr,
	    [&](const alloc_instruction& ainstr) -> event {
		    auto [ptr, event] = m_backend_queue->malloc(ainstr.get_memory_id(), ainstr.get_size(), ainstr.get_alignment());
		    m_allocations.emplace(ainstr.get_allocation_id(), allocation{ainstr.get_memory_id(), ptr});
		    return std::move(event);
	    },
	    [&](const free_instruction& ainstr) -> event {
		    const auto it = m_allocations.find(ainstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto [memory, ptr] = it->second;
		    m_allocations.erase(it);
		    return m_backend_queue->free(memory, ptr);
	    },
	    [&](const init_buffer_instruction& ibinstr) -> event {
		    const auto user_ptr = m_buffer_user_pointers.at(ibinstr.get_buffer_id());
		    const auto host_ptr = m_allocations.at(ibinstr.get_host_allocation_id()).pointer;
		    memcpy(host_ptr, user_ptr, ibinstr.get_size());
		    return completed_synchronous();
	    },
	    [&](const copy_instruction& ainstr) -> event {
		    const auto source_base_ptr = m_allocations.at(ainstr.get_source_allocation()).pointer;
		    const auto dest_base_ptr = m_allocations.at(ainstr.get_dest_allocation()).pointer;
		    if(ainstr.get_source_memory() == host_memory_id && ainstr.get_dest_memory() == host_memory_id) {
			    const auto dispatch_copy = [&](const auto dims) {
				    memcpy_strided_host(source_base_ptr, dest_base_ptr, ainstr.get_element_size(), range_cast<dims.value>(ainstr.get_source_range()),
				        id_cast<dims.value>(ainstr.get_offset_in_source()), range_cast<dims.value>(ainstr.get_dest_range()),
				        id_cast<dims.value>(ainstr.get_offset_in_dest()), range_cast<dims.value>(ainstr.get_copy_range()));
			    };
			    switch(ainstr.get_dimensions()) {
			    case 0: dispatch_copy(std::integral_constant<int, 0>()); break;
			    case 1: dispatch_copy(std::integral_constant<int, 1>()); break;
			    case 2: dispatch_copy(std::integral_constant<int, 2>()); break;
			    case 3: dispatch_copy(std::integral_constant<int, 3>()); break;
			    default: abort();
			    }
			    return completed_synchronous();
		    } else {
			    return m_backend_queue->memcpy_strided_device(ainstr.get_dimensions(), ainstr.get_source_memory(), ainstr.get_dest_memory(), source_base_ptr,
			        dest_base_ptr, ainstr.get_element_size(), ainstr.get_source_range(), ainstr.get_offset_in_source(), ainstr.get_dest_range(),
			        ainstr.get_offset_in_dest(), ainstr.get_copy_range());
		    }
	    },
	    [&](const sycl_kernel_instruction& skinstr) -> event {
		    std::vector<closure_hydrator::accessor_info> accessor_infos;
		    accessor_infos.reserve(skinstr.get_allocation_map().size());
		    for(const auto& aa : skinstr.get_allocation_map()) {
			    const auto ptr = m_allocations.at(aa.aid).pointer;
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
			    accessor_infos.push_back(
			        closure_hydrator::accessor_info{ptr, aa.allocation_range, aa.offset_in_allocation, aa.buffer_subrange, nullptr /* TODO */});
#else
			    accessor_infos.push_back(closure_hydrator::accessor_info{ptr, aa.allocation_range, aa.offset_in_allocation, aa.buffer_subrange});
#endif
		    }

		    std::vector<void*> reduction_ptrs;     // TODO
		    bool is_reduction_initializer = false; // TODO

		    closure_hydrator::get_instance().arm(target::device, std::move(accessor_infos));

		    return m_backend_queue->launch_kernel(
		        skinstr.get_device_id(), skinstr.get_launcher(), skinstr.get_execution_range(), reduction_ptrs, is_reduction_initializer);
	    },
	    [&](const host_kernel_instruction& hkinstr) -> event {
		    std::vector<closure_hydrator::accessor_info> accessor_infos;
		    accessor_infos.reserve(hkinstr.get_allocation_map().size());
		    for(const auto& aa : hkinstr.get_allocation_map()) {
			    const auto ptr = m_allocations.at(aa.aid).pointer;
			    accessor_infos.push_back(closure_hydrator::accessor_info{ptr, aa.allocation_range, aa.offset_in_allocation, aa.buffer_subrange});
		    }

		    closure_hydrator::get_instance().arm(target::host_task, std::move(accessor_infos));

		    const auto& launch = hkinstr.get_launcher();
		    return launch(m_host_queue, hkinstr.get_execution_range());
	    },
	    [&](const send_instruction& sinstr) -> event {
		    const auto allocation_base = m_allocations.at(sinstr.get_source_allocation_id()).pointer;
		    const communicator::stride stride{
		        sinstr.get_allocation_range(),
		        subrange<3>{sinstr.get_offset_in_allocation(), sinstr.get_send_range()},
		        sinstr.get_element_size(),
		    };
		    return m_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_tag(), allocation_base, stride);
	    },
	    [&](const recv_instruction& rinstr) -> event {
		    const auto allocation_base = m_allocations.at(rinstr.get_dest_allocation_id()).pointer;
		    return m_recv_arbiter.begin_aggregated_recv(rinstr.get_transfer_id(), rinstr.get_buffer_id(), allocation_base, rinstr.get_allocation_range(),
		        rinstr.get_offset_in_buffer(), rinstr.get_offset_in_allocation(), rinstr.get_recv_range(), rinstr.get_element_size());
	    },
	    [&](const horizon_instruction& hinstr) -> event {
		    if(m_delegate != nullptr) { m_delegate->horizon_reached(hinstr.get_horizon_task_id()); }
		    return completed_synchronous();
	    },
	    [&](const epoch_instruction& einstr) -> event {
		    switch(einstr.get_epoch_action()) {
		    case epoch_action::none: break;
		    case epoch_action::barrier:
			    MPI_Barrier(MPI_COMM_WORLD); // TODO this should not be in executor
			    break;
		    case epoch_action::shutdown: m_expecting_more_submissions = false; break;
		    }
		    if(m_delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
			    m_delegate->epoch_reached(einstr.get_epoch_task_id());
		    }
		    return completed_synchronous();
	    });
}

void instruction_executor::pilot_message_received(const node_id from, const pilot_message& pilot) { m_pilot_queue.push_back(std::pair{from, pilot}); }

} // namespace celerity::detail
