#include "instruction_executor.h"
#include "buffer_storage.h" // for memcpy_strided_host
#include "communicator.h"
#include <atomic>
#include <mutex>

namespace celerity::detail {

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, const communicator_factory& comm_factory, delegate* dlg)
    : m_delegate(dlg), m_backend_queue(std::move(backend_queue)), m_communicator(comm_factory.make_communicator(this)),
      m_thread(&instruction_executor::loop, this) {}

instruction_executor::~instruction_executor() {
	m_submission_queue.push_back(nullptr /* sentinel for "no more submissions" */);
	m_thread.join();
}

void instruction_executor::submit(const instruction& instr) { m_submission_queue.push_back(&instr); }

void instruction_executor::loop() {
	struct pending_instruction_info {
		size_t n_unmet_dependencies;
	};
	struct active_instruction_info {
		event completion_event;

		bool is_complete() const {
			return utils::match(
			    completion_event,                                                                   //
			    [](const std::unique_ptr<backend::event>& evt) { return evt->is_complete(); },      //
			    [](const std::unique_ptr<communicator::event>& evt) { return evt->is_complete(); }, //
			    [](const completed_synchronous) { return true; });
		};
	};

	std::vector<const instruction*> loop_submission_queue;
	std::vector<pilot_message> loop_pilot_queue;
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
							pending_instructions.erase(pending_instr);
							ready_instructions.push_back(pending_instr);
						}
					}
				}
				active_it = active_instructions.erase(active_it);
			} else {
				++active_it;
			}
		}

		if(m_submission_queue.swap_if_nonempty(loop_submission_queue)) {
			for(const auto incoming_instr : loop_submission_queue) {
				const auto n_unmet_dependencies = static_cast<size_t>(std::count_if(incoming_instr->get_dependencies().begin(),
				    incoming_instr->get_dependencies().end(),
				    [&](const instruction::dependency& dep) { return pending_instructions.count(dep.node) != 0 || active_instructions.count(dep.node) != 0; }));
				if(n_unmet_dependencies > 0) {
					pending_instructions.emplace(incoming_instr, pending_instruction_info{n_unmet_dependencies});
				} else {
					ready_instructions.push_back(incoming_instr);
				}
			}
			loop_submission_queue.clear();
		}

		if(m_pilot_queue.swap_if_nonempty(loop_pilot_queue)) {
			// TODO
			loop_pilot_queue.clear();
		}

		for(const auto ready_instr : ready_instructions) {
			active_instructions.emplace(ready_instr, active_instruction_info{begin_executing(*ready_instr)});
		}
		ready_instructions.clear();
	}
}

std::pair<void*, instruction_executor::event> instruction_executor::malloc(const memory_id memory, const size_t size, const size_t alignment) {
	if(memory == host_memory_id) {
		const auto ptr = std::aligned_alloc(alignment, size);
		return {ptr, completed_synchronous()};
	} else {
		auto [ptr, event] = m_backend_queue->malloc(memory, size, alignment);
		return {ptr, std::move(event)};
	}
}

instruction_executor::event instruction_executor::free(const memory_id memory, void* const ptr) {
	if(memory == host_memory_id) {
		std::free(ptr);
		return completed_synchronous();
	} else {
		return m_backend_queue->free(memory, ptr);
	}
}

instruction_executor::event instruction_executor::begin_executing(const instruction& instr) {
	// TODO submit synchronous operations to thread pool
	return utils::match(
	    instr,
	    [&](const alloc_instruction& ainstr) -> event {
		    auto [ptr, event] = malloc(ainstr.get_memory_id(), ainstr.get_size(), ainstr.get_alignment());
		    m_allocations.emplace(ainstr.get_allocation_id(), allocation{ainstr.get_memory_id(), ptr});
		    return std::move(event);
	    },
	    [&](const free_instruction& ainstr) -> event {
		    const auto it = m_allocations.find(ainstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto& [memory, ptr] = it->second;
		    auto evt = free(memory, ptr);
		    m_allocations.erase(it);
		    return evt;
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
		    return m_backend_queue->launch_kernel(skinstr.get_device_id(), skinstr.get_launcher(), skinstr.get_execution_range());
	    },
	    [](const host_kernel_instruction& hkinstr) -> event {
		    auto& launcher = hkinstr.get_launcher();
		    // TODO launch in thread pool
		    launcher(hkinstr.get_execution_range(), hkinstr.get_global_range(), MPI_COMM_WORLD); // TOOD MPI Communicators
		    return completed_synchronous();
	    },
	    [](const send_instruction& sinstr) -> event {
		    // TODO MPI_Isend(...)
		    return completed_synchronous();
	    },
	    [](const recv_instruction&) -> event {
		    // TODO find associated pilot; then MPI_Irecv(...)
		    return completed_synchronous();
	    },
	    [&](const horizon_instruction& hinstr) -> event {
		    if(m_delegate != nullptr) { m_delegate->instruction_checkpoint_reached(hinstr.get_horizon_task_id()); }
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
		    if(m_delegate != nullptr) { m_delegate->instruction_checkpoint_reached(einstr.get_epoch_task_id()); }
		    return completed_synchronous();
	    });
}

void instruction_executor::pilot_message_received(node_id from, const pilot_message& pilot) { m_pilot_queue.push_back(pilot); }

} // namespace celerity::detail
