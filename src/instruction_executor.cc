#include "instruction_executor.h"

#include "fmt_internals.h"
#include "instruction_backend.h"
#include "instruction_graph.h"
#include "instruction_queue.h"
#include "utils.h"

#include <type_traits>

namespace celerity::detail {

instruction_executor::instruction_executor(std::unique_ptr<allocation_manager> alloc_manager, std::unique_ptr<out_of_order_instruction_queue> host_queue,
    std::unique_ptr<out_of_order_instruction_queue> mpi_queue, device_queue_map device_queues)
    : m_alloc_manager(std::move(alloc_manager)), m_host_queue(std::move(host_queue)), m_mpi_queue(std::move(mpi_queue)),
      m_device_queues(std::move(device_queues)), m_scheduler(this) {
	using int_backend = std::underlying_type_t<instruction_backend>;
	std::fill(std::begin(m_backend_graph_ordering_support), std::end(m_backend_graph_ordering_support), true);

	assert(m_host_queue != nullptr);
	if(dynamic_cast<graph_order_instruction_queue*>(m_host_queue.get()) == nullptr) {
		m_backend_graph_ordering_support[static_cast<int_backend>(instruction_backend::host)] = false;
	}
	assert(m_mpi_queue != nullptr);
	if(dynamic_cast<graph_order_instruction_queue*>(m_mpi_queue.get()) == nullptr) {
		m_backend_graph_ordering_support[static_cast<int_backend>(instruction_backend::mpi)] = false;
	}
	for(const auto& [did_backend, queue] : m_device_queues) {
		assert(queue != nullptr);
		if(dynamic_cast<graph_order_instruction_queue*>(queue.get()) == nullptr) {
			const auto& [did, backend] = did_backend;
			m_backend_graph_ordering_support[static_cast<int_backend>(backend)] = false;
		}
	}
}

void instruction_executor::submit(std::unique_ptr<instruction> instr) { m_scheduler.submit(std::move(instr)); }

out_of_order_instruction_queue* instruction_executor::select_backend_queue(const instruction_backend backend, const device_id did) {
	const auto it = m_device_queues.find({did, backend});
	if(it == m_device_queues.end()) panic("no instruction queue for D{} on {}", did, backend);
	return it->second.get();
}

out_of_order_instruction_queue* instruction_executor::select_backend_queue(const instruction_backend backend, const std::initializer_list<memory_id>& mids) {
	if(backend == instruction_backend::host) {
		assert(std::all_of(mids.begin(), mids.end(), [](const memory_id mid) { return mid == host_memory_id; }));
		return m_host_queue.get();
	}

	out_of_order_instruction_queue* selected_queue = nullptr;
	for(const auto mid : mids) {
		if(mid == host_memory_id) continue;
		const device_id did = mid - 1;
		// even though we always select the first the queue of the first device in `mids`, we still enforce that there is a backend queue for all devices
		// (should be enough to diagnose an instruction attempting to copy from and Nvidia to an AMD device through CUDA, for example)
		const auto queue = select_backend_queue(backend, did);
		if(selected_queue == nullptr) selected_queue = queue;
	}
	if(selected_queue == nullptr) panic("no common instruction queue for M{} on {}", fmt::join(mids, ","), backend);
	return selected_queue;
}

out_of_order_instruction_queue* instruction_executor::select_backend_queue(const instruction& instr) {
	return utils::match(
	    instr, //
	    [&](const alloc_instruction& ainstr) { return select_backend_queue(ainstr.get_backend(), {ainstr.get_memory_id()}); },
	    [&](const free_instruction& finstr) { return select_backend_queue(finstr.get_backend(), {host_memory_id /* TODO finstr.get_memory_id() */}); },
	    [&](const copy_instruction& cinstr) {
		    return select_backend_queue(cinstr.get_backend(), {cinstr.get_source_memory(), cinstr.get_dest_memory()});
	    }, //
	    [&](const sycl_kernel_instruction& skinstr) { return select_backend_queue(instruction_backend::sycl, skinstr.get_device_id()); },
	    [&](const send_instruction&) { return m_mpi_queue.get(); }, [&](const recv_instruction&) { return m_mpi_queue.get(); },
	    [&](const host_kernel_instruction&) { return m_host_queue.get(); }, //
	    [&](const horizon_instruction&) { return m_host_queue.get(); },     //
	    [&](const epoch_instruction&) { return m_host_queue.get(); });
}

instruction_queue_event instruction_executor::submit_to_backend(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) {
	const auto target_queue = dynamic_cast<graph_order_instruction_queue*>(select_backend_queue(*instr));
	assert(target_queue != nullptr && "backend queue not support graph ordering even though we told the scheduler it does");
	return target_queue->submit(std::move(instr), std::move(dependencies));
}

instruction_queue_event instruction_executor::submit_to_backend(std::unique_ptr<instruction> instr) {
	const auto target_queue = select_backend_queue(*instr);
	return target_queue->submit(std::move(instr));
}

bool instruction_executor::backend_supports_graph_ordering(const instruction_backend backend) const {
	return backend != instruction_backend::host; // we enforce this through the types in device_queue_map
}

} // namespace celerity::detail
