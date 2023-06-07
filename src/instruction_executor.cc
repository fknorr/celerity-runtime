#include "instruction_executor.h"

#include "fmt_internals.h"
#include "instruction_queue.h"
#include "utils.h"

namespace celerity::detail {

instruction_executor::instruction_executor(
    std::unique_ptr<allocation_manager> alloc_manager, std::unique_ptr<out_of_order_instruction_queue> host_queue, device_queue_map device_queues)
    : instruction_scheduler(this), m_alloc_manager(std::move(alloc_manager)), m_host_queue(std::move(host_queue)), m_device_queues(std::move(device_queues)) {}

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
	    },
	    [&](const sycl_kernel_instruction& skinstr) { return select_backend_queue(instruction_backend::sycl, skinstr.get_device_id()); },
	    [&](const auto& /* default */) -> out_of_order_instruction_queue* { return m_host_queue.get(); });
}

instruction_queue_event instruction_executor::submit_to_backend(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) {
	const auto target_queue = select_backend_queue(*instr);
	return target_queue->submit(std::move(instr), std::move(dependencies));
}

} // namespace celerity::detail
