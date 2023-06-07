#include "sycl_instruction_queue.h"

#include "allocation_manager.h"
#include "closure_hydrator.h"
#include "instruction_graph.h"

namespace celerity::detail {

class sycl_event_impl : public instruction_queue_event_impl {
  public:
	sycl_event_impl(sycl::event event) : m_event(std::move(event)) {}
	bool has_completed() const override {
		return m_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
	}
	void block_on() override { m_event.wait(); }
	const sycl::event& get() const { return m_event; }

  private:
	sycl::event m_event;
};

sycl_instruction_queue::sycl_instruction_queue(sycl::queue q, allocation_manager& am) : m_queue(std::move(q)), m_allocation_mgr(&am) {}

instruction_queue_event sycl_instruction_queue::submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) {
	auto sycl_event = m_queue.submit([&](sycl::handler& cgh) {
		for(auto& dep : dependencies) {
			cgh.depends_on(dynamic_cast<const sycl_event_impl&>(*dep).get());
		}
		utils::match(
		    *instr, //
		    [&](const sycl_kernel_instruction& skinstr) {
			    std::vector<closure_hydrator::NOCOMMIT_info> access_infos;
			    access_infos.reserve(skinstr.get_allocation_map().size());
			    for(const auto& aa : skinstr.get_allocation_map()) {
				    access_infos.push_back(closure_hydrator::NOCOMMIT_info{
				        target::device, m_allocation_mgr->get_pointer(aa.aid), aa.allocation_range, aa.offset_in_allocation, aa.buffer_subrange});
			    }
			    closure_hydrator::get_instance().prepare(std::move(access_infos));
			    skinstr.launch(cgh);
		    },
		    [](const auto& /* other */) { panic("invalid instruction type for sycl_queue"); });
	});
	return std::make_unique<sycl_event_impl>(std::move(sycl_event));
}

} // namespace celerity::detail
