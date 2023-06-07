#include "instruction_queue.h"

#include "allocation_manager.h"
#include "closure_hydrator.h"
#include "instruction_graph.h"

namespace celerity::detail {

class multiplex_instruction_queue_event : public instruction_queue_event_impl {
  public:
	multiplex_instruction_queue_event(const size_t inorder_queue_index, instruction_queue_event&& inorder_queue_event)
	    : m_inorder_queue_index(inorder_queue_index), m_inorder_queue_event(std::move(inorder_queue_event)) {}

	size_t get_inorder_queue_index() const { return m_inorder_queue_index; }
	const instruction_queue_event& get_inorder_queue_event() const { return m_inorder_queue_event; }

	bool has_completed() const override { return m_inorder_queue_event->has_completed(); }
	void block_on() override { m_inorder_queue_event->block_on(); }

  private:
	size_t m_inorder_queue_index;
	instruction_queue_event m_inorder_queue_event;
};

multiplex_instruction_queue::multiplex_instruction_queue(std::vector<std::unique_ptr<in_order_instruction_queue>> in_order_queues)
    : m_inorder_queues(std::move(in_order_queues)) {
	assert(!m_inorder_queues.empty());
}

instruction_queue_event multiplex_instruction_queue::submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) {
	size_t target_queue_index;
	if(dependencies.empty()) {
		// Unconstrained case: choose a random queue.
		// TODO can we improve this by estimating queue occupancy from previously submitted events?
		target_queue_index = m_round_robin_inorder_queue_index++ % m_inorder_queues.size();
	} else if(dependencies.size() == 1) {
		// If there is exactly one dependency, we fulfill it by scheduling onto the same queue.
		// TODO this is not actually optimal. If we have already submitted intermediate instructions to the same queue but the new instruction does
		// _not_ depend on these intermediates, we will fail to exploit that concurrency.
		auto& evt = dynamic_cast<const multiplex_instruction_queue_event&>(*dependencies.front());
		target_queue_index = evt.get_inorder_queue_index();
	} else {
		// Choose a queue that we have a dependency on in order to omit at least one call to wait_on.
		// TODO try to be smarter about this:
		//   - there can be multiple dependencies to a single queue => choose the one with the highest number of dependencies
		//   - some dependencies might already be fulfilled when the job is submitted, estimate the likelihood of this condition by counting how many
		//     unrelated instructions we have submitted to that queue in the meantime
		// ... maybe we can even find some scheduling literature that applies here.
		std::vector<size_t> dependency_queues;
		for(const auto& dep : dependencies) {
			dependency_queues.emplace_back(dynamic_cast<const multiplex_instruction_queue_event&>(*dep).get_inorder_queue_index());
		}
		std::sort(dependency_queues.begin(), dependency_queues.end());
		dependency_queues.erase(std::unique(dependency_queues.begin(), dependency_queues.end()), dependency_queues.end());
		assert(!dependency_queues.empty());
		target_queue_index = dependency_queues[m_round_robin_inorder_queue_index++ % dependency_queues.size()];
		for(const auto& dep : dependencies) {
			if(dynamic_cast<const multiplex_instruction_queue_event&>(*dep).get_inorder_queue_index() != target_queue_index) {
				m_inorder_queues[target_queue_index]->wait_on(dep);
			}
		}
	}

	auto evt = m_inorder_queues[target_queue_index]->submit(std::move(instr));
	return std::make_shared<multiplex_instruction_queue_event>(target_queue_index, std::move(evt));
}

} // namespace celerity::detail
