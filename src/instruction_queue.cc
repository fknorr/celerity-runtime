#include "instruction_queue.h"

#include "allocation_manager.h"
#include "closure_hydrator.h"
#include "instruction_graph.h"

namespace celerity::detail {

// TODO host_thread_queues should never actually have to wait_on each other - this "host dependency" should be resolved by stalling the submission instead. That
// way we never unnecessarily serialize execution by blocking queue threads.
class host_thread_queue : public in_order_instruction_queue {
  public:
	using tick = uint64_t;

	class event : public instruction_queue_event_impl {
	  public:
		event(host_thread_queue& q, const tick t) : m_queue(&q), m_tick(t) {}

		host_thread_queue* get_queue() const { return m_queue; }
		tick get_tick() const { return m_tick; }

		bool has_completed() const override { return m_queue->has_completed(m_tick); }
		void block_on() override { m_queue->block_on(m_tick); }

	  private:
		host_thread_queue* m_queue;
		tick m_tick;
	};

	host_thread_queue() : m_thread(&host_thread_queue::thread_main, this) {}
	host_thread_queue(const host_thread_queue&) = delete;
	host_thread_queue& operator=(const host_thread_queue&) = delete;
	~host_thread_queue() { push(stop{}); }

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override {
		auto op = utils::match(
		    *instr, //
		    [&](const host_kernel_instruction& hkinstr) { return hkinstr.bind(MPI_COMM_WORLD /* TODO have a communicator registry */); },
		    [](const auto& /* other */) -> operation { panic("invalid instruction type for host_thread_queue"); });
		const auto t = push(std::move(op));
		return std::make_shared<event>(*this, t);
	}

	void wait_on(const instruction_queue_event& evt) override {
		auto& htevt = dynamic_cast<const event&>(*evt);
		if(htevt.get_queue() == this) return; // trivial
		push(wait_on_other_queue(htevt.get_queue(), htevt.get_tick()));
	}

  private:
	using operation = std::function<void()>;
	struct stop {};
	using wait_on_other_queue = std::tuple<host_thread_queue*, tick>;
	using token = std::variant<operation, wait_on_other_queue, stop>;

	std::thread m_thread;

	mutable std::mutex m_queue_mutex;
	tick m_next_tick = 1;
	std::queue<token> m_queue;
	std::condition_variable m_queue_nonempty;

	mutable std::mutex m_tick_mutex;
	tick m_last_tick = 0;
	std::condition_variable m_tick;

	void thread_main() {
		for(;;) {
			token next;
			{
				std::unique_lock lock(m_queue_mutex);
				while(m_queue.empty()) {
					m_queue_nonempty.wait(lock);
				}
				next = std::move(m_queue.front());
				m_queue.pop();
			}

			if(std::holds_alternative<stop>(next)) break;

			utils::match(
			    next,
			    [](operation& op) {
				    try {
					    op();
				    } catch(std::exception& e) {
					    panic("Exception in host thread queue: {}", e.what()); //
				    } catch(...) {
					    panic("Exception in host thread queue"); //
				    }
			    },
			    [](const wait_on_other_queue& wait) {
				    const auto& [queue, tick] = wait;
				    queue->block_on(tick); // TODO this is a bad idea
			    },
			    [](const auto& /* other */) { panic("unreachable"); });

			std::lock_guard lock(m_tick_mutex);
			m_last_tick++;
			m_tick.notify_all();
		}
	}

	tick push(token&& next) {
		tick t;
		{
			std::lock_guard lock(m_queue_mutex);
			t = m_next_tick++;
			m_queue.push(std::move(next));
		}
		m_queue_nonempty.notify_one();
		return t;
	}

	bool has_completed(tick t) const {
		std::lock_guard lock(m_tick_mutex);
		return m_last_tick >= t;
	}

	void block_on(tick t) {
		std::unique_lock lock(m_tick_mutex);
		while(m_last_tick < t) {
			m_tick.wait(lock);
		}
	}
};

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
