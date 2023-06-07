#include "host_instruction_queue.h"

#include "instruction_graph.h"

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>

namespace celerity::detail {

class host_instruction_queue_event : public instruction_queue_event_impl {
  public:
	host_instruction_queue_event() = default;
	host_instruction_queue_event(const host_instruction_queue_event&) = delete;
	host_instruction_queue_event& operator=(const host_instruction_queue_event&) = delete;

	bool has_completed() const override {
		std::lock_guard lock(m_mutex);
		return m_completed;
	}

	void mark_as_completed() {
		std::lock_guard lock(m_mutex);
		m_completed = true;
		m_updated.notify_all();
	}

	void block_on() override {
		std::unique_lock lock(m_mutex);
		while(!m_completed) {
			m_updated.wait(lock);
		}
	}

  private:
	// TODO this might be pretty heavy to instantiate, but we only need it when the user actually calls block_on.
	// Use an atomic and optionally instantiate mutex + condvar in the call to block_on.
	mutable std::mutex m_mutex;
	bool m_completed = false;
	std::condition_variable m_updated;
};

struct host_instruction_queue::impl {
	using operation = std::function<void()>;

	std::mutex mutex;
	std::queue<operation> submissions;
	bool no_more_submissions = false;
	std::condition_variable updated;

	std::vector<std::thread> m_threads;

	impl(size_t num_threads);
	impl(impl&&) = delete;
	impl& operator=(impl&&) = delete;

	~impl();

	instruction_queue_event submit(std::unique_ptr<instruction> instr);
	void thread_main();
};

host_instruction_queue::impl::impl(const size_t num_threads) {
	m_threads.reserve(num_threads);
	for(size_t i = 0; i < num_threads; ++i) {
		m_threads.emplace_back(&impl::thread_main, this);
	}
}

host_instruction_queue::impl::~impl() {
	{
		std::lock_guard lock(mutex);
		no_more_submissions = true;
	}
	updated.notify_all();
}

instruction_queue_event host_instruction_queue::impl::submit(std::unique_ptr<instruction> instr) {
	auto op = utils::match(
	    *instr, //
	    [&](const host_kernel_instruction& hkinstr) { return hkinstr.bind(MPI_COMM_WORLD /* TODO have a communicator registry */); },
	    [](const auto& /* other */) -> operation { panic("invalid instruction type for host_thread_queue"); });
	auto event = std::make_shared<host_instruction_queue_event>();

	{
		std::lock_guard lock(mutex);
		submissions.push([op = std::move(op), event] {
			op();
			event->mark_as_completed();
		});
	}
	updated.notify_one();

	return std::move(event);
}

void host_instruction_queue::impl::thread_main() {
	std::unique_lock lock(mutex);
	for(;;) {
		while(submissions.empty() && !no_more_submissions) {
			updated.wait(lock);
		}
		if(submissions.empty() && no_more_submissions) break;

		auto op = std::move(submissions.front());
		submissions.pop();
		lock.unlock();

		op();

		lock.lock();
	}
}

host_instruction_queue::host_instruction_queue(const size_t num_threads) : m_impl(std::make_unique<impl>(num_threads)) { assert(num_threads > 0); }

instruction_queue_event host_instruction_queue::submit(std::unique_ptr<instruction> instr) {
	assert(m_impl != nullptr);
	return m_impl->submit(std::move(instr));
}

} // namespace celerity::detail
