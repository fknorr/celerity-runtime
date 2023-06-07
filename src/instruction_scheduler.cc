#include "instruction_scheduler.h"

#include "allocation_manager.h"
#include "instruction_graph.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <variant>

namespace celerity::detail {

void instruction_scheduler::submit(std::unique_ptr<instruction> instr) {
	const auto iid = instr->get_id();
	const auto backend = instr->get_backend();
	const auto backend_supports_graph_ordering = m_delegate->backend_supports_graph_ordering(backend);

	size_t num_inorder_dependencies = 0;
	for(const auto& dep : instr->get_dependencies()) {
		utils::match(
		    m_visible_instructions.at(dep.node->get_id()),
		    [&](pending_instruction_info& dep_piinfo) {
			    if(dep_piinfo.instr->get_backend() == backend && backend_supports_graph_ordering) {
				    dep_piinfo.inorder_submission_dependents.push_back(iid);
			    } else {
				    dep_piinfo.inorder_completion_dependents.push_back(iid);
			    }
			    num_inorder_dependencies += 1;
		    },
		    [&](active_instruction_info& dep_aiinfo) {
			    if(dep_aiinfo.backend != backend) {
				    dep_aiinfo.inorder_completion_dependents.push_back(iid);
				    num_inorder_dependencies += 1;
			    }
		    },
		    [](const finished_instruction_info&) {});
	}

	if(num_inorder_dependencies == 0) {
		auto event = submit_to_backend(std::move(instr));
		m_visible_instructions.emplace(iid, active_instruction_info{backend, std::move(event), {}});
	} else {
		m_visible_instructions.emplace(iid, pending_instruction_info{std::move(instr), num_inorder_dependencies, {}, {}});
	}
}

instruction_scheduler::poll_action instruction_scheduler::poll_events() {
	if(m_poll_set.empty()) return poll_action::idle_until_next_submit;

	const auto first_completed = std::partition(m_poll_set.begin(), m_poll_set.end(), [](const auto& pair) { return !pair.second->has_completed(); });
	if(first_completed == m_poll_set.end()) return poll_action::poll_again;

	std::vector<instruction_info*> ready_set;
	for(auto it = first_completed; it != m_poll_set.end(); ++it) {
		const auto [iid, evt] = *it;
		auto& iinfo = m_visible_instructions.at(iid);
		auto& aiinfo = std::get<active_instruction_info>(iinfo);
		fulfill_dependency(aiinfo.inorder_completion_dependents, ready_set);
		iinfo = finished_instruction_info{};
	}

	m_poll_set.erase(first_completed, m_poll_set.end());

	std::vector<instruction_info*> next_ready_set;
	while(!ready_set.empty()) {
		for(const auto iinfo : ready_set) {
			auto& piinfo = std::get<pending_instruction_info>(*iinfo);
			fulfill_dependency(piinfo.inorder_submission_dependents, next_ready_set);

			const auto backend = piinfo.instr->get_backend();
			auto event = submit_to_backend(std::move(piinfo.instr));
			*iinfo = active_instruction_info{backend, std::move(event), std::move(piinfo.inorder_completion_dependents)};
		}
		ready_set.clear();
		std::swap(ready_set, next_ready_set);
	}

	return m_poll_set.empty() ? poll_action::idle_until_next_submit : poll_action::poll_again;
}

void instruction_scheduler::fulfill_dependency(const std::vector<instruction_id>& dependents, std::vector<instruction_info*>& out_ready_set) {
	for(const auto dependent_iid : dependents) {
		auto& dependent_iinfo = m_visible_instructions.at(dependent_iid);
		auto& dependent_piinfo = std::get<pending_instruction_info>(dependent_iinfo);
		assert(dependent_piinfo.num_inorder_dependencies > 0);
		dependent_piinfo.num_inorder_dependencies -= 1;
		if(dependent_piinfo.num_inorder_dependencies == 0) { out_ready_set.push_back(&dependent_iinfo); }
	}
}

instruction_queue_event instruction_scheduler::submit_to_backend(std::unique_ptr<instruction> instr) {
	const auto iid = instr->get_id();
	const auto deps = instr->get_dependencies();

	std::vector<instruction_queue_event> backend_deps;
	for(auto& dep : deps) {
		const auto& dep_iinfo = m_visible_instructions.at(dep.node->get_id());
		if(!std::holds_alternative<finished_instruction_info>(dep_iinfo)) {
			const auto& dep_aiinfo = std::get<active_instruction_info>(dep_iinfo);
			assert(dep_aiinfo.backend == instr->get_backend());
			backend_deps.push_back(dep_aiinfo.event);
		}
	}
	assert(backend_deps.empty() || m_delegate->backend_supports_graph_ordering(instr->get_backend()));

	auto event = backend_deps.empty() //
	                 ? m_delegate->submit_to_backend(std::move(instr))
	                 : m_delegate->submit_to_backend(std::move(instr), backend_deps);

	m_poll_set.emplace_back(iid, event.get());
	return event;
}

struct async_instruction_scheduler::impl {
	instruction_scheduler m_scheduler;

	std::atomic<bool> m_new_state = true;
	std::mutex m_mutex;
	std::queue<std::unique_ptr<instruction>> m_submission_queue;
	bool m_no_more_submissions = false;
	std::condition_variable m_resume_thread;

	std::thread m_thread;

	explicit impl(instruction_scheduler::delegate* delegate);
	impl(const impl&) = delete;
	impl& operator=(const impl&) = delete;
	~impl();

	void submit(std::unique_ptr<instruction> instr);
	void thread_main();
};

async_instruction_scheduler::impl::impl(instruction_scheduler::delegate* delegate) : m_scheduler(delegate), m_thread(&impl::thread_main, this) {}

async_instruction_scheduler::impl::~impl() {
	{
		std::lock_guard lock(m_mutex);
		m_no_more_submissions = true;
		m_new_state.store(true, std::memory_order_relaxed);
	}
	m_resume_thread.notify_one();
}

void async_instruction_scheduler::impl::submit(std::unique_ptr<instruction> instr) {
	{
		std::lock_guard lock(m_mutex);
		assert(!m_no_more_submissions);
		m_submission_queue.push(std::move(instr));
		m_new_state.store(true, std::memory_order_relaxed);
	}
	m_resume_thread.notify_one();
}

void async_instruction_scheduler::impl::thread_main() {
	using poll_action = instruction_scheduler::poll_action;

	auto next_action = poll_action::idle_until_next_submit;
	bool exit_when_idle = false;
	while(next_action == poll_action::poll_again || !exit_when_idle) {
		if(m_new_state.load(std::memory_order_relaxed)) {
			std::unique_lock lock(m_mutex);
			if(next_action == poll_action::idle_until_next_submit) {
				while(m_submission_queue.empty() && !m_no_more_submissions) {
					m_resume_thread.wait(lock);
				}
			}
			while(!m_submission_queue.empty()) {
				m_scheduler.submit(std::move(m_submission_queue.front()));
				m_submission_queue.pop();
			}
			exit_when_idle = m_no_more_submissions;
			m_new_state.store(false, std::memory_order_relaxed);
		}

		next_action = m_scheduler.poll_events();
	}
}

async_instruction_scheduler::async_instruction_scheduler(instruction_scheduler::delegate* delegate) : m_impl(std::make_unique<impl>(delegate)) {}

async_instruction_scheduler::~async_instruction_scheduler() = default;

void async_instruction_scheduler::submit(std::unique_ptr<instruction> instr) {
	assert(m_impl != nullptr);
	m_impl->submit(std::move(instr));
}

} // namespace celerity::detail
