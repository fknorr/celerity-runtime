#pragma once

#include "async_event.h"
#include "double_buffered_queue.h"
#include "named_threads.h"
#include "tracy.h"
#include "utils.h"

#include <future>
#include <thread>
#include <type_traits>
#include <variant>

namespace celerity::detail {

class thread_queue_event : public async_event_impl {
  public:
	struct completion {
		void* result = nullptr;
		std::optional<std::chrono::nanoseconds> execution_time;
	};

	explicit thread_queue_event(std::future<completion> future) : m_state(std::move(future)) {}

	bool is_complete() override { return get_completed() != nullptr; }

	void* get_result() override {
		const auto completed = get_completed();
		assert(completed);
		return completed->result;
	}

	std::optional<std::chrono::nanoseconds> get_native_execution_time() override {
		const auto completed = get_completed();
		assert(completed);
		return completed->execution_time;
	}

  private:
	std::variant<std::future<completion>, completion> m_state;

	completion* get_completed() {
		if(const auto completed = std::get_if<completion>(&m_state)) return completed;
		if(auto& future = std::get<std::future<completion>>(m_state); future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			return &m_state.emplace<completion>(future.get());
		}
		return nullptr;
	}
};

class thread_queue {
  public:
	thread_queue() : thread_queue("cy-thread") {}
	explicit thread_queue(std::string name, const bool enable_profiling = false) : m_impl(new impl(std::move(name), enable_profiling)) {}

	thread_queue(const thread_queue&) = delete;
	thread_queue(thread_queue&&) = default;
	thread_queue& operator=(const thread_queue&) = delete;
	thread_queue& operator=(thread_queue&&) = default;

	~thread_queue() {
		if(m_impl != nullptr) {
			m_impl->queue.push(job{});
			m_impl->thread.join();
		}
	}

	template <typename Fn>
	async_event submit(Fn&& fn) {
		assert(m_impl != nullptr);
		job job(std::forward<Fn>(fn));
		auto evt = make_async_event<thread_queue_event>(job.promise.get_future());
		m_impl->queue.push(std::move(job));
		return evt;
	}

  private:
	struct job {
		using completion = thread_queue_event::completion;

		std::function<void*()> fn;
		std::promise<completion> promise;

		job() = default; // empty (default-constructed) fn signals termination

		template <typename Fn, std::enable_if_t<std::is_same_v<std::invoke_result_t<Fn>, void>, int> = 0>
		job(Fn&& fn) : fn([fn = std::forward<Fn>(fn)] { return std::invoke(fn), nullptr; }) {}

		template <typename Fn, std::enable_if_t<std::is_invocable_r_v<void*, Fn>, int> = 0>
		job(Fn&& fn) : fn([fn = std::forward<Fn>(fn)] { return std::invoke(fn); }) {}
	};

	// pimpl'd to keep thread_queue movable
	struct impl {
		const bool enable_profiling;
		double_buffered_queue<job> queue;
		std::thread thread;

		explicit impl(std::string name, const bool enable_profiling) : enable_profiling(enable_profiling), thread(&impl::thread_main, this, std::move(name)) {}

		void execute(job& job) const {
			std::chrono::steady_clock::time_point start;
			if(enable_profiling) { start = std::chrono::steady_clock::now(); }

			thread_queue_event::completion completion;
			completion.result = job.fn();

			std::chrono::steady_clock::time_point end;
			if(enable_profiling) {
				const auto end = std::chrono::steady_clock::now();
				completion.execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
			}

			job.promise.set_value(completion);
		}

		void loop() {
			for(;;) {
				queue.wait_while_empty();
				for(auto& job : queue.pop_all()) {
					if(!job.fn) return;
					execute(job);
				}
			}
		}

		void thread_main(const std::string& name) {
			set_thread_name(get_current_thread_handle(), name);
			CELERITY_DETAIL_TRACY_SET_THREAD_NAME(name.c_str());

			try {
				loop();
			} catch(std::exception& e) { //
				utils::panic("exception in thread queue: {}", e.what());
			} catch(...) { //
				utils::panic("exception in thread queue");
			}
		}
	};

	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
