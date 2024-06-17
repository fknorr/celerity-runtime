#pragma once

#include "async_event.h"
#include "double_buffered_queue.h"
#include "named_threads.h"
#include "tracy.h"
#include "utils.h"

#include <future>
#include <thread>

namespace celerity::detail {

class thread_queue_event : public async_event_impl {
  public:
	explicit thread_queue_event(std::future<void*> result) : m_result(std::move(result)) {}

	bool is_complete() const override {
		if(!m_result.valid()) return true;
		return m_result.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	void* take_result() override { return m_result.get(); }

  private:
	std::future<void*> m_result;
};

class thread_queue {
  public:
	thread_queue() : thread_queue("cy-thread") {}
	explicit thread_queue(std::string name) : m_impl(new impl(std::move(name))) {}

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
		job job(std::in_place, std::forward<Fn>(fn));
		auto evt = make_async_event<thread_queue_event>(job.promise.get_future());
		m_impl->queue.push(std::move(job));
		return evt;
	}

  private:
	struct job {
		std::function<void*()> fn;
		std::promise<void*> promise;

		job() = default; // empty (default-constructed) fn signals termination

		template <typename Fn>
		job(std::in_place_t /* tag */, Fn&& fn)
		    : fn([fn = std::forward<Fn>(fn)] {
			      if constexpr(std::is_void_v<decltype(fn())>) {
				      fn();
				      return nullptr;
			      } else {
				      return fn();
			      }
		      }) {}
	};

	struct impl {
		double_buffered_queue<job> queue;
		std::thread thread;

		explicit impl(std::string name) : thread(&impl::thread_main, this, std::move(name)) {}

		void loop() {
			for(;;) {
				queue.wait_while_empty();
				for(auto& job : queue.pop_all()) {
					if(!job.fn) return;
					job.promise.set_value(job.fn());
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

static_assert(std::is_constructible_v<thread_queue, std::string>);

} // namespace celerity::detail
