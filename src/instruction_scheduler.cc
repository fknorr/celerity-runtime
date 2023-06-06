#include "instruction_scheduler.h"
#include "buffer_storage.h" // TODO included for CELERITY_CUDA_CHECK, consider moving that
#include "closure_hydrator.h"
#include "instruction_graph.h"
#include <cuda_runtime.h> // TODO move CUDA stuff to separate source

#include <condition_variable>
#include <memory>
#include <mutex>
#include <variant>

namespace celerity::detail {

class cuda_event final : public instruction_queue_event_impl {
  public:
	cuda_event() {
		cudaEvent_t event;
		CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
		m_event = std::unique_ptr<CUevent_st, deleter>(event);
	}

	bool has_completed() const override {
		switch(const auto result = cudaEventQuery(m_event.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

	void block_on() override { CELERITY_CUDA_CHECK(cudaEventSynchronize, m_event.get()); }
	cudaEvent_t get() const { return m_event.get(); }

  private:
	struct deleter {
		void operator()(cudaEvent_t event) { CELERITY_CUDA_CHECK(cudaEventDestroy, event); }
	};
	std::unique_ptr<CUevent_st, deleter> m_event;
};

class cuda_stream final : public in_order_instruction_queue {
  public:
	explicit cuda_stream(const int cuda_device_id, allocation_manager& am) : m_allocation_mgr(&am) {
		CELERITY_CUDA_CHECK(cudaSetDevice, cuda_device_id);
		cudaStream_t stream;
		CELERITY_CUDA_CHECK(cudaStreamCreate, &stream);
		m_stream = std::unique_ptr<CUstream_st, deleter>(stream);
	}

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override {
		utils::match(
		    *instr, //
		    [this](const alloc_instruction& ainstr) { submit_malloc(ainstr); }, [this](const free_instruction& finstr) { submit_free(finstr); },
		    [this](const copy_instruction& cinstr) { submit_copy(cinstr); },
		    [&](const auto& /* other */) { panic("Invalid instruction type on cuda_stream"); });
		auto evt = std::make_shared<cuda_event>();
		CELERITY_CUDA_CHECK(cudaEventRecord, evt->get(), m_stream.get());
		return evt;
	}

	void wait_on(const instruction_queue_event& evt) override {
		CELERITY_CUDA_CHECK(cudaStreamWaitEvent, m_stream.get(), dynamic_cast<const cuda_event&>(*evt).get(), 0);
	}

  private:
	struct deleter {
		void operator()(cudaStream_t stream) { CELERITY_CUDA_CHECK(cudaStreamDestroy, stream); }
	};
	std::unique_ptr<CUstream_st, deleter> m_stream;
	allocation_manager* m_allocation_mgr;

	void submit_malloc(const alloc_instruction& ainstr) const {
		void* ptr;
		CELERITY_CUDA_CHECK(cudaMallocAsync, &ptr, ainstr.get_size(), m_stream.get());
		assert(reinterpret_cast<uintptr_t>(ptr) % ainstr.get_alignment() == 0); // TODO handle large alignments
		m_allocation_mgr->begin_tracking(ainstr.get_allocation_id(), ptr);
	}

	void submit_free(const free_instruction& finstr) const {
		const auto ptr = m_allocation_mgr->get_pointer(finstr.get_allocation_id());
		CELERITY_CUDA_CHECK(cudaFreeAsync, ptr, m_stream.get());
		m_allocation_mgr->end_tracking(finstr.get_allocation_id());
	}

	void submit_copy(const copy_instruction& cinstr) const {
		const auto source_base_ptr = m_allocation_mgr->get_pointer(cinstr.get_source_allocation());
		const auto dest_base_ptr = m_allocation_mgr->get_pointer(cinstr.get_dest_allocation());

		switch(cinstr.get_dimensions()) {
		case 0: {
			CELERITY_CUDA_CHECK(cudaMemcpyAsync, dest_base_ptr, source_base_ptr, cinstr.get_element_size(), cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 1: {
			const auto source_base_offset = get_linear_index(range_cast<1>(cinstr.get_source_range()), id_cast<1>(cinstr.get_offset_in_source()));
			const auto dest_base_offset = get_linear_index(range_cast<1>(cinstr.get_dest_range()), id_cast<1>(cinstr.get_offset_in_dest()));
			CELERITY_CUDA_CHECK(cudaMemcpyAsync, static_cast<char*>(dest_base_ptr) + cinstr.get_element_size() * dest_base_offset,
			    static_cast<const char*>(source_base_ptr) + cinstr.get_element_size() * source_base_offset,
			    cinstr.get_copy_range()[0] * cinstr.get_element_size(), cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 2: {
			const auto source_base_offset = get_linear_index(range_cast<2>(cinstr.get_source_range()), id_cast<2>(cinstr.get_offset_in_source()));
			const auto dest_base_offset = get_linear_index(range_cast<2>(cinstr.get_dest_range()), id_cast<2>(cinstr.get_offset_in_dest()));
			CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, static_cast<char*>(dest_base_ptr) + cinstr.get_element_size() * dest_base_offset,
			    cinstr.get_dest_range()[1] * cinstr.get_element_size(),
			    static_cast<const char*>(source_base_ptr) + cinstr.get_element_size() * source_base_offset,
			    cinstr.get_source_range()[1] * cinstr.get_element_size(), cinstr.get_copy_range()[1] * cinstr.get_element_size(), cinstr.get_copy_range()[0],
			    cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 3: {
			cudaMemcpy3DParms parms = {};
			parms.srcPos =
			    make_cudaPos(cinstr.get_offset_in_source()[2] * cinstr.get_element_size(), cinstr.get_offset_in_source()[1], cinstr.get_offset_in_source()[0]);
			parms.srcPtr =
			    make_cudaPitchedPtr(const_cast<void*>(source_base_ptr), cinstr.get_source_range()[2] * cinstr.get_element_size(), cinstr.get_source_range()[2],
			        cinstr.get_source_range()[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
			parms.dstPos =
			    make_cudaPos(cinstr.get_offset_in_dest()[2] * cinstr.get_element_size(), cinstr.get_offset_in_dest()[1], cinstr.get_offset_in_dest()[0]);
			parms.dstPtr = make_cudaPitchedPtr(
			    dest_base_ptr, cinstr.get_dest_range()[2] * cinstr.get_element_size(), cinstr.get_dest_range()[2], cinstr.get_dest_range()[1]);
			parms.extent = {cinstr.get_copy_range()[2] * cinstr.get_element_size(), cinstr.get_copy_range()[1], cinstr.get_copy_range()[0]};
			parms.kind = cudaMemcpyDefault;
			CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, m_stream.get());
			break;
		}
		default: panic("copy dimensions out of range");
		}
	}
};

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

class sycl_queue : public out_of_order_instruction_queue {
  public:
	explicit sycl_queue(sycl::queue q, allocation_manager& am) : m_queue(std::move(q)), m_allocation_mgr(&am) {}

	instruction_queue_event submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) override {
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

  private:
	sycl::queue m_queue;
	allocation_manager* m_allocation_mgr;
};

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

// Turns multiple in_order_instruction_queues into an out_of_order_instruction_queue.
// Use cases: Host-task "thread pool", CUDA concurrency.
// TODO is this actually optimal for any of these applications?
//   - For CUDA we might want to have separate Kernel / D2H / H2D copy streams for maximum utilization
//   - Async submissions do not really do anything for host code, we should rather submit these just in time to avoid needlessly blocking inside host threads to
//     wait for events from their sibling queues
class multiplex_instruction_queue : public out_of_order_instruction_queue {
  public:
	// TODO consider just storing a pointer to the queue in the event base type
	using queue_index = uint32_t;

	class event : public instruction_queue_event_impl {
	  public:
		event(const queue_index idx, instruction_queue_event&& evt) : m_queue_index(idx), m_event(std::move(evt)) {}

		queue_index get_queue_index() const { return m_queue_index; }
		const instruction_queue_event& get_queue_event() const { return m_event; }

		bool has_completed() const override { return m_event->has_completed(); }
		void block_on() override { m_event->block_on(); }

	  private:
		queue_index m_queue_index;
		instruction_queue_event m_event;
	};

	explicit multiplex_instruction_queue(std::vector<std::unique_ptr<in_order_instruction_queue>> in_order_queues) : m_queues(std::move(in_order_queues)) {
		assert(!m_queues.empty());
	}

	instruction_queue_event submit(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies) override {
		queue_index target_queue_index;
		if(dependencies.empty()) {
			// Unconstrained case: choose a random queue.
			// TODO can we improve this by estimating queue occupancy from previously submitted events?
			target_queue_index = m_round_robin_index++ % m_queues.size();
		} else if(dependencies.size() == 1) {
			// If there is exactly one dependency, we fulfill it by scheduling onto the same queue.
			// TODO this is not actually optimal. If we have already submitted intermediate instructions to the same queue but the new instruction does
			// _not_ depend on these intermediates, we will fail to exploit that concurrency.
			auto& evt = dynamic_cast<const event&>(*dependencies.front());
			target_queue_index = evt.get_queue_index();
		} else {
			// Choose a queue that we have a dependency on in order to omit at least one call to wait_on.
			// TODO try to be smarter about this:
			//   - there can be multiple dependencies to a single queue => choose the one with the highest number of dependencies
			//   - some dependencies might already be fulfilled when the job is submitted, estimate the likelihood of this condition by counting how many
			//     unrelated instructions we have submitted to that queue in the meantime
			// ... maybe we can even find some scheduling literature that applies here.
			std::vector<queue_index> dependency_queues;
			for(const auto& dep : dependencies) {
				dependency_queues.emplace_back(dynamic_cast<const event&>(*dep).get_queue_index());
			}
			std::sort(dependency_queues.begin(), dependency_queues.end());
			dependency_queues.erase(std::unique(dependency_queues.begin(), dependency_queues.end()), dependency_queues.end());
			assert(!dependency_queues.empty());
			target_queue_index = dependency_queues[m_round_robin_index++ % dependency_queues.size()];
			for(const auto& dep : dependencies) {
				if(dynamic_cast<const event&>(*dep).get_queue_index() != target_queue_index) { m_queues[target_queue_index]->wait_on(dep); }
			}
		}

		auto evt = m_queues[target_queue_index]->submit(std::move(instr));
		return std::make_shared<event>(target_queue_index, std::move(evt));
	}

  private:
	std::vector<std::unique_ptr<in_order_instruction_queue>> m_queues;
	queue_index m_round_robin_index = 0;
};

class polling_instruction_observer {
  public:
	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = delete;
		delegate& operator=(const delegate&) = delete;
		virtual ~delegate() = default;
		virtual void event_completed(const instruction_id iid, instruction_queue_event event) = 0;
	};

	explicit polling_instruction_observer(delegate* delegate) : m_delegate(delegate), m_thread(&polling_instruction_observer::thread_main, this) {}

	polling_instruction_observer(const polling_instruction_observer&) = delete;
	polling_instruction_observer& operator=(const polling_instruction_observer&) = delete;

	~polling_instruction_observer() {
		{
			std::lock_guard lock(m_mutex);
			m_exit = true;
		}
		m_unpause.notify_one();
	}

	void notify_on_completion(const instruction_id iid, instruction_queue_event event) {
		bool maybe_paused;
		{
			std::lock_guard lock(m_mutex);
			maybe_paused = m_active_events.empty();
			m_active_events.emplace_back(iid, std::move(event));
		}
		if(maybe_paused) { m_unpause.notify_one(); }
	}

  private:
	std::mutex m_mutex;
	delegate* m_delegate;
	bool m_exit = false;
	std::condition_variable m_unpause;
	std::vector<std::pair<instruction_id, instruction_queue_event>> m_active_events;
	std::thread m_thread;

	void thread_main() {
		decltype(m_active_events) completed;
		std::unique_lock lock(m_mutex);
		for(;;) {
			while(!m_exit && m_active_events.empty()) {
				m_unpause.wait(lock); // TODO consider only pausing when the entire runtime is idle
			}
			if(m_exit) break;

			const auto first_completed =
			    std::remove_if(m_active_events.begin(), m_active_events.end(), [](const auto& pair) { return pair.second->has_completed(); });
			if(first_completed == m_active_events.end()) continue;

			completed.assign(std::make_move_iterator(first_completed), std::make_move_iterator(m_active_events.end()));
			m_active_events.erase(first_completed, m_active_events.end());

			lock.unlock();
			for(auto& [iid, evt] : completed) {
				m_delegate->event_completed(std::move(iid), evt);
			}
			completed.clear();
			lock.lock();
		}
	}
};

class instruction_backlog final : private polling_instruction_observer::delegate {
  public:
	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = delete;
		delegate& operator=(const delegate&) = delete;
		virtual ~delegate() = default;
		virtual void dependencies_satisfied(std::unique_ptr<instruction> instr) = 0;
	};

	explicit instruction_backlog(delegate* delegate) : m_delegate(delegate), m_observer(this) {}

	void push(std::unique_ptr<instruction> instr, std::vector<std::pair<instruction_id, instruction_queue_event>> incomplete_dependencies) {
		{
			std::lock_guard lock(m_mutex);
			const auto iid = instr->get_id();
			m_pending_instructions.emplace(iid, pending_instruction{std::move(instr), incomplete_dependencies.size()});
			for(const auto& [dep_iid, dep_event] : incomplete_dependencies) {
				m_waiting_on.emplace(dep_iid, iid);
			}
		}
		// m_observer is thread-safe, keep lock duration short
		for(auto& [dep_iid, dep_event] : incomplete_dependencies) {
			m_observer.notify_on_completion(dep_iid, std::move(dep_event));
		}
	}

  private:
	struct pending_instruction {
		std::unique_ptr<instruction> instr;
		size_t num_incomplete_dependencies;
	};

	std::mutex m_mutex;
	delegate* m_delegate;
	polling_instruction_observer m_observer;
	std::unordered_multimap<instruction_id /* dependency */, instruction_id /* pending instruction */> m_waiting_on;
	std::unordered_map<instruction_id, pending_instruction> m_pending_instructions;

	void event_completed(const instruction_id iid, instruction_queue_event event) override {
		std::vector<std::unique_ptr<instruction>> satisfied;
		{
			std::lock_guard lock(m_mutex);
			const auto [first, last] = m_waiting_on.equal_range(iid);
			for(auto it = first; it != last;) {
				auto& pending = m_pending_instructions.at(it->second);
				assert(pending.num_incomplete_dependencies > 0);
				pending.num_incomplete_dependencies -= 1;
				if(pending.num_incomplete_dependencies == 0) {
					satisfied.push_back(std::move(pending.instr));
					it = m_waiting_on.erase(it);
				} else {
					++it;
				}
			}
		}
		for(auto& instr : satisfied) {
			m_delegate->dependencies_satisfied(std::move(instr));
		}
	}
};

struct instruction_scheduler::impl : instruction_backlog::delegate {
	std::mutex mutex;
	std::unique_ptr<allocation_manager> alloc_manager;
	multiplex_instruction_queue host_queue;
	std::unordered_map<device_id, int> cuda_device_ids;
	std::unordered_map<device_id, multiplex_instruction_queue> cuda_queues;
	std::unordered_map<device_id, sycl_queue> sycl_queues;
	std::unordered_map<instruction_id, std::unique_ptr<instruction>> queued_instructions;
	// TODO GC this on horizons / epochs. It should be tracking iids for finished instructions as well (mind the race around instructions finishing
	// asynchronously)
	std::unordered_map<instruction_id, instruction_queue_event> active_instructions;
	instruction_backlog backlog;

	impl(std::unique_ptr<allocation_manager> alloc_manager, multiplex_instruction_queue host_queue, std::unordered_map<device_id, int> cuda_device_ids,
	    std::unordered_map<device_id, multiplex_instruction_queue> cuda_queues, std::unordered_map<device_id, sycl_queue> sycl_queues)
	    : alloc_manager(std::move(alloc_manager)), host_queue(std::move(host_queue)), cuda_device_ids(std::move(cuda_device_ids)),
	      cuda_queues(std::move(cuda_queues)), sycl_queues(std::move(sycl_queues)), backlog(this) {}

	out_of_order_instruction_queue* select_queue(const instruction& isntr);
	out_of_order_instruction_queue* select_queue(const instruction_backend backend, const std::initializer_list<memory_id>& mids);
	out_of_order_instruction_queue* select_queue(const instruction_backend backend, const device_id did);

	void submit(std::unique_ptr<instruction> instr);
	void dependencies_satisfied_internal(std::unique_ptr<instruction> instr);
	void dependencies_satisfied(std::unique_ptr<instruction> instr) override;
};

out_of_order_instruction_queue* instruction_scheduler::impl::select_queue(const instruction_backend backend, const std::initializer_list<memory_id>& mids) {
	const auto find_in_device_map = [&](auto& device_map) {
		assert(std::all_of(mids.begin(), mids.end(), [&](const memory_id mid) { return mid == host_memory_id || device_map.count(mid - 1) > 0; }));
		for(const auto mid : mids) {
			if(mid == host_memory_id) continue;
			const device_id did = mid - 1;
			if(const auto it = device_map.find(did); it != device_map.end()) { return &it->second; }
		}
		panic("no matching instruction queue");
	};

	switch(backend) {
	case instruction_backend::host:
		assert(std::all_of(mids.begin(), mids.end(), [](const memory_id mid) { return mid == host_memory_id; }));
		return &host_queue;
	case instruction_backend::sycl: return find_in_device_map(sycl_queues);
	case instruction_backend::cuda: return find_in_device_map(cuda_queues);
	}
}

out_of_order_instruction_queue* instruction_scheduler::impl::select_queue(const instruction_backend backend, const device_id did) {
	switch(backend) {
	case instruction_backend::host: panic("cannot select host backend for device");
	case instruction_backend::sycl: return &sycl_queues.at(did);
	case instruction_backend::cuda: return &cuda_queues.at(did);
	}
}

out_of_order_instruction_queue* instruction_scheduler::impl::select_queue(const instruction& instr) {
	return utils::match(
	    instr, [&](const alloc_instruction& ainstr) { return select_queue(ainstr.get_backend(), {ainstr.get_memory_id()}); },
	    [&](const free_instruction& finstr) { return select_queue(finstr.get_backend(), {host_memory_id /* TODO finstr.get_memory_id() */}); },
	    [&](const copy_instruction& cinstr) {
		    return select_queue(cinstr.get_backend(), {cinstr.get_source_memory(), cinstr.get_dest_memory()});
	    },
	    [&](const sycl_kernel_instruction& skinstr) { return select_queue(instruction_backend::sycl, skinstr.get_device_id()); },
	    [&](const auto& /* default */) -> out_of_order_instruction_queue* { return &host_queue; });
}

instruction_scheduler::instruction_scheduler(std::unordered_map<device_id, int> cuda_device_ids, std::unordered_map<device_id, sycl::queue> sycl_queues) {
	constexpr size_t num_host_threads = 4;
	constexpr size_t num_cuda_streams_per_device = 4;

	auto am = std::make_unique<allocation_manager>();

	std::vector<std::unique_ptr<in_order_instruction_queue>> host_instr_queues;
	host_instr_queues.reserve(num_host_threads);
	for(size_t i = 0; i < num_host_threads; ++i) {
		host_instr_queues.emplace_back(std::make_unique<host_thread_queue>());
	}

	std::unordered_map<device_id, multiplex_instruction_queue> cuda_instr_queues;
	for(const auto& [did, cuda_did] : cuda_device_ids) {
		std::vector<std::unique_ptr<in_order_instruction_queue>> cuda_streams;
		cuda_streams.reserve(num_cuda_streams_per_device);
		for(size_t i = 0; i < num_host_threads; ++i) {
			cuda_streams.emplace_back(std::make_unique<cuda_stream>(cuda_did, *am));
		}
		cuda_instr_queues.emplace(did, multiplex_instruction_queue(std::move(cuda_streams)));
	}

	std::unordered_map<device_id, sycl_queue> sycl_instr_queues;
	for(const auto& [did, q] : sycl_queues) {
		sycl_instr_queues.emplace(did, sycl_queue(q, *am));
	}

	m_impl.reset(new impl(std::move(am), multiplex_instruction_queue(std::move(host_instr_queues)), std::move(cuda_device_ids), std::move(cuda_instr_queues),
	    std::move(sycl_instr_queues)));
}

instruction_scheduler::~instruction_scheduler() = default;

void instruction_scheduler::impl::submit(std::unique_ptr<instruction> instr) {
	std::lock_guard lock(mutex);

	std::vector<instruction_queue_event> dependencies;
	size_t num_pending_dependencies = 0;
	for(const auto& dep : instr->get_dependencies()) {
		if(const auto active = active_instructions.find(dep.node->get_id()); active != active_instructions.end()) {
			dependencies.push_back(active->second);
		} else {
			++num_pending_dependencies;
		}
	}

	if(num_pending_dependencies == 0) {
		const auto target_queue = select_queue(*instr);
		target_queue->submit(std::move(instr), std::move(dependencies));
	}
}

void instruction_scheduler::impl::dependencies_satisfied_internal(std::unique_ptr<instruction> instr) {
	std::lock_guard lock(mutex);
	const auto target_queue = select_queue(*instr);
	target_queue->submit(std::move(instr), std::move(backend_dependencies));
}

void instruction_scheduler::impl::dependencies_satisfied(std::unique_ptr<instruction> instr) {
	std::lock_guard lock(mutex);
	dependencies_satisfied_internal(std::move(instr));
}

void instruction_scheduler::submit(std::unique_ptr<instruction> instr) { m_impl->submit(std::move(instr)); }


class abstract_instruction_scheduler {
  public:
	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate& operator=(const delegate&) = default;
		virtual ~delegate() = default;
		virtual instruction_queue_event submit_to_backend(std::unique_ptr<instruction> instr, const std::vector<instruction_queue_event>& dependencies);
	};

	enum class poll_action {
		poll_again,
		idle_until_next_submit,
	};

	explicit abstract_instruction_scheduler(delegate* delegate) : m_delegate(delegate) {}

	void submit(std::unique_ptr<instruction> instr);

	poll_action poll_events();

  private:
	struct pending_instruction_info {
		std::unique_ptr<instruction> instr;
		size_t num_inorder_dependencies;
		std::vector<instruction_id> inorder_submission_dependents;
		std::vector<instruction_id> inorder_completion_dependents;
	};
	struct active_instruction_info {
		instruction_backend backend;
		instruction_queue_event event;
		std::vector<instruction_id> inorder_completion_dependents;
	};
	struct finished_instruction_info {};
	using instruction_info = std::variant<pending_instruction_info, active_instruction_info, finished_instruction_info>;

	delegate* m_delegate;
	std::unordered_map<instruction_id, instruction_info> m_visible_instructions;      // TODO GC on effective epoch
	std::vector<std::pair<instruction_id, instruction_queue_event_impl*>> m_poll_set; // events are owned by m_visible_instructions

	void fulfill_dependency(const std::vector<instruction_id>& dependents, std::vector<instruction_info*>& out_ready_set);

	[[nodiscard]] instruction_queue_event submit_to_backend(std::unique_ptr<instruction> instr);
};

void abstract_instruction_scheduler::submit(std::unique_ptr<instruction> instr) {
	const auto iid = instr->get_id();
	const auto backend = instr->get_backend();

	size_t num_inorder_dependencies = 0;
	for(const auto& dep : instr->get_dependencies()) {
		utils::match(
		    m_visible_instructions.at(dep.node->get_id()),
		    [&](pending_instruction_info& dep_piinfo) {
			    if(dep_piinfo.instr->get_backend() == backend) {
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

abstract_instruction_scheduler::poll_action abstract_instruction_scheduler::poll_events() {
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

void abstract_instruction_scheduler::fulfill_dependency(const std::vector<instruction_id>& dependents, std::vector<instruction_info*>& out_ready_set) {
	for(const auto dependent_iid : dependents) {
		auto& dependent_iinfo = m_visible_instructions.at(dependent_iid);
		auto& dependent_piinfo = std::get<pending_instruction_info>(dependent_iinfo);
		assert(dependent_piinfo.num_inorder_dependencies > 0);
		dependent_piinfo.num_inorder_dependencies -= 1;
		if(dependent_piinfo.num_inorder_dependencies == 0) { out_ready_set.push_back(&dependent_iinfo); }
	}
}

instruction_queue_event abstract_instruction_scheduler::submit_to_backend(std::unique_ptr<instruction> instr) {
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
	auto event = m_delegate->submit_to_backend(std::move(instr), backend_deps);
	m_poll_set.emplace_back(iid, event.get());
	return event;
}

} // namespace celerity::detail
