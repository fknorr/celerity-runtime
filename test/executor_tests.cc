#include "dry_run_executor.h"
#include "executor.h"
#include "live_executor.h"

#include "test_utils.h"

#include <condition_variable>
#include <mutex>

using namespace celerity;
using namespace celerity::detail;

struct mock_host_object_instance final : host_object_instance {
	std::atomic<bool>* destroyed;

	explicit mock_host_object_instance(std::atomic<bool>* const destroyed) : destroyed(destroyed) {}
	mock_host_object_instance(const mock_host_object_instance&) = delete;
	mock_host_object_instance(mock_host_object_instance&&) = delete;
	mock_host_object_instance& operator=(const mock_host_object_instance&) = delete;
	mock_host_object_instance& operator=(mock_host_object_instance&&) = delete;
	~mock_host_object_instance() override { *destroyed = true; }
};

class mock_reducer final : public reducer {
  public:
	std::atomic<bool>* destroyed;

	explicit mock_reducer(std::atomic<bool>* const destroyed) : destroyed(destroyed) {}
	mock_reducer(const mock_reducer&) = delete;
	mock_reducer(mock_reducer&&) = delete;
	mock_reducer& operator=(const mock_reducer&) = delete;
	mock_reducer& operator=(mock_reducer&&) = delete;
	~mock_reducer() override { *destroyed = true; }

	void reduce(void* dest, const void* src, size_t src_count) const override { utils::panic("not implemented"); }
	void fill_identity(void* dest, size_t count) const override { utils::panic("not implemented"); }
};

class mock_local_communicator final : public communicator {
  public:
	size_t get_num_nodes() const override { return 1; }
	node_id get_local_node_id() const override { return 0; }

	std::vector<inbound_pilot> poll_inbound_pilots() override { return {}; }

	void send_outbound_pilot(const outbound_pilot& pilot) override { utils::panic("not implemented"); }
	async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override { utils::panic("not implemented"); }
	async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override { utils::panic("not implemented"); }

	std::unique_ptr<communicator> collective_clone() override { return std::make_unique<mock_local_communicator>(); }
	void collective_barrier() override {}
};

class mock_backend final : public backend {
  public:
	mock_backend(const system_info& system) : m_system(system) {}

	const system_info& get_system_info() const override { return m_system; }

	void* debug_alloc(size_t size) override { return malloc(size); }
	void debug_free(void* ptr) override { free(ptr); }

	async_event enqueue_host_alloc(size_t size, size_t alignment) override { utils::panic("not implemented"); }
	async_event enqueue_device_alloc(device_id device, size_t size, size_t alignment) override { utils::panic("not implemented"); }
	async_event enqueue_host_free(void* ptr) override { utils::panic("not implemented"); }
	async_event enqueue_device_free(device_id device, void* ptr) override { utils::panic("not implemented"); }

	async_event enqueue_host_task(size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
	    const box<3>& execution_range, const communicator* collective_comm) override {
		utils::panic("not implemented");
	}

	async_event enqueue_device_kernel(device_id device, size_t device_lane, const device_kernel_launcher& launcher,
	    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override {
		utils::panic("not implemented");
	}

	async_event enqueue_host_copy(size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
	    const region<3>& copy_region, const size_t elem_size) override {
		utils::panic("not implemented");
	}

	async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override {
		utils::panic("not implemented");
	}

  private:
	system_info m_system;
};

class mock_fence_promise : public fence_promise {
  public:
	void fulfill() override {
		std::lock_guard lock(m_mutex);
		m_fulfilled = true;
		m_cv.notify_all();
	}

	allocation_id get_user_allocation_id() override { utils::panic("not implemented"); }

	void wait() {
		std::unique_lock lock(m_mutex);
		m_cv.wait(lock, [this] { return m_fulfilled; });
	}

  private:
	std::mutex m_mutex;
	bool m_fulfilled = false;
	std::condition_variable m_cv;
};

enum class executor_type { dry_run, live };

template <>
struct Catch::StringMaker<executor_type> {
	static std::string convert(executor_type value) {
		switch(value) {
		case executor_type::dry_run: return "dry_run";
		case executor_type::live: return "live";
		}
	}
};

class executor_test_context final : private executor::delegate {
  public:
	explicit executor_test_context(const executor_type type) {
		if(type == executor_type::dry_run) {
			m_executor = std::make_unique<dry_run_executor>(static_cast<executor::delegate*>(this));
		} else {
			const auto system = test_utils::make_system_info(1 /* num devices */, false /* d2d copies*/);
			m_executor = std::make_unique<live_executor>(
			    std::make_unique<mock_backend>(system), std::make_unique<mock_local_communicator>(), static_cast<executor::delegate*>(this));
		}
	}

	executor_test_context(const executor_test_context&) = delete;
	executor_test_context(executor_test_context&&) = delete;
	executor_test_context& operator=(const executor_test_context&) = delete;
	executor_test_context& operator=(executor_test_context&&) = delete;

	~executor_test_context() { REQUIRE(m_has_shut_down); }

	std::tuple<task_id, instruction_id> init() { return epoch({}, epoch_action::none, instruction_garbage{}); }

	std::tuple<task_id, instruction_id> horizon(const std::vector<instruction_id>& predecessors, instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		const auto iid = submit<horizon_instruction>({}, m_next_task_id++, std::move(garbage));
		return {tid, iid};
	}

	std::tuple<task_id, instruction_id> epoch(const std::vector<instruction_id>& predecessors, const epoch_action action, instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		const auto iid = submit<epoch_instruction>({}, m_next_task_id++, action, std::move(garbage));
		return {tid, iid};
	}

	instruction_id fence_and_wait(const std::vector<instruction_id>& predecessors) {
		mock_fence_promise fence_promise;
		const auto iid = submit<fence_instruction>(predecessors, &fence_promise);
		fence_promise.wait();
		return iid;
	}

	void shutdown(const std::vector<instruction_id>& predecessors) {
		CHECK(!m_has_shut_down);
		submit<epoch_instruction>({}, m_next_task_id++, epoch_action::shutdown, instruction_garbage{});
		m_executor->wait();
		m_has_shut_down = true;
	}

	executor& get_executor() { return *m_executor; }

  private:
	instruction_id m_next_iid = 1;
	task_id m_next_task_id = 1;
	std::vector<std::unique_ptr<instruction>> m_instructions; // we need to guarantee liveness as long as the executor thread is around
	bool m_has_shut_down = false;
	std::unique_ptr<executor> m_executor;
	std::atomic<task_id> m_last_horizon_reached{0};
	std::atomic<task_id> m_last_epoch_reached{0};

	template <typename Instruction, typename... CtorParams>
	instruction_id submit(const std::vector<instruction_id>& predecessors, CtorParams&&... ctor_args) {
		const auto iid = instruction_id(m_next_iid++);
		auto instr = std::make_unique<Instruction>(iid, 0, std::forward<CtorParams>(ctor_args)...);
		for(const auto pred : predecessors) {
			instr->add_dependency(pred);
		}
		m_executor->submit({instr.get()}, {});
		m_instructions.push_back(std::move(instr));
		return iid;
	}

	void horizon_reached(const task_id tid) override { m_last_horizon_reached.store(tid); }
	void epoch_reached(const task_id tid) override { m_last_epoch_reached.store(tid); }
};

TEST_CASE_METHOD(test_utils::executor_fixture, "dry_run_executor warns when encountering a fence instruction ", "[executor][dry_run]") {
	executor_test_context ectx(executor_type::dry_run);
	const auto [_, init_epoch] = ectx.init();
	const auto fence = ectx.fence_and_wait({init_epoch});
	CHECK(test_utils::log_contains_exact(log_level::warn, "Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. The result of this "
	                                                      "operation will not match the expected output of an actual run."));
	ectx.shutdown({fence});
}

TEST_CASE_METHOD(test_utils::executor_fixture, "executors free all reducers that appear in horizon / epoch garbage lists  ", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	const auto [_1, init_epoch] = ectx.init();

	const reduction_id rid(123);
	std::atomic<bool> destroyed{false};
	ectx.get_executor().announce_reducer(rid, std::make_unique<mock_reducer>(&destroyed));

	const auto instruction_type = GENERATE(values<std::string>({"epoch", "horizon"}));
	CAPTURE(instruction_type);

	const auto [_2, collector] = instruction_type == "epoch" ? ectx.epoch({init_epoch}, epoch_action::none, instruction_garbage{{rid}, {}})
	                                                         : ectx.horizon({init_epoch}, instruction_garbage{{rid}, {}});
	const auto fence = ectx.fence_and_wait({collector});
	CHECK(destroyed);

	ectx.shutdown({fence});
}
