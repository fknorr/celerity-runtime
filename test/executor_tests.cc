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

class mock_executor_delegate : public executor::delegate {
  public:
	std::atomic<task_id> last_horizon_reached{0};
	std::atomic<task_id> last_epoch_reached{0};

	mock_executor_delegate() = default;
	mock_executor_delegate(const mock_executor_delegate&) = delete;
	mock_executor_delegate(mock_executor_delegate&&) = delete;
	mock_executor_delegate& operator=(const mock_executor_delegate&) = delete;
	mock_executor_delegate& operator=(mock_executor_delegate&&) = delete;
	virtual ~mock_executor_delegate() = default;

	void horizon_reached(task_id tid) override { last_horizon_reached = tid; }
	void epoch_reached(task_id tid) override { last_epoch_reached = tid; }
};

auto generate_executors() {
	const auto executor_type = GENERATE(values<std::string>({"dry_run", "live"}));
	CAPTURE(executor_type);

	auto delegate = std::make_unique<mock_executor_delegate>();

	std::unique_ptr<executor> executor;
	if(executor_type == "dry_run") {
		executor = std::make_unique<dry_run_executor>(delegate.get());
	} else {
		const auto system = test_utils::make_system_info(1 /* num devices */, false /* d2d copies*/);
		executor = std::make_unique<live_executor>(std::make_unique<mock_backend>(system), std::make_unique<mock_local_communicator>(), delegate.get());
	}

	return std::tuple{executor_type, std::move(executor), std::move(delegate)};
}

TEST_CASE("executors free all reducers that appear in horizon / epoch garbage lists  ", "[executor]") {
	const auto [executor_type, executor, delegate] = generate_executors();
	CAPTURE(executor_type);

	const auto instruction_type = GENERATE(values<std::string>({"epoch", "horizon"}));
	CAPTURE(instruction_type);

	const auto init_epoch = std::make_unique<epoch_instruction>(instruction_id(1), 0, task_id(1), epoch_action::none, instruction_garbage{});
	executor->submit({init_epoch.get()}, {});

	const reduction_id rid(123);
	std::atomic<bool> destroyed{false};
	executor->announce_reducer(rid, std::make_unique<mock_reducer>(&destroyed));

	std::unique_ptr<instruction> collecting_instruction;
	if(instruction_type == "epoch") {
		collecting_instruction = std::make_unique<epoch_instruction>(instruction_id(2), 0, task_id(2), epoch_action::none, instruction_garbage{{rid}, {}});
	} else {
		collecting_instruction = std::make_unique<horizon_instruction>(instruction_id(2), 0, task_id(2), instruction_garbage{{rid}, {}});
	}
	collecting_instruction->add_dependency(init_epoch->get_id());

	mock_fence_promise fence_promise;
	const auto fence = std::make_unique<fence_instruction>(instruction_id(3), 0, &fence_promise);
	fence->add_dependency(collecting_instruction->get_id());

	executor->submit({collecting_instruction.get(), fence.get()}, {});

	fence_promise.wait();
	CHECK(destroyed);

	const auto shutdown_epoch = std::make_unique<epoch_instruction>(instruction_id(4), 0, task_id(4), epoch_action::shutdown, instruction_garbage{});
	executor->submit({shutdown_epoch.get()}, {});
    executor->wait();
}
