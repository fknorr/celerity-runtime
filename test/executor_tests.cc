#include "dry_run_executor.h"
#include "executor.h"
#include "live_executor.h"

#include "test_utils.h"

#include <condition_variable>
#include <mutex>

using namespace celerity;
using namespace celerity::detail;

struct mock_host_object final : host_object_instance {
	std::atomic<bool>* destroyed;

	explicit mock_host_object(std::atomic<bool>* const destroyed) : destroyed(destroyed) {}
	mock_host_object(const mock_host_object&) = delete;
	mock_host_object(mock_host_object&&) = delete;
	mock_host_object& operator=(const mock_host_object&) = delete;
	mock_host_object& operator=(mock_host_object&&) = delete;
	~mock_host_object() override { *destroyed = true; }
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

	void reduce(void* dest, const void* src, size_t src_count) const override { (void)dest, (void)src, (void)src_count; }
	void fill_identity(void* dest, size_t count) const override { (void)dest, (void)count; }
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
	struct host_alloc {
		size_t size = 0;
		size_t alignment = 0;
		void* result = nullptr;

		friend bool operator==(const host_alloc& lhs, const host_alloc& rhs) { return lhs.size == rhs.size && lhs.alignment == rhs.alignment; }
		friend bool operator!=(const host_alloc& lhs, const host_alloc& rhs) { return !(lhs == rhs); }
	};

	struct device_alloc {
		device_id device = 0;
		size_t size = 0;
		size_t alignment = 0;
		void* result = nullptr;

		friend bool operator==(const device_alloc& lhs, const device_alloc& rhs) {
			return lhs.device == rhs.device && lhs.size == rhs.size && lhs.alignment == rhs.alignment;
		}
		friend bool operator!=(const device_alloc& lhs, const device_alloc& rhs) { return !(lhs == rhs); }
	};

	struct host_free {
		void* ptr = nullptr;

		friend bool operator==(const host_free& lhs, const host_free& rhs) { return lhs.ptr == rhs.ptr; }
		friend bool operator!=(const host_free& lhs, const host_free& rhs) { return !(lhs == rhs); }
	};

	struct device_free {
		device_id device = 0;
		void* ptr = nullptr;

		friend bool operator==(const device_free& lhs, const device_free& rhs) { return lhs.device == rhs.device && lhs.ptr == rhs.ptr; }
		friend bool operator!=(const device_free& lhs, const device_free& rhs) { return !(lhs == rhs); }
	};

	struct host_task {
		size_t host_lane;
		std::vector<closure_hydrator::accessor_info> accessor_infos;
		box<3> execution_range;
		const communicator* collective_comm = nullptr;

		friend bool operator==(const host_task& lhs, const host_task& rhs) {
			return lhs.host_lane == rhs.host_lane && lhs.accessor_infos == rhs.accessor_infos && lhs.execution_range == rhs.execution_range
			       && lhs.collective_comm == rhs.collective_comm;
		}
		friend bool operator!=(const host_task& lhs, const host_task& rhs) { return !(lhs == rhs); }
	};

	struct device_kernel {
		device_id device = 0;
		size_t device_lane = 0;
		std::vector<closure_hydrator::accessor_info> accessor_infos;
		box<3> execution_range;
		std::vector<void*> reduction_ptrs;

		friend bool operator==(const device_kernel& lhs, const device_kernel& rhs) {
			return lhs.device == rhs.device && lhs.device_lane == rhs.device_lane && lhs.accessor_infos == rhs.accessor_infos
			       && lhs.execution_range == rhs.execution_range && lhs.reduction_ptrs == rhs.reduction_ptrs;
		}
		friend bool operator!=(const device_kernel& lhs, const device_kernel& rhs) { return !(lhs == rhs); }
	};

	struct host_copy {
		size_t host_lane = 0;
		const void* source_base = nullptr;
		void* dest_base = nullptr;
		box<3> source_box;
		box<3> dest_box;
		region<3> copy_region;
		size_t elem_size = 0;

		friend bool operator==(const host_copy& lhs, const host_copy& rhs) {
			return lhs.host_lane == rhs.host_lane && lhs.source_base == rhs.source_base && lhs.dest_base == rhs.dest_base && lhs.source_box == rhs.source_box
			       && lhs.dest_box == rhs.dest_box && lhs.copy_region == rhs.copy_region && lhs.elem_size == rhs.elem_size;
		}
		friend bool operator!=(const host_copy& lhs, const host_copy& rhs) { return !(lhs == rhs); }
	};

	struct device_copy {
		device_id device = 0;
		size_t device_lane = 0;
		const void* source_base = nullptr;
		void* dest_base = nullptr;
		box<3> source_box;
		box<3> dest_box;
		region<3> copy_region;
		size_t elem_size = 0;

		friend bool operator==(const device_copy& lhs, const device_copy& rhs) {
			return lhs.device == rhs.device && lhs.device_lane == rhs.device_lane && lhs.source_base == rhs.source_base && lhs.dest_base == rhs.dest_base
			       && lhs.source_box == rhs.source_box && lhs.dest_box == rhs.dest_box && lhs.copy_region == rhs.copy_region && lhs.elem_size == rhs.elem_size;
		}
		friend bool operator!=(const device_copy& lhs, const device_copy& rhs) { return !(lhs == rhs); }
	};

	using operation = std::variant<host_alloc, device_alloc, host_free, device_free, host_task, device_kernel, host_copy, device_copy>;

	mock_backend(const system_info& system, std::vector<operation>* const log) : m_system(system), m_log(log) {}

	const system_info& get_system_info() const override { return m_system; }

	void* debug_alloc(size_t size) override { return malloc(size); }
	void debug_free(void* ptr) override { free(ptr); }

	async_event enqueue_host_alloc(const size_t size, const size_t alignment) override {
		const auto ptr = mock_alloc(size, alignment);
		m_log->push_back(host_alloc{size, alignment, ptr});
		return make_complete_event(ptr);
	}

	async_event enqueue_device_alloc(const device_id device, const size_t size, const size_t alignment) override {
		const auto ptr = mock_alloc(size, alignment);
		m_log->push_back(device_alloc{device, size, alignment, ptr});
		return make_complete_event(ptr);
	}

	async_event enqueue_host_free(void* const ptr) override {
		m_log->push_back(host_free{ptr});
		return make_complete_event();
	}

	async_event enqueue_device_free(const device_id device, void* const ptr) override {
		m_log->push_back(device_free{device, ptr});
		return make_complete_event();
	}

	async_event enqueue_host_task(const size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
	    const box<3>& execution_range, const communicator* const collective_comm) override //
	{
		m_log->push_back(host_task{host_lane, std::move(accessor_infos), execution_range, collective_comm});
		return make_complete_event();
	}

	async_event enqueue_device_kernel(const device_id device, const size_t device_lane, const device_kernel_launcher& launcher,
	    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override //
	{
		m_log->push_back(device_kernel{device, device_lane, std::move(accessor_infos), execution_range, reduction_ptrs});
		return make_complete_event();
	}

	async_event enqueue_host_copy(const size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override //
	{
		m_log->push_back(host_copy{host_lane, source_base, dest_base, source_box, dest_box, copy_region, elem_size});
		return make_complete_event();
	}

	async_event enqueue_device_copy(const device_id device, const size_t device_lane, const void* const source_base, void* const dest_base,
	    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override //
	{
		m_log->push_back(device_copy{device, device_lane, source_base, dest_base, source_box, dest_box, copy_region, elem_size});
		return make_complete_event();
	}

  private:
	system_info m_system;
	uintptr_t m_last_mock_alloc_address = 0;
	std::vector<operation>* m_log;

	void* mock_alloc(const size_t size, const size_t alignment) {
		CHECK(size > 0);
		CHECK(alignment > 0);
		CHECK(size >= alignment);
		CHECK(size % alignment == 0);
		const auto address = (m_last_mock_alloc_address + alignment) / alignment * alignment;
		m_last_mock_alloc_address = address + size;
		return reinterpret_cast<void*>(address);
	}
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
			const auto system = test_utils::make_system_info(4 /* num devices */, false /* d2d copies*/);
			m_executor = std::make_unique<live_executor>(
			    std::make_unique<mock_backend>(system, &m_backend_log), std::make_unique<mock_local_communicator>(), static_cast<executor::delegate*>(this));
		}
	}

	executor_test_context(const executor_test_context&) = delete;
	executor_test_context(executor_test_context&&) = delete;
	executor_test_context& operator=(const executor_test_context&) = delete;
	executor_test_context& operator=(executor_test_context&&) = delete;

	~executor_test_context() { REQUIRE(m_has_shut_down); }

	std::tuple<task_id, instruction_id> init() {
		const auto iid = submit<epoch_instruction>({}, task_manager::initial_epoch_task, epoch_action::none, instruction_garbage{});
		return {task_manager::initial_epoch_task, iid};
	}

	std::tuple<task_id, instruction_id> horizon(const std::vector<instruction_id>& predecessors, instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		const auto iid = submit<horizon_instruction>({}, tid, std::move(garbage));
		return {tid, iid};
	}

	std::tuple<task_id, instruction_id> epoch(const std::vector<instruction_id>& predecessors, const epoch_action action, instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		const auto iid = submit<epoch_instruction>({}, tid, action, std::move(garbage));
		return {tid, iid};
	}

	instruction_id alloc(const std::vector<instruction_id>& predecessors, const allocation_id aid, const size_t size, const size_t alignment) {
		return submit<alloc_instruction>(predecessors, aid, size, alignment);
	}

	instruction_id free(const std::vector<instruction_id>& predecessors, const allocation_id aid) { return submit<free_instruction>(predecessors, aid); }

	instruction_id host_task(const std::vector<instruction_id>& predecessors, const box<3>& execution_range, const range<3>& global_range,
	    buffer_access_allocation_map amap, const collective_group_id cgid) //
	{
		return submit<host_task_instruction>(predecessors, host_task_launcher{}, execution_range, global_range, std::move(amap),
		    cgid CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::host_compute, task_id(1), "task_name"));
	}

	instruction_id device_kernel(const std::vector<instruction_id>& predecessors, const device_id did, const box<3>& execution_range,
	    buffer_access_allocation_map amap, buffer_access_allocation_map rmap) //
	{
		return submit<device_kernel_instruction>(predecessors, did, device_kernel_launcher{}, execution_range, std::move(amap),
		    std::move(rmap) CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::device_compute, task_id(1), "task_name"));
	}

	instruction_id destroy_host_object(const std::vector<instruction_id>& predecessors, const host_object_id hoid) {
		return submit<destroy_host_object_instruction>(predecessors, hoid);
	}

	instruction_id fence_and_wait(const std::vector<instruction_id>& predecessors) {
		mock_fence_promise fence_promise;
		const auto iid = submit<fence_instruction>(predecessors, &fence_promise);
		fence_promise.wait();
		return iid;
	}

	std::vector<mock_backend::operation> shutdown(const std::vector<instruction_id>& predecessors) {
		CHECK(!m_has_shut_down);
		epoch(predecessors, epoch_action::shutdown, instruction_garbage{});
		m_executor->wait();
		m_has_shut_down = true;
		return std::move(m_backend_log);
	}

	executor& get_executor() { return *m_executor; }

	task_id get_last_epoch() { return m_epochs.get(); }
	task_id get_last_horizon() { return m_horizons.get(); }

	void await_horizon(const task_id tid) { m_horizons.await(tid); }
	void await_epoch(const task_id tid) { m_epochs.await(tid); }

  private:
	instruction_id m_next_iid = 1;
	task_id m_next_task_id = task_manager::initial_epoch_task + 1;
	std::vector<std::unique_ptr<instruction>> m_instructions; // we need to guarantee liveness as long as the executor thread is around
	bool m_has_shut_down = false;
	std::unique_ptr<executor> m_executor;
	epoch_monitor m_horizons{0};
	epoch_monitor m_epochs{0};
	std::vector<mock_backend::operation> m_backend_log; // mutated by executor thread - do not access before shutdown!

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

	void horizon_reached(const task_id tid) override { m_horizons.set(tid); }
	void epoch_reached(const task_id tid) override { m_epochs.set(tid); }
};


TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "executors notify delegate when encountering a horizon / epoch", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	const auto [init_epoch_tid, init_epoch] = ectx.init();

	CHECK(ectx.get_last_epoch() == task_id(0));
	CHECK(ectx.get_last_horizon() == task_id(0));

	instruction_id node = 0;

	SECTION("on epochs") {
		const auto [epoch_tid, epoch] = ectx.epoch({init_epoch}, epoch_action::none);
		ectx.await_epoch(epoch_tid);
		CHECK(ectx.get_last_epoch() == epoch_tid);
		node = epoch;
	}

	SECTION("on horizons") {
		const auto [horizon_tid, horizon] = ectx.horizon({init_epoch});
		ectx.await_horizon(horizon_tid);
		CHECK(ectx.get_last_horizon() == horizon_tid);
		node = horizon;
	}

	ectx.shutdown({node});
}

TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "dry_run_executor warns when encountering a fence instruction ", "[executor][dry_run]") {
	executor_test_context ectx(executor_type::dry_run);
	const auto [_, init_epoch] = ectx.init();
	const auto fence = ectx.fence_and_wait({init_epoch});
	CHECK(test_utils::log_contains_exact(log_level::warn, "Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. The result of this "
	                                                      "operation will not match the expected output of an actual run."));
	ectx.shutdown({fence});
}

TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "executors free all reducers that appear in garbage lists  ", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	const auto [_1, init_epoch] = ectx.init();

	const reduction_id rid(123);
	std::atomic<bool> destroyed{false};
	ectx.get_executor().announce_reducer(rid, std::make_unique<mock_reducer>(&destroyed));

	instruction_id collector = 0;

	SECTION("on epochs") {
		const auto [epoch_tid, epoch] = ectx.epoch({init_epoch}, epoch_action::none, instruction_garbage{{rid}, {}});
		ectx.await_epoch(epoch_tid);
		collector = epoch;
	}

	SECTION("on horizons") {
		const auto [horizon_tid, horizon] = ectx.horizon({init_epoch}, instruction_garbage{{rid}, {}});
		ectx.await_horizon(horizon_tid);
		collector = horizon;
	}

	ectx.shutdown({collector});
}

TEST_CASE("host objects lifetime is controlled by destroy_host_object_instruction", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	const auto [_1, init_epoch] = ectx.init();

	const host_object_id hoid(42);
	std::atomic<bool> destroyed{false};
	ectx.get_executor().announce_host_object_instance(hoid, std::make_unique<mock_host_object>(&destroyed));
	CHECK_FALSE(destroyed.load());

	const auto [after_announce_tid, after_announce] = ectx.horizon({init_epoch});
	CHECK_FALSE(destroyed.load());

	ectx.await_horizon(after_announce_tid);
	CHECK_FALSE(destroyed.load());

	const auto destroy = ectx.destroy_host_object({after_announce}, hoid);
	const auto [after_destroy_tid, after_destroy] = ectx.horizon({destroy});
	ectx.await_horizon(after_destroy_tid);
	CHECK(destroyed);

	ectx.shutdown({after_destroy});
}

TEST_CASE("live_executor passes correct allocations to host tasks", "[executor]") {
	executor_test_context ectx(executor_type::live);
	const auto [_1, init_epoch] = ectx.init();

	const auto alloc1 = ectx.alloc({init_epoch}, allocation_id(host_memory_id, 1), 1024, 8);
	const auto alloc2 = ectx.alloc({init_epoch, alloc1}, allocation_id(host_memory_id, 2), 2048, 16);

	const auto ht_amap = buffer_access_allocation_map{
	    {
	        allocation_id(host_memory_id, 1),
	        box<3>({0, 0, 0}, {32, 32, 1}),
	        box<3>({8, 8, 0}, {24, 24, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(1, "buffer1"),
	    },
	    {
	        allocation_id(host_memory_id, 2),
	        box<3>({0, 0, 0}, {64, 16, 1}),
	        box<3>({0, 0, 0}, {16, 16, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(2, "buffer2"),
	    },
	};
	const auto ht = ectx.host_task({alloc1, alloc2}, box<3>{{0, 1, 2}, {3, 4, 5}}, range<3>{7, 8, 9}, ht_amap, non_collective_group_id);

	const auto free1 = ectx.free({ht}, allocation_id(host_memory_id, 1));
	const auto free2 = ectx.free({ht, free1}, allocation_id(host_memory_id, 2));

	const auto log = ectx.shutdown({free1, free2});
	REQUIRE(log.size() == 5);

	const auto log_alloc1 = std::get<mock_backend::host_alloc>(log[0]);
	CHECK(log_alloc1.size == 1024);
	CHECK(log_alloc1.alignment == 8);
	CHECK(log_alloc1.result != nullptr);

	const auto log_alloc2 = std::get<mock_backend::host_alloc>(log[1]);
	CHECK(log_alloc2.size == 2048);
	CHECK(log_alloc2.alignment == 16);
	CHECK(log_alloc2.result != nullptr);

	const auto log_ht = std::get<mock_backend::host_task>(log[2]);
	CHECK(log_ht.collective_comm == nullptr);
	CHECK(log_ht.execution_range == box<3>{{0, 1, 2}, {3, 4, 5}});

	REQUIRE(log_ht.accessor_infos.size() == 2);
	CHECK(log_ht.accessor_infos[0].ptr == log_alloc1.result);
	CHECK(log_ht.accessor_infos[0].allocated_box_in_buffer == ht_amap[0].allocated_box_in_buffer);
	CHECK(log_ht.accessor_infos[0].accessed_box_in_buffer == ht_amap[0].accessed_box_in_buffer);
	CHECK(log_ht.accessor_infos[1].ptr == log_alloc2.result);
	CHECK(log_ht.accessor_infos[1].allocated_box_in_buffer == ht_amap[1].allocated_box_in_buffer);
	CHECK(log_ht.accessor_infos[1].accessed_box_in_buffer == ht_amap[1].accessed_box_in_buffer);

	const auto log_free1 = std::get<mock_backend::host_free>(log[3]);
	CHECK(log_free1.ptr == log_alloc1.result);

	const auto log_free2 = std::get<mock_backend::host_free>(log[4]);
	CHECK(log_free2.ptr == log_alloc2.result);
}

TEST_CASE("live_executor passes correct allocations to device kernels", "[executor]") {
	executor_test_context ectx(executor_type::live);
	const auto [_1, init_epoch] = ectx.init();

	const auto did = GENERATE(values<device_id>({0, 1}));
	const auto mid = memory_id(first_device_memory_id + did);

	const auto alloc1 = ectx.alloc({init_epoch}, allocation_id(mid, 1), 1024, 8);
	const auto alloc2 = ectx.alloc({init_epoch, alloc1}, allocation_id(mid, 2), 2048, 16);
	const auto alloc3 = ectx.alloc({init_epoch, alloc2}, allocation_id(mid, 3), 4, 4);
	const auto alloc4 = ectx.alloc({init_epoch, alloc3}, allocation_id(mid, 4), 4, 4);

	const auto amap = buffer_access_allocation_map{
	    {
	        allocation_id(mid, 1),
	        box<3>({0, 0, 0}, {32, 32, 1}),
	        box<3>({8, 8, 0}, {24, 24, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(1, "buffer1"),
	    },
	    {
	        allocation_id(mid, 2),
	        box<3>({0, 0, 0}, {64, 16, 1}),
	        box<3>({0, 0, 0}, {16, 16, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(2, "buffer2"),
	    },
	};
	const auto rmap = buffer_access_allocation_map{
	    {
	        allocation_id(mid, 3),
	        box<3>({0, 0, 0}, {1, 1, 1}),
	        box<3>({0, 0, 0}, {1, 1, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(3, "buffer3"),
	    },
	    {
	        allocation_id(mid, 4),
	        box<3>({0, 0, 0}, {1, 1, 1}),
	        box<3>({0, 0, 0}, {1, 1, 1}),
	        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(4, "buffer4"),
	    },
	};
	const auto kernel = ectx.device_kernel({alloc1, alloc2, alloc3, alloc4}, did, box<3>({1, 2, 3}, {4, 5, 6}), amap, rmap);

	const auto free1 = ectx.free({kernel}, allocation_id(mid, 1));
	const auto free2 = ectx.free({kernel, free1}, allocation_id(mid, 2));
	const auto free3 = ectx.free({kernel, free2}, allocation_id(mid, 3));
	const auto free4 = ectx.free({kernel, free3}, allocation_id(mid, 4));

	const auto log = ectx.shutdown({free1, free2, free3, free4});
	REQUIRE(log.size() == 9);

	const auto log_alloc1 = std::get<mock_backend::device_alloc>(log[0]);
	CHECK(log_alloc1.device == did);
	CHECK(log_alloc1.size == 1024);
	CHECK(log_alloc1.alignment == 8);
	CHECK(log_alloc1.result != nullptr);

	const auto log_alloc2 = std::get<mock_backend::device_alloc>(log[1]);
	CHECK(log_alloc2.device == did);
	CHECK(log_alloc2.size == 2048);
	CHECK(log_alloc2.alignment == 16);
	CHECK(log_alloc2.result != nullptr);

	const auto log_alloc3 = std::get<mock_backend::device_alloc>(log[2]);
	CHECK(log_alloc3.device == did);
	CHECK(log_alloc3.size == 4);
	CHECK(log_alloc3.alignment == 4);
	CHECK(log_alloc3.result != nullptr);

	const auto log_alloc4 = std::get<mock_backend::device_alloc>(log[3]);
	CHECK(log_alloc4.device == did);
	CHECK(log_alloc4.size == 4);
	CHECK(log_alloc4.alignment == 4);
	CHECK(log_alloc4.result != nullptr);

	const auto log_kernel = std::get<mock_backend::device_kernel>(log[4]);
	CHECK(log_kernel.device == did);
	CHECK(log_kernel.execution_range == box<3>({1, 2, 3}, {4, 5, 6}));

	REQUIRE(log_kernel.accessor_infos.size() == 2);
	CHECK(log_kernel.accessor_infos[0].ptr == log_alloc1.result);
	CHECK(log_kernel.accessor_infos[0].allocated_box_in_buffer == amap[0].allocated_box_in_buffer);
	CHECK(log_kernel.accessor_infos[0].accessed_box_in_buffer == amap[0].accessed_box_in_buffer);
	CHECK(log_kernel.accessor_infos[1].ptr == log_alloc2.result);
	CHECK(log_kernel.accessor_infos[1].allocated_box_in_buffer == amap[1].allocated_box_in_buffer);
	CHECK(log_kernel.accessor_infos[1].accessed_box_in_buffer == amap[1].accessed_box_in_buffer);

	CHECK(log_kernel.reduction_ptrs == std::vector{log_alloc3.result, log_alloc4.result});

	const auto log_free1 = std::get<mock_backend::device_free>(log[5]);
	CHECK(log_free1.device == did);
	CHECK(log_free1.ptr == log_alloc1.result);

	const auto log_free2 = std::get<mock_backend::device_free>(log[6]);
	CHECK(log_free2.device == did);
	CHECK(log_free2.ptr == log_alloc2.result);

	const auto log_free3 = std::get<mock_backend::device_free>(log[7]);
	CHECK(log_free3.device == did);
	CHECK(log_free3.ptr == log_alloc3.result);

	const auto log_free4 = std::get<mock_backend::device_free>(log[8]);
	CHECK(log_free4.device == did);
	CHECK(log_free4.ptr == log_alloc4.result);
}
