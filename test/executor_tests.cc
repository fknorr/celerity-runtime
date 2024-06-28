#include "dry_run_executor.h"
#include "executor.h"
#include "live_executor.h"

#include "test_utils.h"

#include <condition_variable>
#include <mutex>

using namespace celerity;
using namespace celerity::detail;


// The mock implementations in this test append all operations to a sequential operations_log, which is inspected after executor shutdown.
namespace ops {

struct common_alloc {
	size_t size = 0;
	size_t alignment = 0;
	void* result = nullptr;
};

struct host_alloc : common_alloc {};

struct device_alloc : common_alloc {
	device_id device = 0;
};

struct common_free {
	void* ptr = nullptr;
};

struct host_free : common_free {};

struct device_free : common_free {
	device_id device = 0;
};

struct host_task {
	size_t host_lane;
	std::vector<closure_hydrator::accessor_info> accessor_infos;
	box<3> execution_range;
	const communicator* collective_comm = nullptr;
};

struct device_kernel {
	device_id device = 0;
	size_t device_lane = 0;
	std::vector<closure_hydrator::accessor_info> accessor_infos;
	box<3> execution_range;
	std::vector<void*> reduction_ptrs;
};

struct common_copy {
	const void* source_base = nullptr;
	void* dest_base = nullptr;
	box<3> source_box;
	box<3> dest_box;
	region<3> copy_region;
	size_t elem_size = 0;
};

struct host_copy : common_copy {
	size_t host_lane = 0;
};

struct device_copy : common_copy {
	device_id device = 0;
	size_t device_lane = 0;
};

struct reduce {
	void* dest = nullptr;
	const void* src = nullptr;
	size_t src_count = 0;
};

struct fill_identity {
	void* dest = nullptr;
	size_t count = 0;
};

struct send_outbound_pilot {
	outbound_pilot pilot;
};

struct send_payload {
	node_id to = 0;
	message_id msgid = 0;
	const void* base = nullptr;
	communicator::stride stride;
};

struct receive_payload {
	node_id from = 0;
	message_id msgid = 0;
	void* base = nullptr;
	communicator::stride stride;
};

struct collective_clone {
	int parent_comm_index = 0;
	int child_comm_index = 0;
};

struct collective_barrier {
	int comm_index = 0;
};

} // namespace ops

using operation =
    std::variant<ops::host_alloc, ops::device_alloc, ops::host_free, ops::device_free, ops::host_task, ops::device_kernel, ops::host_copy, ops::device_copy,
        ops::reduce, ops::fill_identity, ops::send_outbound_pilot, ops::send_payload, ops::receive_payload, ops::collective_clone, ops::collective_barrier>;
using operations_log = std::vector<operation>;


struct mock_host_object final : host_object_instance {
	std::atomic<bool>* destroyed = nullptr;

	explicit mock_host_object(std::atomic<bool>* const destroyed) : destroyed(destroyed) {}

	mock_host_object(const mock_host_object&) = delete;
	mock_host_object(mock_host_object&&) = delete;
	mock_host_object& operator=(const mock_host_object&) = delete;
	mock_host_object& operator=(mock_host_object&&) = delete;

	~mock_host_object() override {
		if(destroyed != nullptr) { destroyed->store(true); }
	}
};

class mock_reducer final : public reducer {
  public:
	explicit mock_reducer(std::atomic<bool>* const destroyed, operations_log* const log) : m_destroyed(destroyed), m_log(log) {}

	mock_reducer(const mock_reducer&) = delete;
	mock_reducer(mock_reducer&&) = delete;
	mock_reducer& operator=(const mock_reducer&) = delete;
	mock_reducer& operator=(mock_reducer&&) = delete;

	~mock_reducer() override {
		if(m_destroyed != nullptr) { m_destroyed->store(true); }
	}

	void reduce(void* dest, const void* src, size_t src_count) const override { m_log->push_back(ops::reduce{dest, src, src_count}); }

	void fill_identity(void* dest, size_t count) const override { m_log->push_back(ops::fill_identity{dest, count}); }

  private:
	std::atomic<bool>* m_destroyed;
	operations_log* m_log;
};

class mock_exec_communicator final : public communicator {
  public:
	explicit mock_exec_communicator(operations_log* const log) : m_global_index(s_next_global_index++), m_log(log) {}

	size_t get_num_nodes() const override { return 2; }
	node_id get_local_node_id() const override { return 0; }

	std::vector<inbound_pilot> poll_inbound_pilots() override {
		// TODO somehow queue-receive these from test main thread?
		// or pass in through ctor?
		return {};
	}

	void send_outbound_pilot(const outbound_pilot& pilot) override { m_log->push_back(ops::send_outbound_pilot{pilot}); }

	async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override {
		m_log->push_back(ops::send_payload({to, msgid, base, stride}));
		return make_complete_event();
	}

	async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override {
		m_log->push_back(ops::receive_payload{from, msgid, base, stride});
		return make_complete_event();
	}

	std::unique_ptr<communicator> collective_clone() override {
		auto clone = std::make_unique<mock_exec_communicator>(m_log);
		m_log->push_back(ops::collective_clone{m_global_index, clone->m_global_index});
		return clone;
	}

	void collective_barrier() override { m_log->push_back(ops::collective_barrier{m_global_index}); }

	int get_global_index() const { return m_global_index; }

  private:
	static std::atomic<int> s_next_global_index;

	int m_global_index;
	operations_log* m_log;
};

std::atomic<int> mock_exec_communicator::s_next_global_index{0};

class mock_backend final : public backend {
  public:
	mock_backend(const system_info& system, operations_log* const log) : m_system(system), m_log(log) {}

	const system_info& get_system_info() const override { return m_system; }

	void* debug_alloc(size_t size) override { return malloc(size); }
	void debug_free(void* ptr) override { free(ptr); }

	async_event enqueue_host_alloc(const size_t size, const size_t alignment) override {
		const auto ptr = mock_alloc(size, alignment);
		m_log->push_back(ops::host_alloc{{size, alignment, ptr}});
		return make_complete_event(ptr);
	}

	async_event enqueue_device_alloc(const device_id device, const size_t size, const size_t alignment) override {
		const auto ptr = mock_alloc(size, alignment);
		m_log->push_back(ops::device_alloc{{size, alignment, ptr}, device});
		return make_complete_event(ptr);
	}

	async_event enqueue_host_free(void* const ptr) override {
		m_log->push_back(ops::host_free{{ptr}});
		return make_complete_event();
	}

	async_event enqueue_device_free(const device_id device, void* const ptr) override {
		m_log->push_back(ops::device_free{{ptr}, device});
		return make_complete_event();
	}

	async_event enqueue_host_task(const size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
	    const box<3>& execution_range, const communicator* const collective_comm) override //
	{
		m_log->push_back(ops::host_task{host_lane, std::move(accessor_infos), execution_range, collective_comm});
		return make_complete_event();
	}

	async_event enqueue_device_kernel(const device_id device, const size_t device_lane, const device_kernel_launcher& launcher,
	    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override //
	{
		m_log->push_back(ops::device_kernel{device, device_lane, std::move(accessor_infos), execution_range, reduction_ptrs});
		return make_complete_event();
	}

	async_event enqueue_host_copy(const size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override //
	{
		m_log->push_back(ops::host_copy{{source_base, dest_base, source_box, dest_box, copy_region, elem_size}, host_lane});
		return make_complete_event();
	}

	async_event enqueue_device_copy(const device_id device, const size_t device_lane, const void* const source_base, void* const dest_base,
	    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override //
	{
		m_log->push_back(ops::device_copy{{source_base, dest_base, source_box, dest_box, copy_region, elem_size}, device, device_lane});
		return make_complete_event();
	}

  private:
	system_info m_system;
	uintptr_t m_last_mock_alloc_address = 0;
	operations_log* m_log;

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

class mock_fence_promise final : public fence_promise {
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
			auto backend = std::make_unique<mock_backend>(system, &m_log);
			auto root_comm = std::make_unique<mock_exec_communicator>(&m_log);
			m_root_comm_index = root_comm->get_global_index();
			m_executor = std::make_unique<live_executor>(std::move(backend), std::move(root_comm), static_cast<executor::delegate*>(this));
		}
	}

	executor_test_context(const executor_test_context&) = delete;
	executor_test_context(executor_test_context&&) = delete;
	executor_test_context& operator=(const executor_test_context&) = delete;
	executor_test_context& operator=(executor_test_context&&) = delete;

	~executor_test_context() { REQUIRE(m_has_shut_down); }

	void announce_reducer(const reduction_id rid, std::atomic<bool>* destroyed) {
		m_executor->announce_reducer(rid, std::make_unique<mock_reducer>(destroyed, &m_log));
	}

	void announce_host_object(const host_object_id hoid, std::atomic<bool>* destroyed) {
		m_executor->announce_host_object_instance(hoid, std::make_unique<mock_host_object>(destroyed));
	}

	task_id init() {
		submit<epoch_instruction>(task_manager::initial_epoch_task, epoch_action::none, instruction_garbage{});
		return task_manager::initial_epoch_task;
	}

	task_id horizon(instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		submit<horizon_instruction>(tid, std::move(garbage));
		return tid;
	}

	task_id epoch(const epoch_action action, instruction_garbage garbage = {}) {
		const auto tid = m_next_task_id++;
		submit<epoch_instruction>(tid, action, std::move(garbage));
		return tid;
	}

	void clone_collective_group(const collective_group_id original_cgid, const collective_group_id new_cgid) {
		submit<clone_collective_group_instruction>(original_cgid, new_cgid);
	}

	void alloc(const allocation_id aid, const size_t size, const size_t alignment) { submit<alloc_instruction>(aid, size, alignment); }

	void free(const allocation_id aid) { submit<free_instruction>(aid); }

	void host_task(const box<3>& execution_range, const range<3>& global_range, buffer_access_allocation_map amap, const collective_group_id cgid) {
		submit<host_task_instruction>(host_task_launcher{}, execution_range, global_range, std::move(amap),
		    cgid CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::host_compute, task_id(1), "task_name"));
	}

	void device_kernel(const device_id did, const box<3>& execution_range, buffer_access_allocation_map amap, buffer_access_allocation_map rmap) {
		submit<device_kernel_instruction>(did, device_kernel_launcher{}, execution_range, std::move(amap),
		    std::move(rmap) CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::device_compute, task_id(1), "task_name"));
	}

	void copy(const allocation_with_offset& source_alloc, const allocation_with_offset& dest_alloc, const box<3>& source_box, const box<3>& dest_box,
	    region<3> copy_region, const size_t elem_size) //
	{
		submit<copy_instruction>(source_alloc, dest_alloc, source_box, dest_box, std::move(copy_region), elem_size);
	}

	void destroy_host_object(const host_object_id hoid) { submit<destroy_host_object_instruction>(hoid); }

	void fence_and_wait() {
		mock_fence_promise fence_promise;
		const auto iid = submit<fence_instruction>(&fence_promise);
		fence_promise.wait();
	}

	void fill_identity(const reduction_id rid, const allocation_id aid, const size_t num_values) { submit<fill_identity_instruction>(rid, aid, num_values); }

	void reduce(const reduction_id rid, const allocation_id source_allocation_id, const size_t num_source_values, const allocation_id dest_allocation_id) {
		submit<reduce_instruction>(rid, source_allocation_id, num_source_values, dest_allocation_id);
	}

	void outbound_pilot(const outbound_pilot& pilot) { m_executor->submit({}, {pilot}); }

	void send(const node_id to_nid, const message_id msgid, const allocation_id source_aid, const range<3>& source_alloc_range, const id<3>& offset_in_alloc,
	    const range<3>& send_range, const size_t elem_size) //
	{
		submit<send_instruction>(to_nid, msgid, source_aid, source_alloc_range, offset_in_alloc, send_range, elem_size);
	}

	operations_log finish() {
		CHECK(!m_has_shut_down);
		epoch(epoch_action::shutdown, instruction_garbage{});
		m_executor->wait();
		m_has_shut_down = true;
		return std::move(m_log);
	}

	task_id get_last_epoch() { return m_epochs.get(); }
	task_id get_last_horizon() { return m_horizons.get(); }

	void await_horizon(const task_id tid) { m_horizons.await(tid); }
	void await_epoch(const task_id tid) { m_epochs.await(tid); }

	int get_root_comm_index() const { return m_root_comm_index; }

  private:
	instruction_id m_next_iid = 1;
	std::optional<instruction_id> m_last_iid; // serialize all instructions for simplicity - we do not test OoO capabilities here
	task_id m_next_task_id = task_manager::initial_epoch_task + 1;
	std::vector<std::unique_ptr<instruction>> m_instructions; // we need to guarantee liveness as long as the executor thread is around
	bool m_has_shut_down = false;
	std::unique_ptr<executor> m_executor;
	epoch_monitor m_horizons{0};
	epoch_monitor m_epochs{0};
	operations_log m_log;      // mutated by executor thread - do not access before shutdown!
	int m_root_comm_index = 0; // always 0 for executor_type::dry_run

	template <typename Instruction, typename... CtorParams>
	instruction_id submit(CtorParams&&... ctor_args) {
		const auto iid = instruction_id(m_next_iid++);
		auto instr = std::make_unique<Instruction>(iid, 0, std::forward<CtorParams>(ctor_args)...);
		if(m_last_iid.has_value()) { instr->add_dependency(*m_last_iid); }
		m_executor->submit({instr.get()}, {});
		m_instructions.push_back(std::move(instr));
		m_last_iid = iid;
		return iid;
	}

	void horizon_reached(const task_id tid) override { m_horizons.set(tid); }
	void epoch_reached(const task_id tid) override { m_epochs.set(tid); }
};


TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "executors notify delegate when encountering a horizon / epoch", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	ectx.init();

	CHECK(ectx.get_last_epoch() == task_id(0));
	CHECK(ectx.get_last_horizon() == task_id(0));

	SECTION("on epochs") {
		const auto epoch_tid = ectx.epoch(epoch_action::none);
		ectx.await_epoch(epoch_tid);
		CHECK(ectx.get_last_epoch() == epoch_tid);
	}

	SECTION("on horizons") {
		const auto horizon_tid = ectx.horizon();
		ectx.await_horizon(horizon_tid);
		CHECK(ectx.get_last_horizon() == horizon_tid);
	}

	ectx.finish();
}

TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "dry_run_executor warns when encountering a fence instruction ", "[executor][dry_run]") {
	executor_test_context ectx(executor_type::dry_run);
	ectx.init();
	ectx.fence_and_wait();
	CHECK(test_utils::log_contains_exact(log_level::warn, "Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. The result of this "
	                                                      "operation will not match the expected output of an actual run."));
	ectx.finish();
}

TEST_CASE_METHOD(test_utils::dry_run_executor_fixture, "executors free all reducers that appear in garbage lists  ", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	ectx.init();

	const reduction_id rid(123);
	std::atomic<bool> destroyed{false};
	ectx.announce_reducer(rid, &destroyed);

	SECTION("on epochs") {
		const auto epoch_tid = ectx.epoch(epoch_action::none, instruction_garbage{{rid}, {}});
		ectx.await_epoch(epoch_tid);
	}

	SECTION("on horizons") {
		const auto horizon_tid = ectx.horizon(instruction_garbage{{rid}, {}});
		ectx.await_horizon(horizon_tid);
	}

	ectx.finish();
}

TEST_CASE("host objects lifetime is controlled by destroy_host_object_instruction", "[executor]") {
	const auto executor_type = GENERATE(values({executor_type::dry_run, executor_type::live}));
	CAPTURE(executor_type);

	executor_test_context ectx(executor_type);
	ectx.init();

	const host_object_id hoid(42);
	std::atomic<bool> destroyed{false};
	ectx.announce_host_object(hoid, &destroyed);
	CHECK_FALSE(destroyed.load());

	const auto after_announce_tid = ectx.horizon();
	CHECK_FALSE(destroyed.load());

	ectx.await_horizon(after_announce_tid);
	CHECK_FALSE(destroyed.load());

	ectx.destroy_host_object(hoid);
	const auto after_destroy_tid = ectx.horizon();
	ectx.await_horizon(after_destroy_tid);
	CHECK(destroyed);

	ectx.finish();
}

TEST_CASE("live_executor passes correct allocations to host tasks", "[executor]") {
	executor_test_context ectx(executor_type::live);
	ectx.init();

	ectx.alloc(allocation_id(host_memory_id, 1), 1024, 8);
	ectx.alloc(allocation_id(host_memory_id, 2), 2048, 16);

	const auto amap = buffer_access_allocation_map{
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
	ectx.host_task(box<3>{{0, 1, 2}, {3, 4, 5}}, range<3>{7, 8, 9}, amap, non_collective_group_id);

	ectx.free(allocation_id(host_memory_id, 1));
	ectx.free(allocation_id(host_memory_id, 2));

	const auto log = ectx.finish();
	REQUIRE(log.size() == 5);

	const auto alloc1 = std::get<ops::host_alloc>(log[0]);
	CHECK(alloc1.size == 1024);
	CHECK(alloc1.alignment == 8);
	CHECK(alloc1.result != nullptr);

	const auto alloc2 = std::get<ops::host_alloc>(log[1]);
	CHECK(alloc2.size == 2048);
	CHECK(alloc2.alignment == 16);
	CHECK(alloc2.result != nullptr);

	const auto host_task = std::get<ops::host_task>(log[2]);
	CHECK(host_task.collective_comm == nullptr);
	CHECK(host_task.execution_range == box<3>{{0, 1, 2}, {3, 4, 5}});

	REQUIRE(host_task.accessor_infos.size() == 2);
	CHECK(host_task.accessor_infos[0].ptr == alloc1.result);
	CHECK(host_task.accessor_infos[0].allocated_box_in_buffer == amap[0].allocated_box_in_buffer);
	CHECK(host_task.accessor_infos[0].accessed_box_in_buffer == amap[0].accessed_box_in_buffer);
	CHECK(host_task.accessor_infos[1].ptr == alloc2.result);
	CHECK(host_task.accessor_infos[1].allocated_box_in_buffer == amap[1].allocated_box_in_buffer);
	CHECK(host_task.accessor_infos[1].accessed_box_in_buffer == amap[1].accessed_box_in_buffer);

	const auto free1 = std::get<ops::host_free>(log[3]);
	CHECK(free1.ptr == alloc1.result);

	const auto free2 = std::get<ops::host_free>(log[4]);
	CHECK(free2.ptr == alloc2.result);
}

TEST_CASE("live_executor passes correct allocations to reducers", "[executor]") {
	executor_test_context ectx(executor_type::live);
	ectx.init();

	const auto source_aid = allocation_id(host_memory_id, 1);
	const auto dest_aid = allocation_id(host_memory_id, 2);
	const size_t num_source_values = 4;
	ectx.alloc(source_aid, 4 * sizeof(int), alignof(int));
	ectx.alloc(dest_aid, sizeof(int), alignof(int));

	const reduction_id rid(42);
	ectx.announce_reducer(rid, nullptr);
	ectx.fill_identity(rid, source_aid, num_source_values);
	ectx.reduce(rid, source_aid, num_source_values, dest_aid);

	ectx.free(source_aid);
	ectx.free(dest_aid);

	const auto log = ectx.finish();
	REQUIRE(log.size() == 6);

	const auto source_alloc = std::get<ops::host_alloc>(log[0]);
	const auto dest_alloc = std::get<ops::host_alloc>(log[1]);

	const auto fill_identity = std::get<ops::fill_identity>(log[2]);
	CHECK(fill_identity.dest == source_alloc.result);
	CHECK(fill_identity.count == num_source_values);

	const auto reduce = std::get<ops::reduce>(log[3]);
	CHECK(reduce.dest == dest_alloc.result);
	CHECK(reduce.src == source_alloc.result);
	CHECK(reduce.src_count == num_source_values);

	// correct arguments to alloc / free have already been tested above
}

TEST_CASE("live_executor passes correct allocations to device kernels", "[executor]") {
	executor_test_context ectx(executor_type::live);
	ectx.init();

	const auto did = GENERATE(values<device_id>({0, 1}));
	const auto mid = memory_id(first_device_memory_id + did);

	ectx.alloc(allocation_id(mid, 1), 1024, 8);
	ectx.alloc(allocation_id(mid, 2), 2048, 16);
	ectx.alloc(allocation_id(mid, 3), 4, 4);
	ectx.alloc(allocation_id(mid, 4), 4, 4);

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
	ectx.device_kernel(did, box<3>({1, 2, 3}, {4, 5, 6}), amap, rmap);

	ectx.free(allocation_id(mid, 1));
	ectx.free(allocation_id(mid, 2));
	ectx.free(allocation_id(mid, 3));
	ectx.free(allocation_id(mid, 4));

	const auto log = ectx.finish();
	REQUIRE(log.size() == 9);

	const auto alloc1 = std::get<ops::device_alloc>(log[0]);
	CHECK(alloc1.device == did);
	CHECK(alloc1.size == 1024);
	CHECK(alloc1.alignment == 8);
	CHECK(alloc1.result != nullptr);

	const auto alloc2 = std::get<ops::device_alloc>(log[1]);
	CHECK(alloc2.device == did);
	CHECK(alloc2.size == 2048);
	CHECK(alloc2.alignment == 16);
	CHECK(alloc2.result != nullptr);

	const auto alloc3 = std::get<ops::device_alloc>(log[2]);
	CHECK(alloc3.device == did);
	CHECK(alloc3.size == 4);
	CHECK(alloc3.alignment == 4);
	CHECK(alloc3.result != nullptr);

	const auto alloc4 = std::get<ops::device_alloc>(log[3]);
	CHECK(alloc4.device == did);
	CHECK(alloc4.size == 4);
	CHECK(alloc4.alignment == 4);
	CHECK(alloc4.result != nullptr);

	const auto kernel = std::get<ops::device_kernel>(log[4]);
	CHECK(kernel.device == did);
	CHECK(kernel.execution_range == box<3>({1, 2, 3}, {4, 5, 6}));

	REQUIRE(kernel.accessor_infos.size() == 2);
	CHECK(kernel.accessor_infos[0].ptr == alloc1.result);
	CHECK(kernel.accessor_infos[0].allocated_box_in_buffer == amap[0].allocated_box_in_buffer);
	CHECK(kernel.accessor_infos[0].accessed_box_in_buffer == amap[0].accessed_box_in_buffer);
	CHECK(kernel.accessor_infos[1].ptr == alloc2.result);
	CHECK(kernel.accessor_infos[1].allocated_box_in_buffer == amap[1].allocated_box_in_buffer);
	CHECK(kernel.accessor_infos[1].accessed_box_in_buffer == amap[1].accessed_box_in_buffer);

	CHECK(kernel.reduction_ptrs == std::vector{alloc3.result, alloc4.result});

	const auto free1 = std::get<ops::device_free>(log[5]);
	CHECK(free1.device == did);
	CHECK(free1.ptr == alloc1.result);

	const auto free2 = std::get<ops::device_free>(log[6]);
	CHECK(free2.device == did);
	CHECK(free2.ptr == alloc2.result);

	const auto free3 = std::get<ops::device_free>(log[7]);
	CHECK(free3.device == did);
	CHECK(free3.ptr == alloc3.result);

	const auto free4 = std::get<ops::device_free>(log[8]);
	CHECK(free4.device == did);
	CHECK(free4.ptr == alloc4.result);
}

TEST_CASE("live_executor passes correct allocation pointers to copy instructions", "[executor]") {
	executor_test_context ectx(executor_type::live);
	ectx.init();

	const auto did = device_id(1);
	const auto source_mid = GENERATE(values({host_memory_id, memory_id(first_device_memory_id + 1)}));
	const auto dest_mid = GENERATE(values({host_memory_id, memory_id(first_device_memory_id + 1)}));
	const auto source_offset = GENERATE(values<size_t>({0, 16}));
	const auto dest_offset = GENERATE(values<size_t>({0, 32}));

	const auto source_aid = allocation_id(source_mid, 1);
	ectx.alloc(source_aid, 4096, 8);
	const auto dest_aid = allocation_id(dest_mid, 2);
	ectx.alloc(dest_aid, 4096, 8);

	const auto source_box = box<3>{{4, 0, 0}, {16, 16, 1}};
	const auto dest_box = box<3>{{8, 0, 0}, {16, 16, 1}};
	const auto copy_region = region<3>({box<3>({8, 0, 0}, {12, 8, 1}), box<3>({12, 0, 0}, {16, 4, 1})});
	const auto elem_size = 4;

	ectx.copy(allocation_with_offset(source_aid, source_offset), allocation_with_offset(dest_aid, dest_offset), source_box, dest_box, copy_region, elem_size);

	ectx.free(source_aid);
	ectx.free(dest_aid);

	const auto log = ectx.finish();
	REQUIRE(log.size() == 5);

	const ops::common_alloc* source_alloc = nullptr;
	if(source_mid == host_memory_id) {
		source_alloc = &std::get<ops::host_alloc>(log[0]);
	} else {
		const auto& source_device_alloc = std::get<ops::device_alloc>(log[0]);
		CHECK(source_device_alloc.device == did);
		source_alloc = &source_device_alloc;
	}
	CHECK(source_alloc->size == 4096);
	CHECK(source_alloc->alignment == 8);
	CHECK(source_alloc->result != nullptr);

	const ops::common_alloc* dest_alloc = nullptr;
	if(dest_mid == host_memory_id) {
		dest_alloc = &std::get<ops::host_alloc>(log[1]);
	} else {
		const auto& dest_device_alloc = std::get<ops::device_alloc>(log[1]);
		CHECK(dest_device_alloc.device == did);
		dest_alloc = &dest_device_alloc;
	}
	CHECK(dest_alloc->size == 4096);
	CHECK(dest_alloc->alignment == 8);
	CHECK(dest_alloc->result != nullptr);

	const ops::common_copy* copy = nullptr;
	if(source_mid == host_memory_id && dest_mid == host_memory_id) {
		copy = &std::get<ops::host_copy>(log[2]);
	} else {
		const auto& device_copy = std::get<ops::device_copy>(log[2]);
		CHECK(device_copy.device == did);
		copy = &device_copy;
	}
	CHECK(copy->source_base == reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(source_alloc->result) + source_offset));
	CHECK(copy->dest_base == reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dest_alloc->result) + dest_offset));
	CHECK(copy->source_box == source_box);
	CHECK(copy->dest_box == dest_box);
	CHECK(copy->copy_region == copy_region);
	CHECK(copy->elem_size == elem_size);

	if(source_mid == host_memory_id) {
		const auto source_free = std::get<ops::host_free>(log[3]);
		CHECK(source_free.ptr == source_alloc->result);
	} else {
		const auto source_free = std::get<ops::device_free>(log[3]);
		CHECK(source_free.device == did);
		CHECK(source_free.ptr == source_alloc->result);
	}

	if(dest_mid == host_memory_id) {
		const auto dest_free = std::get<ops::host_free>(log[4]);
		CHECK(dest_free.ptr == dest_alloc->result);
	} else {
		const auto dest_free = std::get<ops::device_free>(log[4]);
		CHECK(dest_free.device == did);
		CHECK(dest_free.ptr == dest_alloc->result);
	}
}

TEST_CASE("live_executor clones and submits barriers on the right communicators", "[executor]") {
	executor_test_context ectx(executor_type::live);
	ectx.init();

	// clone a tree: root -> clone1; root -> clone2; clone1 -> clone3
	ectx.clone_collective_group(root_collective_group_id, root_collective_group_id + 1);
	ectx.clone_collective_group(root_collective_group_id, root_collective_group_id + 2);
	ectx.clone_collective_group(root_collective_group_id + 1, root_collective_group_id + 3);
	ectx.epoch(epoch_action::barrier); // must happen on root communicator

	const auto log = ectx.finish();
	REQUIRE(log.size() == 4);

	const auto clone1 = std::get<ops::collective_clone>(log[0]);
	CHECK(clone1.parent_comm_index == ectx.get_root_comm_index());
	CHECK(clone1.child_comm_index != clone1.parent_comm_index);

	const auto clone2 = std::get<ops::collective_clone>(log[1]);
	CHECK(clone2.parent_comm_index == ectx.get_root_comm_index());
	CHECK(clone2.child_comm_index != clone2.parent_comm_index);
	CHECK(clone2.child_comm_index != clone1.parent_comm_index);

	const auto clone3 = std::get<ops::collective_clone>(log[2]);
	CHECK(clone3.parent_comm_index == clone1.child_comm_index);
	CHECK(clone3.child_comm_index != clone3.parent_comm_index);
	CHECK(clone3.child_comm_index != ectx.get_root_comm_index());

	const auto barrier = std::get<ops::collective_barrier>(log[3]);
	CHECK(barrier.comm_index == ectx.get_root_comm_index());
}

TEST_CASE("live_executor passes the correct metadata and pointers for p2p send", "[executor]") {
	executor_test_context ectx(executor_type::live);

	// outbound pilots are sent immediately, so we post them first to get a deterministically ordered log
	const node_id peer = 1;
	const message_id msgid = 123;
	const transfer_id trid{1, 2, 0};
	const box<3> send_box{{0, 1, 2}, {4, 5, 6}};
	ectx.outbound_pilot({peer, {msgid, trid, send_box}});

	ectx.init();

	const auto source_aid = allocation_id(host_memory_id, 1);
	const auto source_box = box<3>{{0, 0, 0}, {7, 8, 9}};
	ectx.alloc(source_aid, source_box.get_area() * sizeof(int), alignof(int));
	ectx.send(peer, msgid, source_aid, source_box.get_range(), send_box.get_offset(), send_box.get_range(), sizeof(int));

	ectx.free(source_aid);

	const auto log = ectx.finish();
	REQUIRE(log.size() == 4);

	const auto send_pilot = std::get<ops::send_outbound_pilot>(log[0]);
	CHECK(send_pilot.pilot.to == peer);
	CHECK(send_pilot.pilot.message.id == msgid);
	CHECK(send_pilot.pilot.message.transfer_id == trid);
	CHECK(send_pilot.pilot.message.box == send_box);

	const auto alloc = std::get<ops::host_alloc>(log[1]);
	CHECK(alloc.size == source_box.get_area() * sizeof(int));
	CHECK(alloc.alignment == alignof(int));

	const auto send_payload = std::get<ops::send_payload>(log[2]);
	CHECK(send_payload.to == peer);
	CHECK(send_payload.base == alloc.result);
	CHECK(send_payload.msgid == msgid);
	CHECK(send_payload.stride.allocation_range == source_box.get_range());
	CHECK(send_payload.stride.transfer == send_box.get_subrange());
	CHECK(send_payload.stride.element_size == sizeof(int));

	const auto free = std::get<ops::host_free>(log[3]);
	CHECK(free.ptr == alloc.result);
}
