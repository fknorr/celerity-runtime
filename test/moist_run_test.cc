#include "communicator.h"
#include "instruction_graph_test_utils.h"
#include "live_executor.h"
#include "nd_memory.h"
#include "test_utils.h"

#include <any>
#include <condition_variable>
#include <mutex>


namespace celerity::test_utils {

class mock_execution_interconnect {
  public:
	using namespace_id = size_t;
	using collective_id = size_t;

	using stride = detail::communicator::stride;

	enum class collective_operation { none, clone, barrier, destroy };

	constexpr inline static namespace_id root_namespace_id = 0;

	explicit mock_execution_interconnect(const size_t num_nodes) : m_num_nodes(num_nodes), m_namespaces{{root_namespace_id, per_namespace_state(num_nodes)}} {}

	size_t get_num_nodes() const { return m_num_nodes; }

	void send_outbound_pilot(const namespace_id nsid, const node_id sender, const outbound_pilot& pilot) {
		std::lock_guard<std::mutex> lock(m_mutex);
		get_node(nsid, pilot.to).pilot_queue.push_back(inbound_pilot{sender, pilot.message});
	}

	std::vector<inbound_pilot> poll_inbound_pilots(const namespace_id nsid, const node_id nid) {
		std::lock_guard<std::mutex> lock(m_mutex);
		return std::move(get_node(nsid, nid).pilot_queue);
	}

	async_event send_payload(
	    const namespace_id nsid, const node_id from, const node_id to, const message_id msgid, const void* const base, const stride& stride) {
		std::lock_guard<std::mutex> lock(m_mutex);
		auto& recv_queue = get_node(nsid, to).receive_queue;
		if(const auto recv = std::find_if(recv_queue.begin(), recv_queue.end(), //
		       [&](const std::shared_ptr<pending_receive>& p) { return p->from == from && p->msgid == msgid; });
		    recv != recv_queue.end()) {
			peer_to_peer(base, (*recv)->receive_base, stride, (*recv)->receive_stride);
			return make_complete_event();
		} else {
			return make_pending_event(get_node(nsid, from).send_queue.emplace_back(std::make_shared<pending_send>(pending_send{to, msgid, base, stride})));
		}
	}

	async_event receive_payload(const namespace_id nsid, const node_id to, const node_id from, const message_id msgid, void* const base, const stride& stride) {
		std::lock_guard<std::mutex> lock(m_mutex);
		auto& send_queue = get_node(nsid, from).send_queue;
		if(const auto send = std::find_if(send_queue.begin(), send_queue.end(), //
		       [&](const std::shared_ptr<pending_send>& p) { return p->to == to && p->msgid == msgid; });
		    send != send_queue.end()) {
			peer_to_peer((*send)->send_base, base, (*send)->send_stride, stride);
			return make_complete_event();
		} else {
			return make_pending_event(
			    get_node(nsid, to).receive_queue.emplace_back(std::make_shared<pending_receive>(pending_receive{to, msgid, base, stride})));
		}
	}

	namespace_id collective_clone(const namespace_id nsid, const node_id nid) {
		namespace_id clone_nsid = 0;
		collective(
		    nsid, nid, collective_operation::clone,
		    [&] /* arrive */ (const size_t arrive_idx, std::any& result) {
			    if(arrive_idx == m_num_nodes - 1) {
				    std::lock_guard lock(m_mutex);
				    result = clone_nsid = create_namespace();
			    } else {
				    clone_nsid = std::any_cast<namespace_id>(result);
			    }
		    },
		    [] /* complete */ (const size_t /* complete_idx */, std::any& /* result */) {});
		return clone_nsid;
	}

	void collective_barrier(const namespace_id nsid, const node_id nid) {
		collective(
		    nsid, nid, collective_operation::barrier, //
		    [&] /* arrive */ (const size_t /* arrive_idx */, std::any& /* result */) {},
		    [] /* complete */ (const size_t /* complete_idx */, std::any& /* result */) {});
	}

	void collective_destroy(const namespace_id nsid, const node_id nid) {
		collective(
		    nsid, nid, collective_operation::destroy, //
		    [&] /* arrive */ (const size_t /* arrive_idx */, std::any& /* result */) {},
		    [&] /* complete */ (const size_t complete_idx, std::any& /* result */) {
			    if(complete_idx == m_num_nodes - 1) {
				    std::lock_guard lock(m_mutex);
				    m_namespaces.erase(nsid);
			    }
		    });
	}

  private:
	struct pending_send {
		node_id to = 0;
		message_id msgid = 0;
		const void* send_base = nullptr;
		stride send_stride;
	};

	struct pending_receive {
		node_id from = 0;
		message_id msgid = 0;
		void* receive_base = nullptr;
		stride receive_stride;
	};

	struct per_node_state {
		std::vector<inbound_pilot> pilot_queue;
		std::vector<std::shared_ptr<pending_send>> send_queue; // shared_ptrs: we're using weak_ptr.expired to implement thread-safe async events
		std::vector<std::shared_ptr<pending_receive>> receive_queue;
		collective_id last_collective_id = 0;
	};

	struct collective_latch {
		std::mutex mutex;
		std::any result;
		bool do_continue = false;
		std::condition_variable continue_cv;
		size_t num_nodes_continued = 0;
	};

	struct collective_state {
		collective_id id = 0;
		collective_operation operation = collective_operation::none;
		size_t num_nodes_arrived = 0;
		std::shared_ptr<collective_latch> latch;
	};

	struct per_namespace_state {
		std::vector<per_node_state> nodes;
		collective_state current_collective;
		collective_id next_collective_id = 1;
		explicit per_namespace_state(const size_t num_nodes) : nodes(num_nodes) {}
		per_node_state& get_node(const node_id nid) { return nodes.at(nid); }
	};

	class pending_event : public detail::async_event_impl {
	  public:
		pending_event(std::weak_ptr<void> pending) : m_pending(std::move(pending)) {}
		bool is_complete() override { return m_pending.expired(); }

	  private:
		std::weak_ptr<void> m_pending;
	};

	size_t m_num_nodes;
	std::mutex m_mutex;
	std::unordered_map<namespace_id, per_namespace_state> m_namespaces;
	namespace_id m_next_namespace_id = 1;

	static async_event make_pending_event(std::weak_ptr<void> pending) { return async_event(std::make_unique<pending_event>(std::move(pending))); }

	void peer_to_peer(const void* send_base, void* receive_base, const stride& send_stride, const stride& receive_stride) {
		CHECK(send_stride.transfer.range == receive_stride.transfer.range);
		CHECK(send_stride.element_size == receive_stride.element_size);
		detail::nd_copy_host(send_base, receive_base, send_stride.allocation_range, receive_stride.allocation_range, send_stride.transfer.offset,
		    receive_stride.transfer.offset, send_stride.transfer.range, send_stride.element_size);
	}

	namespace_id create_namespace() {
		const namespace_id nsid = m_next_namespace_id++;
		m_namespaces.emplace(nsid, m_num_nodes);
		return nsid;
	}

	per_namespace_state& get_namespace(const namespace_id nsid) { return m_namespaces.at(nsid); }
	per_node_state& get_node(const namespace_id nsid, const node_id nid) { return get_namespace(nsid).get_node(nid); }

	template <typename Arrive, typename Complete>
	void collective(const namespace_id nsid, const node_id nid, const collective_operation operation, Arrive&& arrive, Complete&& complete) {
		size_t arrive_idx = 0;
		std::shared_ptr<collective_latch> latch;

		{
			std::lock_guard lock(m_mutex);
			auto& ns = get_namespace(nsid);
			auto& node = ns.get_node(nid);

			if(ns.current_collective.num_nodes_arrived == 0) {
				ns.current_collective.id = ns.next_collective_id++;
				ns.current_collective.operation = operation;
				ns.current_collective.latch = latch = std::make_shared<collective_latch>();
			} else {
				ns.current_collective.operation = operation;
				latch = ns.current_collective.latch;
			}
			REQUIRE(ns.current_collective.id == node.last_collective_id + 1);
			node.last_collective_id = ns.current_collective.id;
			arrive_idx = ns.current_collective.num_nodes_arrived;
			ns.current_collective.num_nodes_arrived = arrive_idx < m_num_nodes - 1 ? arrive_idx + 1 : 0;
			if(ns.current_collective.num_nodes_arrived == m_num_nodes) { ns.current_collective.latch.reset(); }
		}

		{
			std::unique_lock lock(latch->mutex);
			arrive(arrive_idx, latch->result);

			if(arrive_idx == m_num_nodes - 1) {
				latch->do_continue = true;
				latch->continue_cv.notify_all();
			} else {
				latch->continue_cv.wait(lock, [&] { return latch->do_continue; });
			}
			const auto complete_idx = latch->num_nodes_continued;
			latch->num_nodes_continued += 1;

			complete(complete_idx, latch->result);
		}
	}
};

class mock_execution_communicator final : public detail::communicator {
  public:
	using namespace_id = mock_execution_interconnect::namespace_id;

	mock_execution_communicator(mock_execution_interconnect* interconnect, const namespace_id nsid, const node_id nid, bool from_collective_clone = false)
	    : m_interconnect(interconnect), m_nsid(nsid), m_nid(nid), m_from_collective_clone(from_collective_clone) {}

	mock_execution_communicator(const mock_execution_communicator&) = delete;
	mock_execution_communicator(mock_execution_communicator&&) = delete;
	mock_execution_communicator& operator=(const mock_execution_communicator&) = delete;
	mock_execution_communicator& operator=(mock_execution_communicator&&) = delete;

	~mock_execution_communicator() override {
		// only communicators originating from collective_clone are tied to the executor thread(s)
		if(m_from_collective_clone) { m_interconnect->collective_destroy(m_nsid, m_nid); }
	}

	size_t get_num_nodes() const override { return m_interconnect->get_num_nodes(); }

	node_id get_local_node_id() const override { return m_nid; }

	void send_outbound_pilot(const outbound_pilot& pilot) override { m_interconnect->send_outbound_pilot(m_nsid, m_nid, pilot); }

	std::vector<inbound_pilot> poll_inbound_pilots() override { return m_interconnect->poll_inbound_pilots(m_nsid, m_nid); }

	async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override {
		return m_interconnect->send_payload(m_nsid, m_nid, to, msgid, base, stride);
	}

	async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override {
		return m_interconnect->receive_payload(m_nsid, m_nid, from, msgid, base, stride);
	}

	std::unique_ptr<communicator> collective_clone() override {
		return std::make_unique<mock_execution_communicator>(
		    m_interconnect, m_interconnect->collective_clone(m_nsid, m_nid), m_nid, true /* from_collective_clone */);
	}

	void collective_barrier() override { m_interconnect->collective_barrier(m_nsid, m_nid); }

  private:
	mock_execution_interconnect* m_interconnect;
	namespace_id m_nsid;
	size_t m_nid;
	bool m_from_collective_clone = false;
};

class mock_execution_backend final : public detail::backend {
  public:
	void* debug_alloc(const size_t size) override { return malloc(size); }

	void debug_free(void* const ptr) override { free(ptr); }

	async_event enqueue_host_alloc(const size_t size, const size_t alignment) override { return make_complete_event(aligned_alloc(alignment, size)); }

	async_event enqueue_device_alloc(const device_id memory_device, const size_t size, const size_t alignment) override {
		(void)memory_device, (void)size, (void)alignment;
		return make_complete_event(static_cast<void*>(nullptr));
	}

	async_event enqueue_host_free(void* const ptr) override {
		free(ptr);
		return make_complete_event();
	}

	async_event enqueue_device_free(const device_id memory_device, void* const ptr) override {
		(void)memory_device;
		return make_complete_event();
	}

	async_event enqueue_host_function(const size_t host_lane, const std::function<void()> fn) override { return make_complete_event(); }

	async_event enqueue_device_kernel(const device_id device, const size_t device_lane, const device_kernel_launcher& launcher, const box<3>& execution_range,
	    const std::vector<void*>& reduction_ptrs) override {
		return make_complete_event();
	}

	async_event enqueue_device_copy(const device_id device, const size_t device_lane, const void* const source_base, void* const dest_base,
	    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override {
		return make_complete_event();
	}

	async_event enqueue_host_copy(const size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override {
		return make_complete_event();
	}
};

class mock_node final : public detail::instruction_graph_generator::delegate, public detail::executor::delegate {
  public:
	explicit mock_node(mock_execution_interconnect& interconnect, const node_id local_nid, const size_t num_devices)
	    : m_monitor(0),
	      m_executor(make_system_info(num_devices, true /* supports d2d copies */), std::make_unique<mock_execution_backend>(),
	          std::make_unique<test_utils::mock_execution_communicator>(&interconnect, test_utils::mock_execution_interconnect::root_namespace_id, local_nid),
	          this),
	      m_ictx(interconnect.get_num_nodes(), local_nid, num_devices, true /* d2d */, {} /* policy */, this) {}

	void flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override {
		m_executor.submit(std::move(instructions), std::move(pilots));
	}

	void horizon_reached(const detail::task_id tid) override { (void)tid; }
	void epoch_reached(const detail::task_id tid) override { m_monitor.set(tid); }

	idag_test_context& ictx() { return m_ictx; }
	void wait() { m_executor.wait(); }

  private:
	detail::epoch_monitor m_monitor;
	detail::live_executor m_executor;
	idag_test_context m_ictx;
};

} // namespace celerity::test_utils

TEST_CASE("executor instrumentation", "[live_executor]") {
	test_utils::mock_execution_interconnect interconnect(4);
	std::vector<std::unique_ptr<test_utils::mock_node>> nodes(4);
	for(node_id nid = 0; nid < 4; ++nid) {
		nodes[nid] = std::make_unique<test_utils::mock_node>(interconnect, nid, 4 /* num devices */);
	}
	for(node_id nid = 0; nid < 4; ++nid) {
		nodes[nid]->ictx().epoch(epoch_action::barrier);
	}
	for(node_id nid = 0; nid < 4; ++nid) {
		nodes[nid]->ictx().finish();
	}
	for(node_id nid = 0; nid < 4; ++nid) {
		nodes[nid]->wait();
	}
}

// Test accessor hydration
// Can we avoid actually allocating any memory?