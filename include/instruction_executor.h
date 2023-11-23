#pragma once

#include "backend/queue.h"
#include "communicator.h"
#include "double_buffered_queue.h"
#include "instruction_graph.h"
#include "receive_arbiter.h"
#include "reduction.h"

#include <unordered_map>

namespace celerity::detail {

struct host_object_instance;
struct oob_bounding_box;

class instruction_executor {
  public:
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		virtual void horizon_reached(task_id tid) = 0;
		virtual void epoch_reached(task_id tid) = 0;
	};

	instruction_executor(std::unique_ptr<backend::queue> backend_queue, std::unique_ptr<communicator> comm, delegate* dlg);
	instruction_executor(const instruction_executor&) = delete;
	instruction_executor(instruction_executor&&) = delete;
	instruction_executor& operator=(const instruction_executor&) = delete;
	instruction_executor& operator=(instruction_executor&&) = delete;
	~instruction_executor();

	void wait();

	void submit_instruction(const instruction& instr); // TODO should receive pointer
	void submit_pilot(const outbound_pilot& pilot);

	void announce_buffer_user_pointer(buffer_id bid, const void* ptr);
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance);
	void announce_reduction(reduction_id rid, std::unique_ptr<runtime_reduction> reduction);

  private:
	friend struct executor_testspy;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	struct boundary_check_info {
		struct accessor_info {
			detail::buffer_id buffer_id;
			std::string buffer_name;
			box<3> accessible_box;
		};
		oob_bounding_box* illegal_access_bounding_boxes = nullptr;
		std::vector<accessor_info> accessors;
		detail::task_id task_id;
		std::string task_name;
		celerity::target target;
	};

	class boundary_checked_event final : public async_event_base {
	  public:
		struct incomplete {
			instruction_executor* executor;
			async_event launch_event;
			boundary_check_info oob_info;
		};

		boundary_checked_event(instruction_executor* const executor, async_event&& launch_event, boundary_check_info&& oob_info)
		    : m_state(incomplete{executor, std::move(launch_event), std::move(oob_info)}) {}

		bool is_complete() const override;

	  private:
		mutable std::optional<incomplete> m_state;
	};
#endif

	struct buffer_user_pointer_announcement {
		buffer_id bid;
		const void* ptr;
	};
	struct host_object_instance_announcement {
		host_object_id hoid;
		std::unique_ptr<host_object_instance> instance;
	};
	struct reduction_announcement {
		reduction_id rid;
		std::unique_ptr<runtime_reduction> reduction;
	};
	using submission =
	    std::variant<const instruction*, outbound_pilot, buffer_user_pointer_announcement, host_object_instance_announcement, reduction_announcement>;

	struct pending_instruction_info;
	struct active_instruction_info;

	// immutable
	delegate* m_delegate;
	std::unique_ptr<communicator> m_communicator;

	// accessed by by main and executor threads
	double_buffered_queue<submission> m_submission_queue;

	// accessed by executor thread only (unsynchronized)
	bool m_expecting_more_submissions = true;
	std::unique_ptr<backend::queue> m_backend_queue;
	std::unordered_map<buffer_id, const void*> m_buffer_user_pointers;
	std::unordered_map<allocation_id, void*> m_allocations;
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> m_host_object_instances;
	std::unordered_map<collective_group_id, communicator::collective_group*> m_collective_groups;
	std::unordered_map<reduction_id, std::unique_ptr<runtime_reduction>> m_reductions;
	receive_arbiter m_recv_arbiter;
	host_queue m_host_queue;

	std::thread m_thread;

	void loop();
	void thread_main();

	[[nodiscard]] active_instruction_info begin_executing(const instruction& instr);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	boundary_check_info prepare_accessor_boundary_check(const buffer_access_allocation_map& amap, task_id tid, const std::string& task_name, target target);
#endif

	void prepare_accessor_hydration(
	    target target, const buffer_access_allocation_map& amap CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, const boundary_check_info& oob_info));
};

} // namespace celerity::detail