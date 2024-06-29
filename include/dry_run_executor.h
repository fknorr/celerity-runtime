#pragma once

#include "double_buffered_queue.h"
#include "executor.h"

#include <thread>
#include <variant>

namespace celerity::detail {

/// Executor implementation selected when Celerity performs a dry run, that is, graph generation for debugging purposes without actually allocating memory,
/// launching kernels or issuing data transfers.
///
/// `dry_run_executor` still executes horizon-, epoch- and fence instructions to guarantee forward progress in the user application.
class dry_run_executor final : public executor {
  public:
	/// `dlg` (optional) receives notifications about reached horizons and epochs from the executor thread.
	explicit dry_run_executor(delegate* dlg);

	void announce_user_allocation(allocation_id aid, void* ptr) override;
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) override;
	void announce_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) override;

	void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override;

	void wait() override;

  private:
	using host_object_instance_announcement = std::pair<host_object_id, std::unique_ptr<host_object_instance>>;
	using submission = std::variant<std::vector<const instruction*>, host_object_instance_announcement>;

	double_buffered_queue<submission> m_submission_queue;
	std::thread m_thread;

	void thread_main(delegate* dlg);
};

} // namespace celerity::detail
