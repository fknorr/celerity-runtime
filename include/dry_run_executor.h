#pragma once

#include "executor.h"

#include <condition_variable>
#include <mutex>

namespace celerity::detail {

class dry_run_executor final : public executor {
  public:
	explicit dry_run_executor(delegate* dlg) : m_delegate(dlg) {}

	void announce_user_allocation(allocation_id aid, void* ptr) override;
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) override;
	void announce_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) override;

	void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override;

	void wait() override;

  private:
	delegate* m_delegate;

	// for blocking in wait()
	std::mutex m_resume_mutex;
	std::condition_variable m_resume;
	bool m_has_shut_down = false;

	// host objects must live exactly as long as they would with an live_executor
	std::mutex m_host_object_instances_mutex;
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> m_host_object_instances;
};

} // namespace celerity::detail
