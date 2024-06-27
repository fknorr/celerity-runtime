#pragma once

#include "double_buffered_queue.h"
#include "executor.h"

#include <memory>
#include <thread>
#include <variant>

namespace celerity::detail::live_executor_detail {

struct instruction_pilot_batch {
	std::vector<const instruction*> instructions;
	std::vector<outbound_pilot> pilots;
};
struct user_allocation_announcement {
	allocation_id aid;
	void* ptr = nullptr;
};
struct host_object_instance_announcement {
	host_object_id hoid = 0;
	std::unique_ptr<host_object_instance> instance;
};
struct reducer_announcement {
	reduction_id rid = 0;
	std::unique_ptr<reducer> reduction;
};
using submission = std::variant<instruction_pilot_batch, user_allocation_announcement, host_object_instance_announcement, reducer_announcement>;

} // namespace celerity::detail::live_executor_detail

namespace celerity::detail {

class communicator;
struct system_info;
class backend;

class live_executor final : public executor {
  public:
	explicit live_executor(std::unique_ptr<backend> backend, std::unique_ptr<communicator> root_comm, delegate* dlg);
	live_executor(const live_executor&) = delete;
	live_executor(live_executor&&) = delete;
	live_executor& operator=(const live_executor&) = delete;
	live_executor& operator=(live_executor&&) = delete;
	~live_executor() override;

	void announce_user_allocation(allocation_id aid, void* ptr) override;
	void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) override;
	void announce_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) override;

	void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override;

	void wait() override;

  private:
	friend struct executor_testspy;

	std::unique_ptr<communicator> m_root_comm; // created and destroyed outside of executor thread
	double_buffered_queue<live_executor_detail::submission> m_submission_queue;
	std::thread m_thread;

	void thread_main(std::unique_ptr<backend> backend, delegate* dlg);
};

} // namespace celerity::detail
