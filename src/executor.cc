#include "executor.h"

#include <queue>

#include "distr_queue.h"
#include "log.h"
#include "mpi_support.h"

// TODO: Get rid of this. (This could potentialy even cause deadlocks on large clusters)
constexpr size_t MAX_CONCURRENT_JOBS = 20;

namespace celerity {
namespace detail {
	void duration_metric::resume() {
		assert(!running);
		current_start = clock.now();
		running = true;
	}

	void duration_metric::pause() {
		assert(running);
		duration += std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - current_start);
		running = false;
	}

	executor::executor(
	    node_id local_nid, host_queue& h_queue, device_queue& d_queue, task_manager& tm, buffer_manager& buffer_mngr, reduction_manager& reduction_mngr)
	    : local_nid(local_nid), h_queue(h_queue), d_queue(d_queue), task_mngr(tm), buffer_mngr(buffer_mngr), reduction_mngr(reduction_mngr) {
		btm = std::make_unique<buffer_transfer_manager>();
		metrics.initial_idle.resume();
	}

	void executor::startup() { exec_thrd = std::thread(&executor::run, this); }

	void executor::shutdown() {
		if(exec_thrd.joinable()) { exec_thrd.join(); }

		CELERITY_DEBUG("Executor initial idle time = {}us, compute idle time = {}us, starvation time = {}us", metrics.initial_idle.get().count(),
		    metrics.device_idle.get().count(), metrics.starvation.get().count());
	}

	void executor::run() {
		bool done = false;

		std::queue<command_info> command_queue;
		std::unordered_set<command_id> running_jobs;

		while(!done || !jobs.empty()) {
			// Bail if a device error ocurred.
			if(running_device_compute_jobs > 0) { d_queue.get_sycl_queue().throw_asynchronous(); }

			// We poll transfers from here (in the same thread, interleaved with job updates),
			// as it allows us to omit any sort of locking when interacting with the BTM through jobs.
			// This actually makes quite a big difference, especially for lots of small transfers.
			// The BTM uses non-blocking MPI routines internally, making this a relatively cheap operation.
			btm->poll();

			std::vector<command_id> ready_jobs;
			for(auto it = jobs.begin(); it != jobs.end();) {
				auto& job_handle = it->second;

				if(job_handle.unsatisfied_dependencies > 0) {
					++it;
					continue;
				}

				if(!job_handle.job->is_running()) {
					if(std::find(ready_jobs.cbegin(), ready_jobs.cend(), it->first) == ready_jobs.cend()) { ready_jobs.push_back(it->first); }
					++it;
					continue;
				}

				if(!job_handle.job->is_done()) {
					job_handle.job->update();
					++it;
					continue;
				}

				running_jobs.erase(it->first);

				for(const auto& d : job_handle.dependents) {
					assert(jobs.count(d) == 1);
					jobs[d].unsatisfied_dependencies--;
					if(jobs[d].unsatisfied_dependencies == 0) { ready_jobs.push_back(d); }
				}

				if(isa<device_execute_job>(job_handle.job.get())) {
					running_device_compute_jobs--;
				} else if(const auto epoch = dynamic_cast<epoch_job*>(job_handle.job.get()); epoch && epoch->get_epoch_action() == epoch_action::shutdown) {
					assert(command_queue.empty());
					done = true;
				}

				it = jobs.erase(it);
			}

			// Process newly available jobs
			if(!ready_jobs.empty()) {
				// Make sure to start any PUSH jobs before other jobs, as on some platforms copying data from a compute device while
				// also reading it from within a kernel is not supported. To avoid stalling other nodes, we thus perform the PUSH first.
				std::sort(ready_jobs.begin(), ready_jobs.end(),
				    [this](command_id a, command_id b) { return jobs[a].cmd == command_type::PUSH && jobs[b].cmd != command_type::PUSH; });
				for(command_id cid : ready_jobs) {
					auto& job_handle = jobs.at(cid);
					if(std::none_of(job_handle.conflicts.begin(), job_handle.conflicts.end(),
					       [&](const command_id conflict) { return running_jobs.find(conflict) != running_jobs.end(); })) {
						auto* job = job_handle.job.get();
						job->start();
						job->update();
						running_jobs.insert(cid);
						if(isa<device_execute_job>(job)) { running_device_compute_jobs++; }
					}
				}
			}

			MPI_Status status;
			int flag;
			MPI_Message msg;
			MPI_Improbe(MPI_ANY_SOURCE, mpi_support::TAG_CMD, MPI_COMM_WORLD, &flag, &msg, &status);
			if(flag == 1) {
				// Commands should be small enough to block here (TODO: Re-evaluate this now that we also transfer dependencies)
				auto& cmd = command_queue.emplace<command_info>({});
				int num_bytes;
				MPI_Get_count(&status, MPI_CHAR, &num_bytes);
				size_t num_dependencies;
				std::vector<command_id> refs((num_bytes - sizeof(command_pkg)) / sizeof(command_id));
				const auto data_type = mpi_support::build_single_use_composite_type({
				    {sizeof(command_pkg), &cmd.pkg},                //
				    {sizeof(size_t), &num_dependencies},            //
				    {refs.size() * sizeof(command_id), refs.data()} //
				});
				MPI_Mrecv(MPI_BOTTOM, 1, *data_type, &msg, &status);

				assert(num_dependencies <= refs.size());
				cmd.conflicts = std::vector<command_id>(refs.begin() + static_cast<std::vector<command_id>::difference_type>(num_dependencies), refs.end());
				refs.resize(num_dependencies);
				cmd.dependencies = std::move(refs);

				if(!first_command_received) {
					metrics.initial_idle.pause();
					metrics.device_idle.resume();
					first_command_received = true;
				}
			}

			if(jobs.size() < MAX_CONCURRENT_JOBS && !command_queue.empty()) {
				const auto& cmd = command_queue.front();
				if(!handle_command(cmd)) {
					// In case the command couldn't be handled, don't pop it from the queue.
					continue;
				}
				command_queue.pop();
			}

			if(first_command_received) { update_metrics(); }
		}

		assert(running_device_compute_jobs == 0);
	}

	bool executor::handle_command(const command_info& cmd) {
		const auto& pkg = cmd.pkg;

		// A worker might receive a task command before creating the corresponding horizon task itself
		if(pkg.tid && !task_mngr.has_task(*pkg.tid)) { return false; }

		switch(pkg.cmd) {
		case command_type::HORIZON:
			assert(pkg.tid.has_value());
			create_job<horizon_job>(cmd, task_mngr);
			break;
		case command_type::EPOCH:
			assert(pkg.tid.has_value());
			create_job<epoch_job>(cmd, task_mngr);
			break;
		case command_type::PUSH: create_job<push_job>(cmd, *btm, buffer_mngr); break;
		case command_type::AWAIT_PUSH: create_job<await_push_job>(cmd, *btm); break;
		case command_type::REDUCTION: create_job<reduction_job>(cmd, reduction_mngr); break;
		case command_type::EXECUTION:
			assert(pkg.tid.has_value());
			if(task_mngr.get_task(*pkg.tid)->get_execution_target() == execution_target::HOST) {
				create_job<host_execute_job>(cmd, h_queue, task_mngr, buffer_mngr);
			} else {
				create_job<device_execute_job>(cmd, d_queue, task_mngr, buffer_mngr, reduction_mngr, local_nid);
			}
			break;
		default: assert(!"Unexpected command");
		}
		return true;
	}

	void executor::update_metrics() {
		if(running_device_compute_jobs == 0) {
			if(!metrics.device_idle.is_running()) { metrics.device_idle.resume(); }
		} else {
			if(metrics.device_idle.is_running()) { metrics.device_idle.pause(); }
		}
		if(jobs.empty()) {
			if(!metrics.starvation.is_running()) { metrics.starvation.resume(); }
		} else {
			if(metrics.starvation.is_running()) { metrics.starvation.pause(); }
		}
	}
} // namespace detail
} // namespace celerity
