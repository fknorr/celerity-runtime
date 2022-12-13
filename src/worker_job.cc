#include "worker_job.h"

#include <spdlog/fmt/fmt.h>

#include "buffer_manager.h"
#include "closure_hydrator.h"
#include "device_queue.h"
#include "handler.h"
#include "reduction_manager.h"
#include "runtime.h"
#include "task_manager.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- GENERAL ------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	void worker_job::update() {
		CELERITY_LOG_SET_SCOPED_CTX(m_lctx);
		assert(m_running && !m_done);
		m_tracy_lane.activate();
		const auto before = std::chrono::steady_clock::now();
		m_done = execute(m_pkg);
		m_tracy_lane.deactivate();

		// TODO: We may want to make benchmarking optional with a macro
		const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - before);
		m_bench_sum_execution_time += dt;
		m_bench_sample_count++;
		if(dt < m_bench_min) m_bench_min = dt;
		if(dt > m_bench_max) m_bench_max = dt;

		if(m_done) {
			const auto bench_avg = m_bench_sum_execution_time.count() / m_bench_sample_count;
			const auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start_time).count();
			CELERITY_DEBUG("Job finished after {}us. Polling avg={}, min={}, max={}, samples={}", execution_time, bench_avg, m_bench_min.count(),
			    m_bench_max.count(), m_bench_sample_count);

			m_tracy_lane.destroy();
		}
	}

	void worker_job::start() {
		CELERITY_LOG_SET_SCOPED_CTX(m_lctx);
		assert(!m_running);
		m_running = true;

		const auto desc = fmt::format("cid={}: {}", m_pkg.cid, get_description(m_pkg));
		m_tracy_lane.initialize();
		switch(m_pkg.get_command_type()) {
		case command_type::execution: m_tracy_lane.begin_phase("execution", desc, tracy::Color::ColorType::Blue); break;
		case command_type::push: m_tracy_lane.begin_phase("push", desc, tracy::Color::ColorType::Red); break;
		case command_type::await_push: m_tracy_lane.begin_phase("await push", desc, tracy::Color::ColorType::Yellow); break;
		default: m_tracy_lane.begin_phase("other", desc, tracy::Color::ColorType::Gray); break;
		}

		CELERITY_DEBUG("Starting job: {}", desc);
		m_start_time = std::chrono::steady_clock::now();
	}

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- HORIZON --------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string horizon_job::get_description(const command_pkg& pkg) { return "horizon"; }

	bool horizon_job::execute(const command_pkg& pkg) {
		const auto data = std::get<horizon_data>(pkg.data);
		m_task_mngr.notify_horizon_reached(data.tid);
		return true;
	};

	// --------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------- EPOCH ---------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string epoch_job::get_description(const command_pkg& pkg) { return "epoch"; }

	bool epoch_job::execute(const command_pkg& pkg) {
		const auto data = std::get<epoch_data>(pkg.data);
		m_action = data.action;

		// This barrier currently enables profiling Celerity programs on a cluster by issuing a queue.slow_full_sync() and
		// then observing the execution times of barriers. TODO remove this once we have a better profiling workflow.
		if(m_action == epoch_action::barrier) { MPI_Barrier(MPI_COMM_WORLD); }

		m_task_mngr.notify_epoch_reached(data.tid);
		return true;
	};

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- AWAIT PUSH -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string await_push_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<await_push_data>(pkg.data);
		return fmt::format("await push of buffer {} transfer {}", static_cast<size_t>(data.bid), static_cast<size_t>(data.trid));
	}

	bool await_push_job::execute(const command_pkg& pkg) {
		if(m_data_handle == nullptr) {
			const auto data = std::get<await_push_data>(pkg.data);
			GridRegion<3> expected_region;
			for(size_t i = 0; i < data.num_subranges; ++i) {
				expected_region = GridRegion<3>::merge(expected_region, subrange_to_grid_box(data.region[i]));
			}
			m_data_handle = m_btm.await_push(data.trid, data.bid, expected_region, data.rid);
		}
		return m_data_handle->complete;
	}


	// --------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------- PUSH -------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string push_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<push_data>(pkg.data);
		return fmt::format("push {} of buffer {} transfer {} to node {}", data.sr, static_cast<size_t>(data.bid), static_cast<size_t>(data.trid),
		    static_cast<size_t>(data.target));
	}

	bool push_job::execute(const command_pkg& pkg) {
		if(m_data_handle == nullptr) {
			const auto data = std::get<push_data>(pkg.data);
			// Getting buffer data from the buffer manager may incur a host-side buffer reallocation.
			// If any other tasks are currently using this buffer for reading, we run into problems.
			// To avoid this, we use a very crude buffer locking mechanism for now.
			// FIXME: Get rid of this, replace with finer grained approach.
			if(m_buffer_mngr.is_locked(data.bid, 0 /* FIXME: Host memory id - should use host_queue::get_memory_id */)) { return false; }

			CELERITY_TRACE("Submit buffer to BTM");
			m_data_handle = m_btm.push(data.target, data.trid, data.bid, data.sr, data.rid);
			CELERITY_TRACE("Buffer submitted to BTM");
		}

		return m_data_handle->complete;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- DATA REQUEST ---------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string data_request_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<data_request_data>(pkg.data);
		return fmt::format("request {} of buffer {} from node {}", data.sr, static_cast<size_t>(data.bid), static_cast<size_t>(data.source));
	}

	bool data_request_job::execute(const command_pkg& pkg) { return true; }

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- REDUCTION ----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	bool reduction_job::execute(const command_pkg& pkg) {
		const auto& data = std::get<reduction_data>(pkg.data);
		m_rm.finish_reduction(data.rid);
		return true;
	}

	std::string reduction_job::get_description(const command_pkg& pkg) { return "reduction"; }

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- HOST_EXECUTE ---------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string host_execute_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<execution_data>(pkg.data);
		return fmt::format("HOST_EXECUTE {}", data.sr);
	}

	bool host_execute_job::execute(const command_pkg& pkg) {
		const buffer_manager::buffer_lock_id lock_id = pkg.cid * buffer_manager::max_memories;

		if(!m_submitted) {
			const auto data = std::get<execution_data>(pkg.data);

			auto tsk = m_task_mngr.get_task(data.tid);
			assert(tsk->get_execution_target() == execution_target::host);
			assert(!data.initialize_reductions); // For now, we do not support reductions in host tasks

			if(!m_buffer_mngr.try_lock(lock_id, m_queue.get_memory_id(), tsk->get_buffer_access_map().get_accessed_buffers())) { return false; }

			CELERITY_TRACE("Execute live-pass, scheduling host task in thread pool");

			// NOCOMMIT DRY with device execute?
			const auto& access_map = tsk->get_buffer_access_map();
			std::vector<closure_hydrator::NOCOMMIT_info> access_infos;
			access_infos.reserve(access_map.get_num_accesses());
			for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
				const auto [bid, mode] = access_map.get_nth_access(i);
				const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk->get_dimensions(), data.sr, tsk->get_global_size()));
				const auto info = m_buffer_mngr.begin_host_buffer_access(bid, mode, sr.range, sr.offset);
				access_infos.push_back(closure_hydrator::NOCOMMIT_info{target::host_task, info.ptr, info.backing_buffer_range, info.backing_buffer_offset, sr});
			}

			closure_hydrator::get_instance().prepare(std::move(access_infos));
			// NOCOMMIT TODO: There should be no need to pass cgid or global size from here, store inside launcher.
			m_future = tsk->launch(m_queue, tsk->get_collective_group_id(), tsk->get_global_size(), data.sr);
			assert(m_future.valid());

			for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
				const auto [bid, mode] = access_map.get_nth_access(i);
				const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk->get_dimensions(), data.sr, tsk->get_global_size()));
				m_buffer_mngr.end_buffer_access(buffer_manager::data_location{}.set(m_buffer_mngr.get_host_memory_id()), bid, mode, sr.range, sr.offset);
			}

			m_submitted = true;
			CELERITY_TRACE("Submitted host task to thread pool");
		}

		assert(m_future.valid());
		if(m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			m_buffer_mngr.unlock(lock_id);

			auto info = m_future.get();
			CELERITY_TRACE("Delta time submit -> start: {}us, start -> end: {}us",
			    std::chrono::duration_cast<std::chrono::microseconds>(info.start_time - info.submit_time).count(),
			    std::chrono::duration_cast<std::chrono::microseconds>(info.end_time - info.start_time).count());
			return true;
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------- DEVICE_EXECUTE ------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	device_execute_job::device_execute_job(
	    command_pkg pkg, local_devices& devices, task_manager& tm, buffer_manager& bm, reduction_manager& rm, node_id local_nid)
	    : worker_job(pkg), m_local_devices(devices), m_task_mngr(tm), m_buffer_mngr(bm), m_reduction_mngr(rm), m_local_nid(local_nid) {
		assert(pkg.get_command_type() == command_type::execution);

		m_device_ids = utils::match(
		    std::get<execution_data>(pkg.data).devices, [](on_single_device sd) { return std::vector{sd.device}; },
		    [](replicated_on_devices rd) {
			    std::vector<device_id> dids(rd.num_devices);
			    std::iota(dids.begin(), dids.end(), device_id());
			    return dids;
		    });
	}

	std::string device_execute_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<execution_data>(pkg.data);
		auto tsk = m_task_mngr.get_task(data.tid);
		return fmt::format("DEVICE_EXECUTE task {} ('{}') {} on device(s) {}", tsk->get_id(), tsk->get_debug_name(), data.sr, fmt::join(m_device_ids, ", "));
	}

	bool device_execute_job::execute(const command_pkg& pkg) {
		if(!m_submitted) {
			const auto data = std::get<execution_data>(pkg.data);
			auto tsk = m_task_mngr.get_task(data.tid);
			assert(tsk->get_execution_target() == execution_target::device);

			// lock buffer on all device memories, but (very naively) avoid deadlocks
			for(size_t i = 0; i < m_device_ids.size(); ++i) {
				const auto mid = m_local_devices.get_memory_id(m_device_ids[i]);
				const buffer_manager::buffer_lock_id lock_id = pkg.cid * buffer_manager::max_memories + mid;
				if(!m_buffer_mngr.try_lock(lock_id, mid, tsk->get_buffer_access_map().get_accessed_buffers())) {
					while(i-- > 0) {
						const auto unlock_mid = m_local_devices.get_memory_id(m_device_ids[i]);
						const buffer_manager::buffer_lock_id unlock_lock_id = pkg.cid * buffer_manager::max_memories + unlock_mid;
						m_buffer_mngr.unlock(unlock_lock_id);
					}
					return false;
				}
			}

			CELERITY_TRACE("Execute live-pass, submit kernel to SYCL");

			const auto& access_map = tsk->get_buffer_access_map();
			for(auto did : m_device_ids) {
				std::vector<closure_hydrator::NOCOMMIT_info> access_infos;
				access_infos.reserve(access_map.get_num_accesses());
				for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
					const auto [bid, mode] = access_map.get_nth_access(i);
					const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk->get_dimensions(), data.sr, tsk->get_global_size()));
					const auto info = m_buffer_mngr.begin_device_buffer_access(m_local_devices.get_memory_id(did), bid, mode, sr.range, sr.offset);
					access_infos.push_back(
					    closure_hydrator::NOCOMMIT_info{target::device, info.ptr, info.backing_buffer_range, info.backing_buffer_offset, sr});
				}

				closure_hydrator::get_instance().prepare(std::move(access_infos));
				m_events.push_back(tsk->launch(m_local_devices.get_device_queue(did), data.sr));
			}

			{
				const auto msg = fmt::format("{}: Job submitted to SYCL (blocked on transfers until now!)", pkg.cid);
				TracyMessage(msg.c_str(), msg.size());
			}

			buffer_manager::data_location mids;
			for(auto did : m_device_ids) {
				mids.set(m_local_devices.get_memory_id(did));
			}
			for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
				const auto [bid, mode] = access_map.get_nth_access(i);
				const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk->get_dimensions(), data.sr, tsk->get_global_size()));
				m_buffer_mngr.end_buffer_access(mids, bid, mode, sr.range, sr.offset);
			}

			m_submitted = true;
			CELERITY_TRACE("Kernel submitted to SYCL");
		}

		if(std::all_of(m_events.begin(), m_events.end(), [](const sycl::event& evt) {
			   return evt.get_info<cl::sycl::info::event::command_execution_status>() == cl::sycl::info::event_command_status::complete;
		   })) {
			for(size_t i = 0; i < m_device_ids.size(); ++i) {
				const auto mid = m_local_devices.get_memory_id(m_device_ids[i]);
				const buffer_manager::buffer_lock_id lock_id = pkg.cid * buffer_manager::max_memories + mid;
				m_buffer_mngr.unlock(lock_id);
			}

			const auto data = std::get<execution_data>(pkg.data);
			auto tsk = m_task_mngr.get_task(data.tid);
			for(const auto& reduction : tsk->get_reductions()) {
				const auto element_size = m_buffer_mngr.get_buffer_info(reduction.bid).element_size;
				auto operand = make_uninitialized_payload<std::byte>(element_size);
				m_buffer_mngr.get_buffer_data(reduction.bid, {{}, {1, 1, 1}}, operand.get_pointer());
				m_reduction_mngr.push_overlapping_reduction_data(reduction.rid, m_local_nid, std::move(operand));
			}

			for(size_t i = 0; i < m_device_ids.size(); ++i) {
				if(m_local_devices.get_device_queue(m_device_ids[i]).is_profiling_enabled()) {
					const auto submit = std::chrono::nanoseconds(m_events[i].get_profiling_info<cl::sycl::info::event_profiling::command_submit>());
					const auto start = std::chrono::nanoseconds(m_events[i].get_profiling_info<cl::sycl::info::event_profiling::command_start>());
					const auto end = std::chrono::nanoseconds(m_events[i].get_profiling_info<cl::sycl::info::event_profiling::command_end>());

					CELERITY_TRACE("Delta time submit -> start: {}us, start -> end: {}us",
					    std::chrono::duration_cast<std::chrono::microseconds>(start - submit).count(),
					    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
				}
			}
			return true;
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// -------------------------------------------------------- FENCE -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::string fence_job::get_description(const command_pkg& pkg) { return fmt::format("FENCE"); }

	bool fence_job::execute(const command_pkg& pkg) {
		const auto data = std::get<fence_data>(pkg.data);
		const auto tsk = m_task_mngr.get_task(data.tid);
		const auto promise = tsk->get_fence_promise();
		assert(promise != nullptr);

		promise->fulfill();

		return true;
	}

} // namespace detail
} // namespace celerity
