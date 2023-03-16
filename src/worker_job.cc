#include <numeric>

#include "worker_job.h"

#include <spdlog/fmt/fmt.h>

#include "buffer_manager.h"
#include "closure_hydrator.h"
#include "device_queue.h"
#include "handler.h"
#include "log.h"
#include "reduction_manager.h"
#include "runtime.h"
#include "task_manager.h"
#include "workaround.h"

#include "print_utils.h"

namespace celerity {
namespace detail {

	bool NOMERGE_skip_device_kernel_execution = false;

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- GENERAL ------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	bool worker_job::prepare() {
		CELERITY_LOG_SET_SCOPED_CTX(m_lctx);

		if(!m_tracy_lane.is_initialized()) {
			m_tracy_lane.initialize();
			const auto desc = fmt::format("cid={}: {}", m_pkg.cid, get_description(m_pkg));
			m_tracy_lane.begin_phase("preparation", desc, tracy::Color::ColorType::Pink);
			CELERITY_DEBUG("Preparing job: {}", desc);
		}

		m_tracy_lane.activate();
		const auto result = prepare(m_pkg);
		m_tracy_lane.deactivate();
		return result;
	}

	void worker_job::update() noexcept {
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
		switch(m_pkg.get_command_type()) {
		case command_type::execution: m_tracy_lane.begin_phase("execution", desc, tracy::Color::ColorType::Blue); break;
		case command_type::push: m_tracy_lane.begin_phase("push", desc, tracy::Color::ColorType::Red); break;
		case command_type::await_push: m_tracy_lane.begin_phase("await push", desc, tracy::Color::ColorType::Yellow); break;
		case command_type::broadcast: m_tracy_lane.begin_phase("broadcast", desc, tracy::Color::ColorType::Purple); break;
		case command_type::scatter: m_tracy_lane.begin_phase("scatter", desc, tracy::Color::ColorType::Purple); break;
		case command_type::gather: m_tracy_lane.begin_phase("gather", desc, tracy::Color::ColorType::Purple); break;
		case command_type::allgather: m_tracy_lane.begin_phase("allgather", desc, tracy::Color::ColorType::Purple); break;
		case command_type::alltoall: m_tracy_lane.begin_phase("alltoall", desc, tracy::Color::ColorType::Purple); break;
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

	bool push_job::prepare(const command_pkg& pkg) {
		if(m_frame.get_pointer() == nullptr) {
			ZoneScopedN("push_job::prepare");
			const auto data = std::get<push_data>(pkg.data);
			// Getting buffer data from the buffer manager may incur a host-side buffer reallocation.
			// If any other tasks are currently using this buffer for reading, we run into problems.
			// To avoid this, we use a very crude buffer locking mechanism for now.
			// FIXME: Get rid of this, replace with finer grained approach.
			if(m_buffer_mngr.is_locked(data.bid, 0 /* FIXME: Host memory id - should use host_queue::get_memory_id */)) { return false; }

			const auto element_size = m_buffer_mngr.get_buffer_info(data.bid).element_size;
			unique_frame_ptr<buffer_transfer_manager::data_frame> frame(from_payload_count, data.sr.range.size() * element_size);
			frame->sr = data.sr;
			frame->bid = data.bid;
			frame->rid = data.rid;
			frame->trid = data.trid;
			m_frame_transfer_event = m_buffer_mngr.get_buffer_data(data.bid, data.sr, frame->data);
			m_frame = std::move(frame);
		}
		return m_frame_transfer_event.is_done();
	}

	bool push_job::execute(const command_pkg& pkg) {
		const auto data = std::get<push_data>(pkg.data);
		if(m_data_handle == nullptr) {
			assert(m_frame_transfer_event.is_done());
			CELERITY_TRACE("Submit buffer to BTM");
			m_data_handle = m_btm.push(data.target, std::move(m_frame));
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
		if(!m_submitted) {
			const auto data = std::get<execution_data>(pkg.data);

			const auto& tsk = dynamic_cast<const command_group_task&>(*m_task_mngr.get_task(data.tid));
			assert(tsk.get_execution_target() == execution_target::host);
			assert(!data.initialize_reductions); // For now, we do not support reductions in host tasks

			if(!m_buffer_mngr.try_lock(pkg.cid, m_queue.get_memory_id(), tsk.get_buffer_access_map().get_accessed_buffers())) { return false; }

			CELERITY_TRACE("Execute live-pass, scheduling host task in thread pool");

			// NOCOMMIT DRY with device execute?
			const auto& access_map = tsk.get_buffer_access_map();
			std::vector<closure_hydrator::NOCOMMIT_info> access_infos;
			access_infos.reserve(access_map.get_num_accesses());
			for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
				const auto [bid, mode] = access_map.get_nth_access(i);
				const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk.get_dimensions(), data.sr, tsk.get_global_size()));
				const auto info = m_buffer_mngr.access_host_buffer(bid, mode, sr.range, sr.offset);
				access_infos.push_back(closure_hydrator::NOCOMMIT_info{target::host_task, info.ptr, info.backing_buffer_range, info.backing_buffer_offset, sr});
			}

			closure_hydrator::get_instance().prepare(std::move(access_infos));
			// NOCOMMIT TODO: There should be no need to pass cgid or global size from here, store inside launcher.
			m_future = tsk.launch(m_queue, tsk.get_collective_group_id(), tsk.get_global_size(), data.sr);

			assert(m_future.valid());
			m_submitted = true;
			CELERITY_TRACE("Submitted host task to thread pool");
		}

		assert(m_future.valid());
		if(m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			m_buffer_mngr.unlock(pkg.cid);

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

	std::string device_execute_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<execution_data>(pkg.data);
		const auto& tsk = dynamic_cast<const command_group_task&>(*m_task_mngr.get_task(data.tid));
		return fmt::format("DEVICE_EXECUTE task {} ('{}') {} on device {}", tsk.get_id(), tsk.get_debug_name(), data.sr, m_queue.get_id());
	}

	bool device_execute_job::prepare(const command_pkg& pkg) {
		if(m_async_transfers_done) return true;

		// NOCOMMIT TODO This is not a good test b/c it wouldn't work for kernels without any accessors
		if(m_access_infos.empty()) {
			const auto data = std::get<execution_data>(pkg.data);
			const auto& tsk = dynamic_cast<const command_group_task&>(*m_task_mngr.get_task(data.tid));
			assert(tsk.get_execution_target() == execution_target::device);

			if(!m_buffer_mngr.try_lock(pkg.cid, m_queue.get_memory_id(), tsk.get_buffer_access_map().get_accessed_buffers())) { return false; }

			const auto& access_map = tsk.get_buffer_access_map();
			{
				const auto msg = fmt::format("Preparing buffers for {} accesses", access_map.get_num_accesses());
				TracyMessage(msg.c_str(), msg.size());
				CELERITY_TRACE(msg);
			}
			m_access_infos.reserve(access_map.get_num_accesses());
			for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
				const auto [bid, mode] = access_map.get_nth_access(i);
				const auto sr = grid_box_to_subrange(access_map.get_requirements_for_nth_access(i, tsk.get_dimensions(), data.sr, tsk.get_global_size()));
				try {
					auto info = m_buffer_mngr.access_device_buffer(m_queue.get_memory_id(), bid, mode, sr.range, sr.offset);
					m_access_infos.push_back(
					    closure_hydrator::NOCOMMIT_info{target::device, info.ptr, info.backing_buffer_range, info.backing_buffer_offset, sr});
					m_access_transfer_events.emplace_back(std::move(info.pending_transfers));
				} catch(allocation_error e) {
					CELERITY_CRITICAL("Encountered allocation error while trying to prepare {}", get_description(pkg));
					std::terminate();
				}
			}
		}

		if(!m_async_transfers_done
		    && std::all_of(m_access_transfer_events.cbegin(), m_access_transfer_events.cend(), [](auto& evt) { return evt.is_done(); })) {
			m_async_transfers_done = true;
			const auto msg = fmt::format("{}: Async transfers done", pkg.cid);
			TracyMessage(msg.c_str(), msg.size());
			return true;
		}

		return false;
	}

	bool device_execute_job::execute(const command_pkg& pkg) {
		if(NOMERGE_skip_device_kernel_execution) {
			m_buffer_mngr.unlock(pkg.cid);
			return true;
		}

		if(!m_submitted) {
			const auto data = std::get<execution_data>(pkg.data);
			const auto& tsk = dynamic_cast<const command_group_task&>(*m_task_mngr.get_task(data.tid));
			closure_hydrator::get_instance().prepare(std::move(m_access_infos));
			CELERITY_TRACE("Execute live-pass, submit kernel to SYCL");
			m_event = tsk.launch(m_queue, data.sr);
			m_submitted = true;
			CELERITY_TRACE("Kernel submitted to SYCL");
		}

		const auto status = m_event.get_info<cl::sycl::info::event::command_execution_status>();
		if(status == cl::sycl::info::event_command_status::complete) {
			m_buffer_mngr.unlock(pkg.cid);

			const auto data = std::get<execution_data>(pkg.data);
			const auto& tsk = dynamic_cast<const command_group_task&>(*m_task_mngr.get_task(data.tid));
			for(const auto& reduction : tsk.get_reductions()) {
				const auto element_size = m_buffer_mngr.get_buffer_info(reduction.bid).element_size;
				auto operand = make_uninitialized_payload<std::byte>(element_size);
				const auto evt = m_buffer_mngr.get_buffer_data(reduction.bid, {{}, {1, 1, 1}}, operand.get_pointer());
				evt.wait();
				m_reduction_mngr.push_overlapping_reduction_data(reduction.rid, m_local_nid, std::move(operand));
			}

			if(m_queue.is_profiling_enabled()) {
				const auto submit = std::chrono::nanoseconds(m_event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>());
				const auto start = std::chrono::nanoseconds(m_event.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
				const auto end = std::chrono::nanoseconds(m_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>());

				CELERITY_TRACE("Delta time submit -> start: {}us, start -> end: {}us",
				    std::chrono::duration_cast<std::chrono::microseconds>(start - submit).count(),
				    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
			}
			return true;
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------ GATHER ------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	void fetch_collective_input(buffer_manager& bm, const buffer_id bid, collective_buffer& buffer) {
		ZoneScopedN("fetch input");

		async_event fetch_complete;
		for(auto& [box, offset_bytes, size_bytes] : buffer.get_fetch_boxes()) {
			fetch_complete.merge(bm.get_buffer_data(bid, grid_box_to_subrange(box), buffer.get_payload(offset_bytes)));
		}
		fetch_complete.wait();
	}

	void commit_collective_update(buffer_manager& bm, const buffer_id bid, collective_buffer&& buffer) {
		ZoneScopedN("commit");

		assert(buffer.broadcast_covers_all_updates() || buffer.get_broadcast_boxes().empty()); // if not we definitely messed up region_spec::update/broadcast

		bool used_broadcast = false;
		if(buffer.broadcast_covers_all_updates()) {
			bool lock_success = true;
			auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			auto num_devices = local_devices.num_compute_devices();
			std::vector<bool> mem_locked(num_devices, false);
			for(size_t device_id = 0; device_id < num_devices; ++device_id) {
				auto mem_id = local_devices.get_memory_id(device_id);
				if(bm.try_lock(device_id, mem_id, {bid})) {
					mem_locked[device_id] = true;
				} else {
					lock_success = false;
					CELERITY_TRACE("device broadcast lock unsuccessful");
					break;
				}
			}
			if(lock_success) {
				bool is_broadcast_possible = true;
				for(const auto& [box, offset_bytes, size_bytes] : buffer.get_broadcast_boxes()) {
					const auto sr = grid_box_to_subrange(box);
					is_broadcast_possible &= bm.is_broadcast_possible(bid, sr);
				}
				if(is_broadcast_possible) {
					for(auto& [box, offset_bytes, size_bytes] : buffer.get_broadcast_boxes()) {
						const auto sr = grid_box_to_subrange(box);
						CELERITY_TRACE("immediately_broadcast_data({}, {}, get_payload(offset_bytes))", (int)bid, sr);
						bm.immediately_broadcast_data(bid, sr, buffer.get_payload(offset_bytes));
					}
					used_broadcast = true;
				} else {
					CELERITY_TRACE("device broadcast not possible");
				}
			}
			for(size_t device_id = 0; device_id < num_devices; ++device_id) {
				if(mem_locked[device_id]) { bm.unlock(device_id); }
			}
		}

		if(!used_broadcast) {
			const shared_payload_ptr payload = buffer.take_payload();
			for(auto& [box, offset_bytes, size_bytes] : buffer.get_update_boxes()) {
				const auto sr = grid_box_to_subrange(box);
				CELERITY_TRACE("non-broadcast set_buffer_data({}, {}, payload+{})", (int)bid, sr, offset_bytes);
				bm.set_buffer_data(bid, sr, shared_payload_ptr(payload, offset_bytes));
			}
		}
	}

	static const bool NOMERGE_disable_eager_coherence_updates = getenv("CELERITY_NO_EAGER_COHERENCE_UPDATES") != nullptr;

	async_event make_buffer_region_coherent_on_device(buffer_manager& bm, const buffer_id bid, const GridRegion<3>& region, const device_id did) {
		if(NOMERGE_disable_eager_coherence_updates) return {};

		const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
		const auto mid = local_devices.get_memory_id(did);
		const buffer_manager::buffer_lock_id lid = 1000000 + bid; // FIXME?
		if(!bm.try_lock(lid, mid, {bid})) return {};              // skip eager coherence update if locking the buffer fails

		async_event pending_transfers;
		region.scanByBoxes([&](const GridBox<3>& box) {
			const auto sr = grid_box_to_subrange(box);
			auto info = bm.access_device_buffer(mid, bid, access_mode::read, sr.range, sr.offset); // triggers make_buffer_subrange_coherent
			pending_transfers.merge(std::move(info.pending_transfers));
		});

		bm.unlock(lid);
		return pending_transfers;
	}

	gather_job::gather_job(const command_pkg& pkg, buffer_manager& bm, node_id nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid) {
		assert(std::holds_alternative<gather_data>(pkg.data));
	}

	std::string gather_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<gather_data>(pkg.data);
		return fmt::format("gather {} of buffer {} to {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid), data.root);
	}

	bool gather_job::execute(const command_pkg& pkg) {
		const auto data = std::get<gather_data>(pkg.data);
		const bool receives_data = m_local_nid == data.root;
		const bool sends_data = !receives_data;

		if(!m_started) {
			ZoneScoped;
			assert(data.source_regions[data.root].empty());

			const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

			if(sends_data) {
				collective_buffer::region_spec region_spec;
				region_spec.region = data.source_regions[m_local_nid];
				region_spec.fetch = true;
				region_spec.update = false;
				region_spec.broadcast = false;
				m_send_buffer = collective_buffer({region_spec}, buffer_info.element_size);
				fetch_collective_input(m_buffer_mngr, data.bid, m_send_buffer);
			}

			if(receives_data) {
				std::vector<collective_buffer::region_spec> collective_regions(data.source_regions.size());
				for(node_id nid = 0; nid < data.source_regions.size(); ++nid) {
					auto& r = collective_regions[nid];
					r.region = data.source_regions[nid];
					r.fetch = false;
					r.update = true; // usually consumed only on host or one device
					r.broadcast = false;
				}
				m_recv_buffer = collective_buffer(collective_regions, buffer_info.element_size);
			}

			{
#if TRACY_ENABLE
				ZoneScopedN("Igatherv");
				const auto req_msg =
				    fmt::format("B{}, {} -> {} bytes", data.bid, m_send_buffer.get_payload_size_bytes(), m_recv_buffer.get_payload_size_bytes());
				ZoneText(req_msg.data(), req_msg.size());
#endif
				CELERITY_TRACE("MPI_Igatherv([buffer of size {}], {}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, {}, MPI_COMM_WORLD, &request)",
				    m_send_buffer.get_payload_size_bytes(), m_send_buffer.get_payload_size_bytes(), m_recv_buffer.get_payload_size_bytes(),
				    fmt::join(m_recv_buffer.get_chunk_byte_sizes(), ", "), fmt::join(m_recv_buffer.get_chunk_byte_offsets(), ", "), data.root);
				MPI_Igatherv(m_send_buffer.get_payload(), m_send_buffer.get_payload_size_bytes(), MPI_BYTE, m_recv_buffer.get_payload(),
				    m_recv_buffer.get_chunk_byte_sizes().data(), m_recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, static_cast<int>(data.root),
				    MPI_COMM_WORLD, &m_request);
			}

			// Overlap async MPI_Igatherv with a gather to device 0
			// TODO this is suboptimal if the gathered data is to be used in a (master-node) host task instead of an unsplittable device task.
			const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			if(!data.local_coherence_region.empty() && local_devices.num_compute_devices() > 1) {
				ZoneScopedN("d2d gather");
				m_d2d_gather = make_buffer_region_coherent_on_device(m_buffer_mngr, data.bid, data.local_coherence_region, device_id(0));
			}

			m_started = true;
		}

		if(m_request != MPI_REQUEST_NULL) {
			int mpi_done;
			MPI_Test(&m_request, &mpi_done, MPI_STATUS_IGNORE);
			if(mpi_done) {
#if TRACY_ENABLE
				const std::string_view msg_done = "Igatherv complete";
				TracyMessage(msg_done.data(), msg_done.size());
#endif
				m_request = MPI_REQUEST_NULL;
			}
		}

		if(m_d2d_gather.has_value() && m_d2d_gather->is_done()) {
#if TRACY_ENABLE
			const std::string_view msg_done = "d2d gather complete";
			TracyMessage(msg_done.data(), msg_done.size());
#endif
			m_d2d_gather.reset();
		}

		if(m_request == MPI_REQUEST_NULL && !m_d2d_gather.has_value()) {
			if(receives_data) commit_collective_update(m_buffer_mngr, data.bid, std::move(m_recv_buffer));
			return true;
		} else {
			return false;
		}
	}

	allgather_job::allgather_job(const command_pkg& pkg, buffer_manager& bm, node_id nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid) {
		assert(std::holds_alternative<allgather_data>(pkg.data));
	}

	std::string allgather_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<allgather_data>(pkg.data);
		return fmt::format("allgather {} of buffer {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid));
	}


	bool allgather_job::execute(const command_pkg& pkg) {
		const auto data = std::get<allgather_data>(pkg.data);

		if(!m_started) {
			ZoneScoped;
			const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

			std::vector<collective_buffer::region_spec> collective_regions(data.source_regions.size());
			for(node_id nid = 0; nid < data.source_regions.size(); ++nid) {
				auto& r = collective_regions[nid];
				r.region = data.source_regions[nid];
				r.fetch = nid == m_local_nid;
				r.update = nid != m_local_nid || NOMERGE_disable_eager_coherence_updates;
				r.broadcast = nid != m_local_nid || NOMERGE_disable_eager_coherence_updates;
			}

			m_buffer = collective_buffer(collective_regions, buffer_info.element_size);

			fetch_collective_input(m_buffer_mngr, data.bid, m_buffer);

			{
#if TRACY_ENABLE
				ZoneScopedN("Iallgatherv");
				const auto req_msg = fmt::format("B{}, {} bytes", data.bid, m_buffer.get_payload_size_bytes());
				ZoneText(req_msg.data(), req_msg.size());
#endif
				CELERITY_TRACE("MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD, &request)",
				    m_buffer.get_payload_size_bytes(), fmt::join(m_buffer.get_chunk_byte_sizes(), ", "), fmt::join(m_buffer.get_chunk_byte_offsets(), ", "));
				MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_BYTE, m_buffer.get_payload(), m_buffer.get_chunk_byte_sizes().data(),
				    m_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, MPI_COMM_WORLD, &m_request);
			}

			// Overlap async MPI_Iallgatherv with an allgather between local devices
			const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			if(!data.local_coherence_region.empty() && local_devices.num_compute_devices() > 1) {
				ZoneScopedN("d2d allgather");
				async_event pending_transfers;
				for(device_id did = 0; did < local_devices.num_compute_devices(); ++did) {
					pending_transfers.merge(make_buffer_region_coherent_on_device(m_buffer_mngr, data.bid, data.local_coherence_region, did));
				}
				m_d2d_allgather = std::move(pending_transfers);
			}

			m_started = true;
		}

		if(m_request != MPI_REQUEST_NULL) {
			int mpi_done;
			MPI_Test(&m_request, &mpi_done, MPI_STATUS_IGNORE);
			if(mpi_done) {
#if TRACY_ENABLE
				const std::string_view msg_done = "Iallgatherv complete";
				TracyMessage(msg_done.data(), msg_done.size());
#endif
				m_request = MPI_REQUEST_NULL;
			}
		}

		if(m_d2d_allgather.has_value() && m_d2d_allgather->is_done()) {
#if TRACY_ENABLE
			const std::string_view msg_done = "d2d allgather complete";
			TracyMessage(msg_done.data(), msg_done.size());
#endif
			m_d2d_allgather.reset();
		}

		if(m_request == MPI_REQUEST_NULL && !m_d2d_allgather.has_value()) {
			commit_collective_update(m_buffer_mngr, data.bid, std::move(m_buffer));
			return true;
		} else {
			return false;
		}
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------- BROADCAST -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	broadcast_job::broadcast_job(const command_pkg& pkg, buffer_manager& bm, node_id nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid) {
		assert(std::holds_alternative<broadcast_data>(pkg.data));
	}

	std::string broadcast_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<broadcast_data>(pkg.data);
		return fmt::format("broadcast {} of buffer {} from {}", data.region, static_cast<size_t>(data.bid), data.root);
	}

	bool broadcast_job::execute(const command_pkg& pkg) {
		const auto data = std::get<broadcast_data>(pkg.data);
		const bool sends_data = data.root == m_local_nid;
		const bool receives_data = !sends_data;

		if(!m_started) {
			const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

			collective_buffer::region_spec region_spec;
			region_spec.region = data.region;
			region_spec.fetch = sends_data;
			region_spec.update = region_spec.broadcast = receives_data;

			m_buffer = collective_buffer({std::move(region_spec)}, buffer_info.element_size);

			if(sends_data) { fetch_collective_input(m_buffer_mngr, data.bid, m_buffer); }

			{
#if TRACY_ENABLE
				ZoneScopedN("Ibcast");
				const auto req_msg = fmt::format("B{}, {} bytes", data.bid, m_buffer.get_payload_size_bytes());
				ZoneText(req_msg.data(), req_msg.size());
#endif
				CELERITY_TRACE("MPI_Ibcast([buffer of size {0}], {0}, MPI_BYTE, {1}, MPI_COMM_WORLD, &request)", m_buffer.get_payload_size_bytes(), data.root);
				MPI_Ibcast(m_buffer.get_payload(), static_cast<int>(m_buffer.get_payload_size_bytes()), MPI_BYTE, static_cast<int>(data.root), MPI_COMM_WORLD,
				    &m_request);
			}

			if(sends_data) {
				// Overlap async MPI_Ibcast with a broadcast to sibling devices
				ZoneScopedN("d2d broadcast");
				const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
				async_event pending_transfers;
				for(device_id did = 0; did < local_devices.num_compute_devices(); ++did) {
					pending_transfers.merge(make_buffer_region_coherent_on_device(m_buffer_mngr, data.bid, data.region, did));
				}
				m_d2d_broadcast = std::move(pending_transfers);
			}

			m_started = true;
		}

		if(m_request != MPI_REQUEST_NULL) {
			int mpi_done;
			MPI_Test(&m_request, &mpi_done, MPI_STATUS_IGNORE);
			if(mpi_done) {
#if TRACY_ENABLE
				const std::string_view msg_done = "Ibcast complete";
				TracyMessage(msg_done.data(), msg_done.size());
#endif
				m_request = MPI_REQUEST_NULL;
			}
		}

		if(m_d2d_broadcast.has_value() && m_d2d_broadcast->is_done()) {
#if TRACY_ENABLE
			const std::string_view msg_done = "d2d broadcast complete";
			TracyMessage(msg_done.data(), msg_done.size());
#endif
			m_d2d_broadcast.reset();
		}

		if(m_request == MPI_REQUEST_NULL && !m_d2d_broadcast.has_value()) {
			if(receives_data) commit_collective_update(m_buffer_mngr, data.bid, std::move(m_buffer));
			return true;
		} else {
			return false;
		}
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- SCATTER ------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	scatter_job::scatter_job(const command_pkg& pkg, buffer_manager& bm, node_id local_nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(local_nid) {
		assert(std::holds_alternative<scatter_data>(pkg.data));
	}

	std::string scatter_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<scatter_data>(pkg.data);
		if(data.root == m_local_nid) {
			return fmt::format("scatter {} of buffer {} to all others", merge_regions(data.dest_regions), data.bid);
		} else {
			return fmt::format("scatter {} of buffer {} from {}", data.dest_regions[m_local_nid], data.bid, data.root);
		}
	}

	bool scatter_job::execute(const command_pkg& pkg) {
		const auto data = std::get<scatter_data>(pkg.data);
		const bool sends_data = data.root == m_local_nid;
		const bool receives_data = !data.dest_regions[m_local_nid].empty();

		if(!m_started) {
			const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

			assert(!receives_data || !sends_data);

			if(sends_data) {
				std::vector<collective_buffer::region_spec> send_regions(data.dest_regions.size());
				for(node_id nid = 0; nid < data.dest_regions.size(); ++nid) {
					auto& r = send_regions[nid];
					r.region = data.dest_regions[nid];
					r.fetch = true;
					r.update = false;
					r.broadcast = false; // usually consumed only on host or a single device
				}
				m_send_buffer = collective_buffer(send_regions, buffer_info.element_size);
				fetch_collective_input(m_buffer_mngr, data.bid, m_send_buffer);
			}

			if(receives_data) {
				collective_buffer::region_spec recv_region;
				recv_region.region = data.dest_regions[m_local_nid];
				recv_region.fetch = false;
				recv_region.update = true;
				recv_region.broadcast = false;
				m_recv_buffer = collective_buffer({recv_region}, buffer_info.element_size);
			}

			{
#if TRACY_ENABLE
				ZoneScopedN("Iscatterv");
				const auto req_msg =
				    fmt::format("B{}, {} -> {} bytes", data.bid, m_send_buffer.get_payload_size_bytes(), m_recv_buffer.get_payload_size_bytes());
				ZoneText(req_msg.data(), req_msg.size());
#endif
				const auto root = static_cast<int>(data.root);
				CELERITY_TRACE("MPI_Iscatterv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {}, MPI_BYTE, {}, MPI_COMM_WORLD, &request)",
				    m_send_buffer.get_payload_size_bytes(), fmt::join(m_send_buffer.get_chunk_byte_sizes(), ", "),
				    fmt::join(m_send_buffer.get_chunk_byte_offsets(), ", "), static_cast<int>(m_recv_buffer.get_payload_size_bytes()),
				    m_recv_buffer.get_payload_size_bytes(), root);
				MPI_Iscatterv(m_send_buffer.get_payload(), m_send_buffer.get_chunk_byte_sizes().data(), m_send_buffer.get_chunk_byte_offsets().data(), MPI_BYTE,
				    m_recv_buffer.get_payload(), m_recv_buffer.get_payload_size_bytes(), MPI_BYTE, root, MPI_COMM_WORLD, &m_request);
			}

			const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			assert(local_devices.num_compute_devices() == data.local_device_coherence_regions.size());
			if(data.local_device_coherence_regions.size() > 1) {
				// Overlap async MPI_Iscatter with a scatter to sibling devices
				ZoneScopedN("d2d scatter");
				async_event pending_transfers;
				for(device_id did = 0; did < data.local_device_coherence_regions.size(); ++did) {
					pending_transfers.merge(make_buffer_region_coherent_on_device(m_buffer_mngr, data.bid, data.local_device_coherence_regions[did], did));
				}
				m_d2d_scatter = std::move(pending_transfers);
			}

			m_started = true;
		}

		if(m_request != MPI_REQUEST_NULL) {
			int mpi_done;
			MPI_Test(&m_request, &mpi_done, MPI_STATUS_IGNORE);
			if(mpi_done) {
#if TRACY_ENABLE
				const std::string_view msg_done = "Iscatterv complete";
				TracyMessage(msg_done.data(), msg_done.size());
#endif
				m_request = MPI_REQUEST_NULL;
			}
		}

		if(m_d2d_scatter.has_value() && m_d2d_scatter->is_done()) {
#if TRACY_ENABLE
			const std::string_view msg_done = "d2d scatter complete";
			TracyMessage(msg_done.data(), msg_done.size());
#endif
			m_d2d_scatter.reset();
		}

		if(m_request == MPI_REQUEST_NULL && !m_d2d_scatter.has_value()) {
			if(receives_data) commit_collective_update(m_buffer_mngr, data.bid, std::move(m_recv_buffer));
			return true;
		} else {
			return false;
		}
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- ALLTOALL -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	alltoall_job::alltoall_job(const command_pkg& pkg, buffer_manager& bm) : worker_job(pkg), m_buffer_mngr(bm) {
		assert(std::holds_alternative<alltoall_data>(pkg.data));
	}

	std::string alltoall_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<alltoall_data>(pkg.data);
		return fmt::format("all-to-all buffer {}", data.bid);
	}

	bool alltoall_job::execute(const command_pkg& pkg) {
		const auto data = std::get<alltoall_data>(pkg.data);

		if(!m_started) {
			const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

			std::vector<collective_buffer::region_spec> collective_regions(data.send_regions.size());
			for(node_id nid = 0; nid < data.send_regions.size(); ++nid) {
				auto& r = collective_regions[nid];
				r.region = data.send_regions[nid];
				r.fetch = true;
				r.update = r.broadcast = false;
			}
			m_send_buffer = collective_buffer(collective_regions, buffer_info.element_size);

			fetch_collective_input(m_buffer_mngr, data.bid, m_send_buffer);

			collective_regions.resize(data.recv_regions.size());
			for(node_id nid = 0; nid < data.recv_regions.size(); ++nid) {
				auto& r = collective_regions[nid];
				r.region = data.recv_regions[nid];
				r.fetch = false;
				r.update = true;
				r.broadcast = false;
			}
			m_recv_buffer = collective_buffer(collective_regions, buffer_info.element_size);

			{
#if TRACY_ENABLE
				ZoneScopedN("Ialltoallv");
				const auto req_msg =
				    fmt::format("B{}, {} -> {} bytes", data.bid, m_send_buffer.get_payload_size_bytes(), m_recv_buffer.get_payload_size_bytes());
				ZoneText(req_msg.data(), req_msg.size());
#endif
				CELERITY_TRACE(
				    "MPI_Ialltoallv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD, &request)",
				    m_send_buffer.get_payload_size_bytes(), fmt::join(m_send_buffer.get_chunk_byte_sizes(), ", "),
				    fmt::join(m_send_buffer.get_chunk_byte_offsets(), ", "), m_recv_buffer.get_payload_size_bytes(),
				    fmt::join(m_recv_buffer.get_chunk_byte_sizes(), ", "), fmt::join(m_recv_buffer.get_chunk_byte_offsets(), ", "));
				MPI_Ialltoallv(m_send_buffer.get_payload(), m_send_buffer.get_chunk_byte_sizes().data(), m_send_buffer.get_chunk_byte_offsets().data(),
				    MPI_BYTE, m_recv_buffer.get_payload(), m_recv_buffer.get_chunk_byte_sizes().data(), m_recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE,
				    MPI_COMM_WORLD, &m_request);
			}

			const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			assert(local_devices.num_compute_devices() == data.local_device_coherence_regions.size());
			if(data.local_device_coherence_regions.size() > 1) {
				// Overlap async MPI_Ialltoallv with a broadcast to sibling devices
				ZoneScopedN("d2d alltoall");
				async_event pending_transfers;
				for(device_id did = 0; did < data.local_device_coherence_regions.size(); ++did) {
					pending_transfers.merge(make_buffer_region_coherent_on_device(m_buffer_mngr, data.bid, data.local_device_coherence_regions[did], did));
				}
				m_d2d_alltoall = std::move(pending_transfers);
			}

			m_started = true;
		}

		if(m_request != MPI_REQUEST_NULL) {
			int mpi_done;
			MPI_Test(&m_request, &mpi_done, MPI_STATUS_IGNORE);
			if(mpi_done) {
#if TRACY_ENABLE
				const std::string_view msg_done = "Ialltoallv complete";
				TracyMessage(msg_done.data(), msg_done.size());
#endif
				m_request = MPI_REQUEST_NULL;
			}
		}

		if(m_d2d_alltoall.has_value() && m_d2d_alltoall->is_done()) {
#if TRACY_ENABLE
			const std::string_view msg_done = "d2d alltoall complete";
			TracyMessage(msg_done.data(), msg_done.size());
#endif
			m_d2d_alltoall.reset();
		}

		if(m_request == MPI_REQUEST_NULL && !m_d2d_alltoall.has_value()) {
			commit_collective_update(m_buffer_mngr, data.bid, std::move(m_recv_buffer));
			return true;
		} else {
			return false;
		}
	}

} // namespace detail
} // namespace celerity
