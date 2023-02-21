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

	struct contiguous_box {
		GridBox<3> box;
		size_t offset_bytes;
		size_t size_bytes;
	};

	class contiguous_box_builder {
	  public:
		contiguous_box_builder(const size_t element_size_bytes) : m_element_size_bytes(element_size_bytes) {}

		void reserve(const size_t capacity) { m_complete_boxes.reserve(capacity); }

		void push_box(const GridBox<3>& new_box, bool include) {
			if(new_box.empty()) return;
			const auto new_box_size_bytes = new_box.area() * m_element_size_bytes;
			if(m_ongoing_merge) {
				assert(m_ongoing_merge->box.area() > 0);
				if(include && GridBox<3>::areFusable<0>(m_ongoing_merge->box, new_box)) {
					m_ongoing_merge->box = GridBox<3>::fuse<0>(m_ongoing_merge->box, new_box);
					m_ongoing_merge->size_bytes += new_box_size_bytes;
				} else {
					commit();
					if(include) { begin_new_merge(new_box); }
				}
			} else if(include) {
				begin_new_merge(new_box);
			}
			m_current_offset_bytes += new_box_size_bytes;
		}

		void push_padding(const size_t padding_bytes) {
			if(padding_bytes == 0) return;
			if(m_ongoing_merge) { commit(); }
			m_current_offset_bytes += padding_bytes;
		}

		std::vector<contiguous_box> finish([[maybe_unused]] const size_t expected_total_size_bytes) {
			if(m_ongoing_merge) { commit(); }
			assert(m_current_offset_bytes == expected_total_size_bytes);
			m_current_offset_bytes = 0;
			return std::move(m_complete_boxes);
		}

	  private:
		size_t m_element_size_bytes;
		std::vector<contiguous_box> m_complete_boxes;
		std::optional<contiguous_box> m_ongoing_merge;
		size_t m_current_offset_bytes = 0;

		void commit() {
			assert(m_ongoing_merge);
			m_complete_boxes.push_back(*m_ongoing_merge);
			m_ongoing_merge.reset();
		}

		void begin_new_merge(const GridBox<3>& new_box) {
			assert(!m_ongoing_merge);
			const auto new_box_size_bytes = new_box.area() * m_element_size_bytes;
			m_ongoing_merge = contiguous_box{new_box, m_current_offset_bytes, new_box_size_bytes};
		}
	};

	class collective_buffer {
	  public:
		struct region_spec {
			GridRegion<3> region;
			bool fetch = false;     // will be either communicated to other ranks or broadcast locally (local get_buffer_data)
			bool update = false;    // will invalidate data present on all memories (local set_buffer_data OR broadcast_immediately)
			bool broadcast = false; // will be consumed by all local devices (local broadcast_immediately, implies `update`)
		};

		collective_buffer() = default;

		collective_buffer(const std::vector<region_spec>& peer_regions, const size_t element_size_bytes)
		    : m_chunk_byte_sizes(peer_regions.size()), m_chunk_byte_offsets(peer_regions.size()) {
			ZoneScopedN("alloc collective buffer");

			size_t current_offset_bytes = 0;
			contiguous_box_builder fetch_builder(element_size_bytes);
			contiguous_box_builder update_builder(element_size_bytes);
			contiguous_box_builder broadcast_builder(element_size_bytes);

			for(size_t i = 0; i < peer_regions.size(); ++i) {
				const auto& r = peer_regions[i];
				size_t chunk_size_bytes = 0;
				if(r.fetch | r.update | r.broadcast) {
					r.region.scanByBoxes([&](const GridBox<3>& box) {
						fetch_builder.push_box(box, r.fetch);
						update_builder.push_box(box, r.update);
						broadcast_builder.push_box(box, r.broadcast);
					});
					chunk_size_bytes = r.region.area() * element_size_bytes;
				}
				m_chunk_byte_offsets[i] = static_cast<int>(current_offset_bytes);
				m_chunk_byte_sizes[i] = static_cast<int>(chunk_size_bytes);
				current_offset_bytes += chunk_size_bytes;
				m_broadcast_covers_all_updates &= !r.update | r.broadcast;
			}

			m_fetch_boxes = fetch_builder.finish(current_offset_bytes);
			m_update_boxes = update_builder.finish(current_offset_bytes);
			m_broadcast_boxes = broadcast_builder.finish(current_offset_bytes);

			m_payload_size_bytes = current_offset_bytes;
			m_payload = make_uninitialized_payload<std::byte>(m_payload_size_bytes);

			if(::spdlog::should_log(spdlog::level::trace)) {
				std::string log;
				log += "peer regions:";
				for(auto& r : peer_regions) {
					log += fmt::format("\n    {} (", r.region);
					if(r.fetch) log += "fetch ";
					if(r.update) log += "update ";
					if(r.broadcast) log += "broadcast";
					log += ")";
				}
				log += "\nfetch boxes:";
				for(auto& b : m_fetch_boxes) {
					log += fmt::format("\n    {} @{} +{}", b.box, b.offset_bytes, b.size_bytes);
				}
				log += "\nupdate boxes:";
				for(auto& b : m_update_boxes) {
					log += fmt::format("\n    {} @{} +{}", b.box, b.offset_bytes, b.size_bytes);
				}
				log += "\nbroadcast boxes:";
				for(auto& b : m_broadcast_boxes) {
					log += fmt::format("\n    {} @{} +{}", b.box, b.offset_bytes, b.size_bytes);
				}
				log += fmt::format("\npayload: {} bytes", m_payload_size_bytes);
				CELERITY_TRACE("collective buffer\n{}", log);
			}
		}

		const auto& get_fetch_boxes() const { return m_fetch_boxes; }
		const auto& get_update_boxes() const { return m_update_boxes; }
		const auto& get_broadcast_boxes() const { return m_broadcast_boxes; }
		const auto& get_chunk_byte_sizes() const { return m_chunk_byte_sizes; }
		const auto& get_chunk_byte_offsets() const { return m_chunk_byte_offsets; }
		size_t get_payload_size_bytes() const { return m_payload_size_bytes; }
		void* get_payload(size_t offset_bytes = 0) { return static_cast<std::byte*>(m_payload.get_pointer()) + offset_bytes; }
		bool payload_is_single_contiguous_update() const { return m_update_boxes.size() == 1 && !has_multiple_boxes(); }
		bool broadcast_covers_all_updates() const { return m_broadcast_covers_all_updates; }
		unique_payload_ptr take_payload() { return std::move(m_payload); };

	  private:
		std::vector<contiguous_box> m_fetch_boxes;
		std::vector<contiguous_box> m_update_boxes;
		std::vector<contiguous_box> m_broadcast_boxes;
		std::vector<int> m_chunk_byte_sizes;
		std::vector<int> m_chunk_byte_offsets;
		size_t m_payload_size_bytes = 0;
		unique_payload_ptr m_payload;
		bool m_broadcast_covers_all_updates = true;

		bool has_multiple_boxes() const {
			const GridBox<3>* single_box = nullptr;
			for(const auto vec : {&m_fetch_boxes, &m_update_boxes, &m_broadcast_boxes}) {
				if(vec->size() > 1) return true;
				if(vec->size() == 1) {
					if(single_box && vec->front().box != *single_box) return true;
					single_box = &vec->front().box;
				}
			}
			return false;
		}
	};

	void fetch_collective_input(buffer_manager& bm, const buffer_id bid, collective_buffer& buffer) {
		ZoneScopedN("fetch input");

		async_event fetch_complete;
		for(auto& [box, offset_bytes, size_bytes] : buffer.get_fetch_boxes()) {
			fetch_complete.merge(bm.get_buffer_data(bid, grid_box_to_subrange(box), buffer.get_payload(offset_bytes)));
		}
		fetch_complete.wait();
	}

	// try to immediately upload allgather data to all device memories
	// TODO NOCOMMIT (at least the env var handling)
	const bool device_broadcast_enabled = getenv("CELERITY_USE_ALLGATHER_BROADCAST");

	void commit_collective_update(buffer_manager& bm, const buffer_id bid, collective_buffer&& buffer) {
		ZoneScopedN("commit");

		assert(buffer.broadcast_covers_all_updates() || buffer.get_broadcast_boxes().empty()); // if not we definitely messed up region_spec::update/broadcast

		bool used_broadcast = false;
		if(device_broadcast_enabled && buffer.broadcast_covers_all_updates()) {
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

	gather_job::gather_job(const command_pkg& pkg, buffer_manager& bm, node_id nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid) {
		assert(std::holds_alternative<gather_data>(pkg.data));
	}

	std::string gather_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<gather_data>(pkg.data);
		return fmt::format("gather {} of buffer {} to {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid), data.root);
	}

	bool gather_job::execute(const command_pkg& pkg) {
		ZoneScoped;

		const auto data = std::get<gather_data>(pkg.data);
		assert(data.source_regions[data.root].empty());

		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool receives_data = m_local_nid == data.root;
		const bool sends_data = !receives_data;

		collective_buffer send_buffer;
		if(sends_data) {
			collective_buffer::region_spec region_spec;
			region_spec.region = data.source_regions[m_local_nid];
			region_spec.fetch = true;
			region_spec.update = false;
			region_spec.broadcast = false;
			send_buffer = collective_buffer({region_spec}, buffer_info.element_size);
			fetch_collective_input(m_buffer_mngr, data.bid, send_buffer);
		}

		collective_buffer recv_buffer;
		if(receives_data) {
			std::vector<collective_buffer::region_spec> collective_regions(data.source_regions.size());
			for(node_id nid = 0; nid < data.source_regions.size(); ++nid) {
				auto& r = collective_regions[nid];
				r.region = data.source_regions[nid];
				r.fetch = false;
				r.update = true; // usually consumed only on host or one device
				r.broadcast = false;
			}
			recv_buffer = collective_buffer(collective_regions, buffer_info.element_size);
		}

		{
			ZoneScopedN("Gatherv");
			CELERITY_TRACE("MPI_Gatherv([buffer of size {}], {}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, {}, MPI_COMM_WORLD)",
			    send_buffer.get_payload_size_bytes(), send_buffer.get_payload_size_bytes(), recv_buffer.get_payload_size_bytes(),
			    fmt::join(recv_buffer.get_chunk_byte_sizes(), ", "), fmt::join(recv_buffer.get_chunk_byte_offsets(), ", "), data.root);
			MPI_Gatherv(send_buffer.get_payload(), send_buffer.get_payload_size_bytes(), MPI_BYTE, recv_buffer.get_payload(),
			    recv_buffer.get_chunk_byte_sizes().data(), recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, static_cast<int>(data.root), MPI_COMM_WORLD);
		}

		if(receives_data) commit_collective_update(m_buffer_mngr, data.bid, std::move(recv_buffer));

		return true;
	}

	allgather_job::allgather_job(const command_pkg& pkg, buffer_manager& bm, node_id nid, size_t num_local_devices)
	    : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid), m_num_local_devices(num_local_devices) {
		assert(std::holds_alternative<allgather_data>(pkg.data));
	}

	std::string allgather_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<allgather_data>(pkg.data);
		return fmt::format("allgather {} of buffer {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid));
	}


	bool allgather_job::execute(const command_pkg& pkg) {
		ZoneScoped;
		const auto data = std::get<allgather_data>(pkg.data);

		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		std::vector<collective_buffer::region_spec> collective_regions(data.source_regions.size());
		for(node_id nid = 0; nid < data.source_regions.size(); ++nid) {
			auto& r = collective_regions[nid];
			r.region = data.source_regions[nid];
			r.fetch = nid == m_local_nid;
			r.update = nid != m_local_nid;
			r.broadcast = nid != m_local_nid || /* always h2d-broadcast locally */ m_num_local_devices > 1;
		}

		collective_buffer buffer(collective_regions, buffer_info.element_size);

		fetch_collective_input(m_buffer_mngr, data.bid, buffer);

		{
			ZoneScopedN("Allgatherv");
			// TODO make asynchronous?
			MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, buffer.get_payload(), buffer.get_chunk_byte_sizes().data(), buffer.get_chunk_byte_offsets().data(),
			    MPI_BYTE, MPI_COMM_WORLD);
			CELERITY_TRACE("MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD)",
			    buffer.get_payload_size_bytes(), fmt::join(buffer.get_chunk_byte_sizes(), ", "), fmt::join(buffer.get_chunk_byte_offsets(), ", "));
		}

		commit_collective_update(m_buffer_mngr, data.bid, std::move(buffer));

		return true;
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

		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool sends_data = data.root == m_local_nid;
		const bool receives_data = !sends_data;

		collective_buffer::region_spec region_spec;
		region_spec.region = data.region;
		region_spec.fetch = sends_data;
		region_spec.update = region_spec.broadcast = receives_data;

		collective_buffer buffer({std::move(region_spec)}, buffer_info.element_size);

		if(sends_data) { fetch_collective_input(m_buffer_mngr, data.bid, buffer); }

		{
			ZoneScopedN("Ibcast");
			CELERITY_TRACE("MPI_Ibcast([buffer of size {0}], {0}, MPI_BYTE, {1}, MPI_COMM_WORLD, &request)", buffer.get_payload_size_bytes(), data.root);
			MPI_Request request;
			MPI_Ibcast(
			    buffer.get_payload(), static_cast<int>(buffer.get_payload_size_bytes()), MPI_BYTE, static_cast<int>(data.root), MPI_COMM_WORLD, &request);

			if(sends_data) {
				// Overlap async MPI_Ibcast with a broadcast to sibling devices by triggering make_buffer_subrange_coherent
				ZoneScopedN("d2d broadcast");
				const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
				async_event pending_local_broadcast;
				data.region.scanByBoxes([&](const GridBox<3>& box) {
					const auto sr = grid_box_to_subrange(box);
					for(device_id did = 0; did < local_devices.num_compute_devices(); ++did) {
						auto info = m_buffer_mngr.access_device_buffer(local_devices.get_memory_id(did), data.bid, access_mode::read, sr.range, sr.offset);
						pending_local_broadcast.merge(std::move(info.pending_transfers));
					}
				});
				pending_local_broadcast.wait();
			}

			MPI_Wait(&request, MPI_STATUS_IGNORE);
		}

		if(receives_data) { commit_collective_update(m_buffer_mngr, data.bid, std::move(buffer)); }

		return true;
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

		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool sends_data = data.root == m_local_nid;
		const bool receives_data = !data.dest_regions[m_local_nid].empty();
		assert(!receives_data || !sends_data);

		collective_buffer send_buffer;
		if(sends_data) {
			std::vector<collective_buffer::region_spec> send_regions(data.dest_regions.size());
			for(node_id nid = 0; nid < data.dest_regions.size(); ++nid) {
				auto& r = send_regions[nid];
				r.region = data.dest_regions[nid];
				r.fetch = true;
				r.update = false;
				r.broadcast = false; // usually consumed only on host or a single device
			}
			send_buffer = collective_buffer(send_regions, buffer_info.element_size);
			fetch_collective_input(m_buffer_mngr, data.bid, send_buffer);
		}

		collective_buffer recv_buffer;
		if(receives_data) {
			collective_buffer::region_spec recv_region;
			recv_region.region = data.dest_regions[m_local_nid];
			recv_region.fetch = false;
			recv_region.update = true;
			recv_region.broadcast = false;
			recv_buffer = collective_buffer({recv_region}, buffer_info.element_size);
		}

		{
			ZoneScopedN("Scatterv");
			const auto root = static_cast<int>(data.root);
			CELERITY_TRACE("MPI_Scatterv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {}, MPI_BYTE, {}, MPI_COMM_WORLD)",
			    send_buffer.get_payload_size_bytes(), fmt::join(send_buffer.get_chunk_byte_sizes(), ", "),
			    fmt::join(send_buffer.get_chunk_byte_offsets(), ", "), static_cast<int>(recv_buffer.get_payload_size_bytes()),
			    recv_buffer.get_payload_size_bytes(), root);
			MPI_Scatterv(send_buffer.get_payload(), send_buffer.get_chunk_byte_sizes().data(), send_buffer.get_chunk_byte_offsets().data(), MPI_BYTE,
			    recv_buffer.get_payload(), recv_buffer.get_payload_size_bytes(), MPI_BYTE, root, MPI_COMM_WORLD);
		}

		if(receives_data) { commit_collective_update(m_buffer_mngr, data.bid, std::move(recv_buffer)); }

		return true;
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

		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		std::vector<collective_buffer::region_spec> collective_regions(data.send_regions.size());
		for(node_id nid = 0; nid < data.send_regions.size(); ++nid) {
			auto& r = collective_regions[nid];
			r.region = data.send_regions[nid];
			r.fetch = nid == true;
			r.update = r.broadcast = false;
		}
		collective_buffer send_buffer(collective_regions, buffer_info.element_size);

		fetch_collective_input(m_buffer_mngr, data.bid, send_buffer);

		collective_regions.resize(data.recv_regions.size());
		for(node_id nid = 0; nid < data.recv_regions.size(); ++nid) {
			auto& r = collective_regions[nid];
			r.region = data.recv_regions[nid];
			r.fetch = nid == false;
			r.update = true;
			r.broadcast = false;
		}
		collective_buffer recv_buffer(collective_regions, buffer_info.element_size);

		{
			ZoneScopedN("Ialltoallv");
			CELERITY_TRACE(
			    "MPI_Ialltoallv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD, &request)",
			    send_buffer.get_payload_size_bytes(), fmt::join(send_buffer.get_chunk_byte_sizes(), ", "),
			    fmt::join(send_buffer.get_chunk_byte_offsets(), ", "), recv_buffer.get_payload_size_bytes(),
			    fmt::join(recv_buffer.get_chunk_byte_sizes(), ", "), fmt::join(recv_buffer.get_chunk_byte_offsets(), ", "));
			MPI_Request request;
			MPI_Ialltoallv(send_buffer.get_payload(), send_buffer.get_chunk_byte_sizes().data(), send_buffer.get_chunk_byte_offsets().data(), MPI_BYTE,
			    recv_buffer.get_payload(), recv_buffer.get_chunk_byte_sizes().data(), recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, MPI_COMM_WORLD,
			    &request);

			const auto& local_devices = runtime::get_instance().get_local_devices(); // TODO do not get_instance()
			assert(local_devices.num_compute_devices() == data.local_device_coherence_regions.size());
			if(data.local_device_coherence_regions.size() > 1) {
				// Overlap async MPI_Ibcast with a broadcast to sibling devices by triggering make_buffer_subrange_coherent
				ZoneScopedN("d2d alltoall");
				async_event pending_local_alltoall;
				for(device_id did = 0; did < data.local_device_coherence_regions.size(); ++did) {
					data.local_device_coherence_regions[did].scanByBoxes([&](const GridBox<3>& box) {
						const auto sr = grid_box_to_subrange(box);
						auto info = m_buffer_mngr.access_device_buffer(local_devices.get_memory_id(did), data.bid, access_mode::read, sr.range, sr.offset);
						pending_local_alltoall.merge(std::move(info.pending_transfers));
					});
				}
				pending_local_alltoall.wait();
			}

			MPI_Wait(&request, MPI_STATUS_IGNORE);
		}

		commit_collective_update(m_buffer_mngr, data.bid, std::move(recv_buffer));

		return true;
	}

} // namespace detail
} // namespace celerity
