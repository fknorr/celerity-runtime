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

	gather_job::gather_job(const command_pkg& pkg, buffer_manager& bm, node_id nid) : worker_job(pkg), m_buffer_mngr(bm), m_local_nid(nid) {
		assert(std::holds_alternative<gather_data>(pkg.data));
	}

	std::string gather_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<gather_data>(pkg.data);
		if(data.single_dest_nid) {
			return fmt::format("gather {} of buffer {} to {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid), *data.single_dest_nid);
		} else {
			return fmt::format("allgather {} of buffer {}", data.source_regions[m_local_nid], static_cast<size_t>(data.bid));
		}
	}

	class linearized_region_array {
	  public:
		linearized_region_array() = default;

		linearized_region_array(const std::vector<GridRegion<3>>& source_regions, const size_t element_size)
		    : m_byte_sizes(source_regions.size()), m_byte_offsets(source_regions.size()) {
			std::optional<GridBox<3>> merge;
			size_t merge_size_bytes = 0;
			size_t merge_offset_bytes = 0;
			size_t box_offset_bytes = 0;
			size_t chunk_offset_bytes = 0;
			for(size_t i = 0; i < source_regions.size(); ++i) {
				size_t chunk_size_bytes = 0;
				auto& region = source_regions[i];
				region.scanByBoxes([&](const GridBox<3>& box) {
					const auto box_size_bytes = box.area() * element_size;
					if(merge.has_value()) {
						if(GridBox<3>::areFusable<0>(*merge, box) && merge->get_max()[0] == box.get_min()[0]) {
							*merge = GridBox<3>::fuse<0>(*merge, box);
							merge_size_bytes += box_size_bytes;
						} else {
							m_size_offset_boxes.emplace_back(merge_size_bytes, merge_offset_bytes, *merge);
							merge.reset();
						}
					}
					if(!merge.has_value()) {
						merge.emplace(box);
						merge_size_bytes = box_size_bytes;
						merge_offset_bytes = box_offset_bytes;
					}
					box_offset_bytes += box_size_bytes;
					chunk_size_bytes += box_size_bytes;
				});
				m_byte_sizes[i] = static_cast<int>(chunk_size_bytes);
				m_byte_offsets[i] = static_cast<int>(chunk_offset_bytes);
				chunk_offset_bytes += chunk_size_bytes;
				m_total_size_bytes += chunk_size_bytes;
			}
			if(merge.has_value()) { m_size_offset_boxes.emplace_back(merge_size_bytes, merge_offset_bytes, *merge); }
		}

		const auto& get_chunk_byte_sizes() const { return m_byte_sizes; }
		const auto& get_chunk_byte_offsets() const { return m_byte_offsets; }
		size_t get_size_bytes() const { return m_total_size_bytes; }

	  protected:
		const auto& get_size_offset_boxes() const { return m_size_offset_boxes; }

	  private:
		std::vector<std::tuple<size_t, size_t, GridBox<3>>> m_size_offset_boxes;
		std::vector<int> m_byte_sizes;
		std::vector<int> m_byte_offsets;
		size_t m_total_size_bytes = 0;
	};

	class collective_send_buffer : public linearized_region_array {
	  public:
		collective_send_buffer() = default;

		collective_send_buffer(const std::vector<GridRegion<3>>& source_regions, size_t element_size) : linearized_region_array(source_regions, element_size) {
			ZoneScopedN("allocate send buffer");

			m_payload = make_uninitialized_payload<std::byte>(get_size_bytes());
			m_box_pointers.reserve(get_size_offset_boxes().size());
			for(const auto& [size, offset, box] : get_size_offset_boxes()) {
				m_box_pointers.emplace_back(box, static_cast<std::byte*>(m_payload.get_pointer()) + offset);
			}
		}

		void* get_pointer() { return m_payload.get_pointer(); }
		auto& get_boxes() { return m_box_pointers; }

	  private:
		unique_payload_ptr m_payload;
		std::vector<std::pair<GridBox<3>, void*>> m_box_pointers;
	};

	class collective_receive_buffer : public linearized_region_array {
	  public:
		collective_receive_buffer() = default;

		collective_receive_buffer(const std::vector<GridRegion<3>>& source_regions, size_t element_size)
		    : linearized_region_array(source_regions, element_size), m_element_size(element_size) {
			ZoneScopedN("allocate receive buffer");

			assert(!get_size_offset_boxes().empty());
			m_box_payloads.reserve(get_size_offset_boxes().size());
			for(const auto& [size, offset, box] : get_size_offset_boxes()) {
				auto payload = make_uninitialized_payload<std::byte>(size);
				m_box_payloads.emplace_back(box, std::move(payload));
			}

			assert(!m_box_payloads.empty());
			if(m_box_payloads.size() > 1) {
				m_staging_buffer_opt = std::make_unique<std::byte[]>(get_size_bytes());
				m_receive_buffer = m_staging_buffer_opt.get();
			} else {
				auto& [box, payload] = m_box_payloads.front();
				m_receive_buffer = payload.get_pointer();
			}
		}

		void* get_pointer() { return m_receive_buffer; }

		auto into_boxes() && {
			if(m_staging_buffer_opt) {
				ZoneScopedN("split receive buffer");
				size_t offset_bytes = 0;
				for(auto& [box, payload] : m_box_payloads) {
					const size_t size_bytes = box.area() * m_element_size;
					memcpy(payload.get_pointer(), m_staging_buffer_opt.get() + offset_bytes, size_bytes);
					offset_bytes += size_bytes;
				}
				m_staging_buffer_opt.reset();
			}
			m_receive_buffer = nullptr;
			return std::move(m_box_payloads);
		}

	  private:
		size_t m_element_size;
		std::unique_ptr<std::byte[]> m_staging_buffer_opt;
		std::vector<std::pair<GridBox<3>, unique_payload_ptr>> m_box_payloads;
		void* m_receive_buffer = nullptr;
	};

	void commit_collective_receive(buffer_manager& bm, const buffer_id bid, collective_receive_buffer&& recv_buffer) {
		ZoneScopedN("commit");

		auto box_payloads = std::move(recv_buffer).into_boxes();

		bool used_broadcast = false;
		// try to immediately upload allgather data to all device memories
		// TODO NOCOMMIT (at least the env var handling)
		if(getenv("CELERITY_USE_ALLGATHER_BROADCAST")) {
			bool lock_success = true;
			auto& local_devices = runtime::get_instance().get_local_devices();
			auto num_devices = local_devices.num_compute_devices();
			std::vector<bool> mem_locked(num_devices, false);
			for(size_t device_id = 0; device_id < num_devices; ++device_id) {
				auto mem_id = local_devices.get_memory_id(device_id);
				if(bm.try_lock(device_id, mem_id, {bid})) {
					mem_locked[device_id] = true;
				} else {
					lock_success = false;
					break;
				}
			}
			if(lock_success) {
				bool is_broadcast_possible = true;
				for(const auto& [box, _] : box_payloads) {
					const auto sr = grid_box_to_subrange(box);
					is_broadcast_possible &= bm.is_broadcast_possible(bid, sr);
				}
				if(is_broadcast_possible) {
					for(auto& [box, payload] : box_payloads) {
						const auto sr = grid_box_to_subrange(box);
						CELERITY_TRACE("immediately_broadcast_data({}, {}, payload)", (int)bid, sr);
						bm.immediately_broadcast_data(bid, sr, std::move(payload));
					}
					used_broadcast = true;
				}
			}
			for(size_t device_id = 0; device_id < num_devices; ++device_id) {
				if(mem_locked[device_id]) { bm.unlock(device_id); }
			}
		}
		if(!used_broadcast) {
			for(auto& [box, payload] : box_payloads) {
				const auto sr = grid_box_to_subrange(box);
				CELERITY_TRACE("non-broadcast set_buffer_data({}, {}, payload)", (int)bid, sr);
				bm.set_buffer_data(bid, sr, std::move(payload));
			}
		}
	}

	bool gather_job::execute(const command_pkg& pkg) {
		ZoneScoped;
		const auto data = std::get<gather_data>(pkg.data);

		// TODO usually we do not want lock host memory, but all device memories here
		if(!m_buffer_mngr.try_lock(pkg.cid, host_memory_id, {data.bid})) return false;
		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool sends_data = !data.source_regions[m_local_nid].empty();
		const bool receives_data = !data.single_dest_nid.has_value() || *data.single_dest_nid == m_local_nid;

		collective_send_buffer send_buffer;
		if(sends_data) {
			ZoneScopedN("fetch input");
			send_buffer = collective_send_buffer({data.source_regions[m_local_nid]}, buffer_info.element_size);
			for(auto& [box, pointer] : send_buffer.get_boxes()) {
				m_buffer_mngr.get_buffer_data(data.bid, grid_box_to_subrange(box), pointer).wait();
			}
		}

		collective_receive_buffer recv_buffer;
		if(receives_data) {
			ZoneScopedN("alloc output");
			recv_buffer = collective_receive_buffer(data.source_regions, buffer_info.element_size);
		}

		if(data.single_dest_nid) {
			ZoneScopedN("Gatherv");
			const auto root = static_cast<int>(*data.single_dest_nid);
			MPI_Gatherv(send_buffer.get_pointer(), send_buffer.get_size_bytes(), MPI_BYTE, recv_buffer.get_pointer(), recv_buffer.get_chunk_byte_sizes().data(),
			    recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, root, MPI_COMM_WORLD);
			CELERITY_TRACE("MPI_Gatherv([buffer of size {}], {}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, {}, MPI_COMM_WORLD)",
			    send_buffer.get_size_bytes(), send_buffer.get_size_bytes(), recv_buffer.get_size_bytes(), fmt::join(recv_buffer.get_chunk_byte_sizes(), ", "),
			    fmt::join(recv_buffer.get_chunk_byte_offsets(), ", "), root);
		} else {
			ZoneScopedN("Allgatherv");
			// TODO make asynchronous?
			MPI_Allgatherv(send_buffer.get_pointer(), send_buffer.get_size_bytes(), MPI_BYTE, recv_buffer.get_pointer(),
			    recv_buffer.get_chunk_byte_sizes().data(), recv_buffer.get_chunk_byte_offsets().data(), MPI_BYTE, MPI_COMM_WORLD);
			CELERITY_TRACE("MPI_Allgatherv([buffer of size {}], {}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD)",
			    send_buffer.get_size_bytes(), send_buffer.get_size_bytes(), recv_buffer.get_size_bytes(), fmt::join(recv_buffer.get_chunk_byte_sizes(), ", "),
			    fmt::join(recv_buffer.get_chunk_byte_offsets(), ", "));
		}

		if(receives_data) { commit_collective_receive(m_buffer_mngr, data.bid, std::move(recv_buffer)); }

		m_buffer_mngr.unlock(pkg.cid);
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
		return fmt::format("broadcast {} of buffer {} from {}", data.region, static_cast<size_t>(data.bid), data.source_nid);
	}

	bool broadcast_job::execute(const command_pkg& pkg) {
		const auto data = std::get<broadcast_data>(pkg.data);

		if(!m_buffer_mngr.try_lock(pkg.cid, host_memory_id, {data.bid})) return false;
		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool sends_data = data.source_nid == m_local_nid;
		const bool receives_data = !sends_data;

		collective_send_buffer send_buffer;
		collective_receive_buffer recv_buffer;
		size_t byte_size;
		void* buffer;
		if(sends_data) {
			ZoneScopedN("fetch input");
			send_buffer = collective_send_buffer({data.region}, buffer_info.element_size);
			for(auto& [box, pointer] : send_buffer.get_boxes()) {
				m_buffer_mngr.get_buffer_data(data.bid, grid_box_to_subrange(box), pointer).wait();
			}
			buffer = send_buffer.get_pointer();
			byte_size = send_buffer.get_size_bytes();
		} else {
			ZoneScopedN("alloc output");
			recv_buffer = collective_receive_buffer({data.region}, buffer_info.element_size);
			buffer = recv_buffer.get_pointer();
			byte_size = recv_buffer.get_size_bytes();
		}

		{
			ZoneScopedN("Bcast");
			CELERITY_TRACE("MPI_Bcast([buffer of size {0}], {0}, MPI_BYTE, {1}, MPI_COMM_WORLD)", byte_size, data.source_nid);
			MPI_Bcast(buffer, static_cast<int>(byte_size), MPI_BYTE, static_cast<int>(data.source_nid), MPI_COMM_WORLD);
		}

		if(receives_data) { commit_collective_receive(m_buffer_mngr, data.bid, std::move(recv_buffer)); }

		m_buffer_mngr.unlock(pkg.cid);
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
		if(data.source_nid == m_local_nid) {
			return fmt::format("scatter {} of buffer {} to all", data.source_region, data.bid);
		} else {
			return fmt::format("scatter {} of buffer {} from {}", data.dest_regions[m_local_nid], data.bid, data.source_nid);
		}
	}

	bool scatter_job::execute(const command_pkg& pkg) {
		const auto data = std::get<scatter_data>(pkg.data);

		if(!m_buffer_mngr.try_lock(pkg.cid, host_memory_id, {data.bid})) return false;
		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		const bool sends_data = data.source_nid == m_local_nid;
		const bool receives_data = !data.dest_regions[m_local_nid].empty();

		collective_send_buffer send_buffer;
		if(sends_data) {
			ZoneScopedN("fetch input");
			send_buffer = collective_send_buffer(data.dest_regions, buffer_info.element_size);
			for(auto& [box, pointer] : send_buffer.get_boxes()) {
				m_buffer_mngr.get_buffer_data(data.bid, grid_box_to_subrange(box), pointer).wait();
			}
		}

		collective_receive_buffer recv_buffer;
		if(receives_data) {
			ZoneScopedN("alloc output");
			recv_buffer = collective_receive_buffer({data.dest_regions[m_local_nid]}, buffer_info.element_size);
		}

		{
			ZoneScopedN("Scatterv");
			const auto root = static_cast<int>(data.source_nid);
			CELERITY_TRACE("MPI_Scatterv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {}, MPI_BYTE, {}, MPI_COMM_WORLD)",
			    send_buffer.get_size_bytes(), fmt::join(send_buffer.get_chunk_byte_sizes(), ", "), fmt::join(send_buffer.get_chunk_byte_offsets(), ", "),
			    recv_buffer.get_size_bytes(), recv_buffer.get_size_bytes(), root);
			MPI_Scatterv(send_buffer.get_pointer(), send_buffer.get_chunk_byte_sizes().data(), send_buffer.get_chunk_byte_offsets().data(), MPI_BYTE,
			    recv_buffer.get_pointer(), recv_buffer.get_size_bytes(), MPI_BYTE, root, MPI_COMM_WORLD);
		}

		if(receives_data) { commit_collective_receive(m_buffer_mngr, data.bid, std::move(recv_buffer)); }

		m_buffer_mngr.unlock(pkg.cid);
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

		if(!m_buffer_mngr.try_lock(pkg.cid, host_memory_id, {data.bid})) return false;
		const auto buffer_info = m_buffer_mngr.get_buffer_info(data.bid);

		// TODO get around INT_MAX limitation
		std::vector<int> send_chunk_byte_sizes(data.send_regions.size());
		std::vector<int> send_chunk_byte_offsets(send_chunk_byte_sizes.size());
		int send_buffer_size = 0;
		for(size_t i = 0; i < data.send_regions.size(); ++i) {
			send_chunk_byte_sizes[i] = data.send_regions[i].area() * buffer_info.element_size;
			send_chunk_byte_offsets[i] = send_buffer_size;
			send_buffer_size += send_chunk_byte_sizes[i];
		}

		unique_payload_ptr send_buffer;
		{
			ZoneScopedN("fetch input");
			send_buffer = make_uninitialized_payload<std::byte>(send_buffer_size);
			auto send_buffer_cursor = static_cast<std::byte*>(send_buffer.get_pointer());
			for(const auto& send : data.send_regions) {
				send.scanByBoxes([&](const GridBox<3>& box) {
					m_buffer_mngr.get_buffer_data(data.bid, grid_box_to_subrange(box), send_buffer_cursor).wait();
					send_buffer_cursor += box.area() * buffer_info.element_size;
				});
			}
		}

		std::vector<int> recv_chunk_byte_sizes(data.recv_regions.size());
		std::vector<int> recv_chunk_byte_offsets(recv_chunk_byte_sizes.size());
		int recv_buffer_size = 0;
		for(size_t i = 0; i < data.recv_regions.size(); ++i) {
			recv_chunk_byte_sizes[i] = data.recv_regions[i].area() * buffer_info.element_size;
			recv_chunk_byte_offsets[i] = recv_buffer_size;
			recv_buffer_size += recv_chunk_byte_sizes[i];
		}

		std::unique_ptr<std::byte[]> recv_buffer; // TODO can become a unique_payload_ptr once we eliminate the box-copy below
		{
			ZoneScopedN("alloc output");
			recv_buffer = std::make_unique<std::byte[]>(recv_buffer_size);
		}

		{
			ZoneScopedN("Alltoallv");
			CELERITY_TRACE("MPI_Alltoallv([buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, [buffer of size {}], {{{}}}, {{{}}}, MPI_BYTE, MPI_COMM_WORLD)",
			    send_buffer_size, fmt::join(send_chunk_byte_sizes, ", "), fmt::join(send_chunk_byte_offsets, ", "), recv_buffer_size,
			    fmt::join(recv_chunk_byte_sizes, ", "), fmt::join(recv_chunk_byte_offsets, ", "));
			MPI_Alltoallv(send_buffer.get_pointer(), send_chunk_byte_sizes.data(), send_chunk_byte_offsets.data(), MPI_BYTE, recv_buffer.get(),
			    recv_chunk_byte_sizes.data(), recv_chunk_byte_offsets.data(), MPI_BYTE, MPI_COMM_WORLD);
		}

		{
			ZoneScopedN("commit");
			auto recv_buffer_cursor = static_cast<std::byte*>(recv_buffer.get());
			for(const auto& recv : data.recv_regions) {
				recv.scanByBoxes([&](const GridBox<3>& box) {
					const auto box_size_bytes = box.area() * buffer_info.element_size;
					auto box_buffer = make_uninitialized_payload<std::byte>(box_size_bytes); // TODO eliminate copy
					memcpy(box_buffer.get_pointer(), recv_buffer_cursor, box_size_bytes);
					m_buffer_mngr.set_buffer_data(data.bid, grid_box_to_subrange(box), std::move(box_buffer));
					recv_buffer_cursor += box_size_bytes;
				});
			}
		}

		m_buffer_mngr.unlock(pkg.cid);
		return true;
	}

} // namespace detail
} // namespace celerity
