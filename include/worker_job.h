#pragma once

#include <cassert>
#include <chrono>
#include <future>
#include <limits>
#include <utility>

#include "buffer_transfer_manager.h"
#include "closure_hydrator.h"
#include "command.h"
#include "host_queue.h"
#include "log.h"

namespace celerity {
namespace detail {

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

	class device_queue;
	class executor;
	class task_manager;
	class reduction_manager;
	class buffer_manager;

	class worker_job;

	class worker_job {
	  public:
		worker_job(const worker_job&) = delete;
		worker_job(worker_job&&) = delete;

		virtual ~worker_job() = default;

		bool prepare();
		void start();
		void update() noexcept;

		bool is_running() const { return m_running; }
		bool is_done() const { return m_done; }

	  protected:
		template <typename... Es>
		explicit worker_job(const command_pkg& pkg, std::tuple<Es...> ctx = {}) : m_pkg(pkg), m_lctx(make_log_context(pkg, ctx)) {}

	  private:
		command_pkg m_pkg;
		log_context m_lctx;
		bool m_running = false;
		bool m_done = false;

		// Benchmarking
		std::chrono::steady_clock::time_point m_start_time;
		std::chrono::microseconds m_bench_sum_execution_time = {};
		size_t m_bench_sample_count = 0;
		std::chrono::microseconds m_bench_min = std::numeric_limits<std::chrono::microseconds>::max();
		std::chrono::microseconds m_bench_max = std::numeric_limits<std::chrono::microseconds>::min();
		tracy_async_lane m_tracy_lane;

		template <typename... Es>
		log_context make_log_context(const command_pkg& pkg, const std::tuple<Es...>& ctx = {}) {
			if(const auto tid = pkg.get_tid()) {
				return log_context{std::tuple_cat(std::tuple{"task", *tid, "job", pkg.cid}, ctx)};
			} else {
				return log_context{std::tuple_cat(std::tuple{"job", pkg.cid}, ctx)};
			}
		}

		// NOCOMMIT TODO Get rid of this package parameter API for all virtual functions
		virtual bool prepare(const command_pkg& pkg) { return true; }

		virtual bool execute(const command_pkg& pkg) = 0;

		/**
		 * Returns a human-readable job description for logging.
		 */
		virtual std::string get_description(const command_pkg& pkg) = 0;
	};

	class horizon_job : public worker_job {
	  public:
		horizon_job(const command_pkg& pkg, task_manager& tm) : worker_job(pkg), m_task_mngr(tm) { assert(pkg.get_command_type() == command_type::horizon); }

	  private:
		task_manager& m_task_mngr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class epoch_job : public worker_job {
	  public:
		epoch_job(const command_pkg& pkg, task_manager& tm) : worker_job(pkg), m_task_mngr(tm), m_action(std::get<epoch_data>(pkg.data).action) {
			assert(pkg.get_command_type() == command_type::epoch);
		}

		epoch_action get_epoch_action() const { return m_action; }

	  private:
		task_manager& m_task_mngr;
		epoch_action m_action;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * Informs the data_transfer_manager about the awaited push, then waits until the transfer has been received and completed.
	 */
	class await_push_job : public worker_job {
	  public:
		await_push_job(const command_pkg& pkg, buffer_transfer_manager& btm) : worker_job(pkg), m_btm(btm) {
			assert(pkg.get_command_type() == command_type::await_push);
		}

	  private:
		buffer_transfer_manager& m_btm;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> m_data_handle = nullptr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class push_job : public worker_job {
	  public:
		push_job(const command_pkg& pkg, buffer_transfer_manager& btm, buffer_manager& bm) : worker_job(pkg), m_btm(btm), m_buffer_mngr(bm) {
			assert(pkg.get_command_type() == command_type::push);
		}

	  private:
		buffer_transfer_manager& m_btm;
		buffer_manager& m_buffer_mngr;
		unique_frame_ptr<buffer_transfer_manager::data_frame> m_frame;
		async_event m_frame_transfer_event;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> m_data_handle = nullptr;

		bool prepare(const command_pkg& pkg) override;
		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class data_request_job : public worker_job {
	  public:
		data_request_job(const command_pkg& pkg, buffer_transfer_manager& btm) : worker_job(pkg), m_btm(btm) {
			assert(pkg.get_command_type() == command_type::data_request);
		}

	  private:
		[[maybe_unused]] buffer_transfer_manager& m_btm;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class reduction_job : public worker_job {
	  public:
		reduction_job(const command_pkg& pkg, reduction_manager& rm) : worker_job(pkg, std::tuple{"rid", std::get<reduction_data>(pkg.data).rid}), m_rm(rm) {
			assert(pkg.get_command_type() == command_type::reduction);
		}

	  private:
		reduction_manager& m_rm;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	// host-compute jobs, master-node tasks and collective host tasks
	class host_execute_job : public worker_job {
	  public:
		host_execute_job(const command_pkg& pkg, host_queue& queue, task_manager& tm, buffer_manager& bm)
		    : worker_job(pkg), m_queue(queue), m_task_mngr(tm), m_buffer_mngr(bm) {
			assert(pkg.get_command_type() == command_type::execution);
		}

	  private:
		host_queue& m_queue;
		task_manager& m_task_mngr;
		buffer_manager& m_buffer_mngr;
		std::future<host_queue::execution_info> m_future;
		bool m_submitted = false;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * TODO: Optimization opportunity: If we don't have any outstanding await-pushes, submitting the kernel to SYCL right away may be faster,
	 * as it can already start copying buffers to the device (i.e. let SYCL do the scheduling).
	 */
	class device_execute_job : public worker_job {
	  public:
		device_execute_job(const command_pkg& pkg, device_queue& queue, task_manager& tm, buffer_manager& bm, reduction_manager& rm, node_id local_nid)
		    : worker_job(pkg), m_queue(queue), m_task_mngr(tm), m_buffer_mngr(bm), m_reduction_mngr(rm), m_local_nid(local_nid) {
			assert(pkg.get_command_type() == command_type::execution);
		}

		device_id get_device_id() const { return m_queue.get_id(); }

	  private:
		device_queue& m_queue;
		task_manager& m_task_mngr;
		buffer_manager& m_buffer_mngr;
		reduction_manager& m_reduction_mngr;
		node_id m_local_nid;
		cl::sycl::event m_event;
		bool m_submitted = false;

		bool m_async_transfers_done = false;
		std::vector<closure_hydrator::NOCOMMIT_info> m_access_infos;
		std::vector<async_event> m_access_transfer_events;

		bool prepare(const command_pkg& pkg) override;
		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class gather_job : public worker_job {
	  public:
		gather_job(const command_pkg& pkg, buffer_manager& bm, node_id local_nid);

	  private:
		buffer_manager& m_buffer_mngr;
		node_id m_local_nid;
		bool m_started = false;
		collective_buffer m_send_buffer;
		collective_buffer m_recv_buffer;
		MPI_Request m_request = MPI_REQUEST_NULL;
		std::optional<async_event> m_d2d_gather;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class allgather_job : public worker_job {
	  public:
		allgather_job(const command_pkg& pkg, buffer_manager& bm, node_id local_nid);

	  private:
		buffer_manager& m_buffer_mngr;
		node_id m_local_nid;
		bool m_started = false;
		collective_buffer m_buffer;
		MPI_Request m_request = MPI_REQUEST_NULL;
		std::optional<async_event> m_d2d_allgather;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class broadcast_job : public worker_job {
	  public:
		broadcast_job(const command_pkg& pkg, buffer_manager& bm, node_id local_nid);

	  private:
		buffer_manager& m_buffer_mngr;
		node_id m_local_nid;
		bool m_started = false;
		collective_buffer m_buffer;
		MPI_Request m_request = MPI_REQUEST_NULL;
		std::optional<async_event> m_d2d_broadcast;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class scatter_job : public worker_job {
	  public:
		scatter_job(const command_pkg& pkg, buffer_manager& bm, node_id local_nid);

	  private:
		buffer_manager& m_buffer_mngr;
		node_id m_local_nid;
		bool m_started = false;
		collective_buffer m_send_buffer;
		collective_buffer m_recv_buffer;
		MPI_Request m_request = MPI_REQUEST_NULL;
		std::optional<async_event> m_d2d_scatter;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class alltoall_job : public worker_job {
	  public:
		alltoall_job(const command_pkg& pkg, buffer_manager& bm);

	  private:
		buffer_manager& m_buffer_mngr;
		bool m_started = false;
		collective_buffer m_send_buffer;
		collective_buffer m_recv_buffer;
		MPI_Request m_request = MPI_REQUEST_NULL;
		std::optional<async_event> m_d2d_alltoall;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

} // namespace detail
} // namespace celerity
