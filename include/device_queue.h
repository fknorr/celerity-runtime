#pragma once

#include <memory>
#include <utility>

#include <CL/sycl.hpp>

#include "backend/backend.h"
#include "config.h"
#include "device_selection.h"
#include "log.h"
#include "types.h"

namespace celerity {
namespace detail {

	struct device_allocation {
		void* ptr = nullptr;
		size_t size_bytes = 0;
	};

	class allocation_error : public std::runtime_error {
	  public:
		allocation_error(const std::string& msg) : std::runtime_error(msg) {}
	};

	/**
	 * The @p device_queue wraps the actual SYCL queue and is used to submit kernels.
	 */
	class device_queue {
	  public:
		device_queue(device_id did, memory_id mid) : m_did(did), m_mid(mid) {}

		void init(const config& cfg, sycl::device device);

		device_id get_id() const { return m_did; }

		memory_id get_memory_id() const { return m_mid; }

		/**
		 * @brief Executes the kernel associated with task @p ctsk over the chunk @p chnk.
		 */
		template <typename Fn>
		cl::sycl::event submit(Fn&& fn) {
			auto evt = m_sycl_queue->submit([fn = std::forward<Fn>(fn)](cl::sycl::handler& sycl_handler) { fn(sycl_handler); });
#if CELERITY_WORKAROUND(HIPSYCL)
			// hipSYCL does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
			// blocking the executor loop (see https://github.com/illuhad/hipSYCL/issues/599). Instead, we explicitly flush the queue to be able to continue
			// using our polling-based approach.
			m_sycl_queue->get_context().hipSYCL_runtime()->dag().flush_async();
#endif
			return evt;
		}

		template <typename T>
		[[nodiscard]] device_allocation malloc(const size_t count) {
			const size_t size_bytes = count * sizeof(T);
			assert(m_sycl_queue != nullptr);
			assert(m_global_mem_allocated_bytes + size_bytes < m_global_mem_total_size_bytes);
			CELERITY_DEBUG("Allocating {} bytes on device {} (memory {})", size_bytes, m_did, m_mid);
			T* ptr = nullptr;
			try {
				ptr = sycl::aligned_alloc_device<T>(alignof(T), count, *m_sycl_queue);
			} catch(sycl::exception& e) {
				CELERITY_CRITICAL("sycl::aligned_alloc_device failed with exception: {}", e.what());
				ptr = nullptr;
			}
			if(ptr == nullptr) {
				throw allocation_error(
				    fmt::format("Allocation of {} bytes on device {} (memory {}) failed; likely out of memory. Currently allocated: {} out of {} bytes.",
				        count * sizeof(T), m_did, m_mid, m_global_mem_allocated_bytes, m_global_mem_total_size_bytes));
			}
			m_global_mem_allocated_bytes += size_bytes;
			return device_allocation{ptr, size_bytes};
		}

		void free(device_allocation alloc) {
			assert(m_sycl_queue != nullptr);
			assert(alloc.size_bytes <= m_global_mem_allocated_bytes);
			assert(alloc.ptr != nullptr || alloc.size_bytes == 0);
			CELERITY_DEBUG("Freeing {} bytes on device {} (memory {})", alloc.size_bytes, m_did, m_mid);
			if(alloc.size_bytes != 0) { sycl::free(alloc.ptr, *m_sycl_queue); }
			m_global_mem_allocated_bytes -= alloc.size_bytes;
		}

		size_t get_global_memory_total_size_bytes() const { return m_global_mem_total_size_bytes; }

		size_t get_global_memory_allocated_bytes() const { return m_global_mem_allocated_bytes; }

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() { m_sycl_queue->wait_and_throw(); }

		/**
		 * @brief Returns whether device profiling is enabled.
		 */
		bool is_profiling_enabled() const { return m_device_profiling_enabled; }

		cl::sycl::queue& get_sycl_queue() const {
			assert(m_sycl_queue != nullptr);
			return *m_sycl_queue;
		}

	  private:
		device_id m_did;
		memory_id m_mid;
		size_t m_global_mem_total_size_bytes = 0;
		size_t m_global_mem_allocated_bytes = 0;
		std::unique_ptr<cl::sycl::queue> m_sycl_queue;
		bool m_device_profiling_enabled = false;

		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

} // namespace detail
} // namespace celerity
