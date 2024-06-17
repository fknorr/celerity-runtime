#pragma once

#include "async_event.h"

#include "backend/backend.h"

#include <fmt/format.h>

namespace celerity::detail::sycl_backend_detail {

void flush_queue(sycl::queue& queue);

async_event launch_kernel(
    sycl::queue& queue, const device_kernel_launcher& launch, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs, bool enable_profiling);

void handle_errors(const sycl::exception_list& errors);

} // namespace celerity::detail::sycl_backend_detail

namespace celerity::detail {

class sycl_event final : public async_event_impl {
  public:
	sycl_event() = default;
	sycl_event(sycl::event event, bool profiling_enabled) : m_event(std::move(event)), m_profiling_enabled(profiling_enabled) {}

	bool is_complete() override;

	std::optional<std::chrono::nanoseconds> get_native_execution_time() override;

  private:
	sycl::event m_event;
	bool m_profiling_enabled;
};

class sycl_backend : public backend {
  public:
	explicit sycl_backend(const std::vector<sycl::device>& devices, bool enable_profiling);
	sycl_backend(const sycl_backend&) = delete;
	sycl_backend(sycl_backend&&) = delete;
	sycl_backend& operator=(const sycl_backend&) = delete;
	sycl_backend& operator=(sycl_backend&&) = delete;
	~sycl_backend() override;

	const system_info& get_system_info() const override;

	void* debug_alloc(size_t size) override;

	void debug_free(void* ptr) override;

	async_event enqueue_host_alloc(size_t size, size_t alignment) override;

	async_event enqueue_device_alloc(device_id device, size_t size, size_t alignment) override;

	async_event enqueue_host_free(void* ptr) override;

	async_event enqueue_device_free(device_id device, void* ptr) override;

	async_event enqueue_host_function(size_t host_lane, std::function<void()> fn) override;

	async_event enqueue_device_kernel(device_id device, size_t device_lane, const device_kernel_launcher& launcher, const box<3>& execution_range,
	    const std::vector<void*>& reduction_ptrs) override;

	async_event enqueue_host_copy(size_t host_lane, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
	    const region<3>& copy_region, const size_t elem_size) override;

  protected:
	sycl::queue& get_device_queue(device_id device, size_t lane);

	system_info& get_system_info();

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

class sycl_generic_backend final : public sycl_backend {
  public:
	using sycl_backend::sycl_backend;

	async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override;
};

#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
class sycl_cuda_backend final : public sycl_backend {
  public:
	sycl_cuda_backend(const std::vector<sycl::device>& devices, bool enable_profiling);

	async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override;
};
#endif

enum class sycl_backend_type { generic, cuda };

struct sycl_backend_enumerator {
	using backend_type = sycl_backend_type;
	using device_type = sycl::device;

	std::vector<backend_type> compatible_backends(const sycl::device& device) const;

	std::vector<backend_type> available_backends() const;

	bool is_specialized(backend_type type) const;

	int get_priority(backend_type type) const;
};

std::unique_ptr<backend> make_sycl_backend(const sycl_backend_type type, const std::vector<sycl::device>& devices, bool enable_profiling);

} // namespace celerity::detail
