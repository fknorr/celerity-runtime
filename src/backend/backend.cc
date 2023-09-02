#include "backend/backend.h"

namespace celerity::detail::backend {

sycl_event::sycl_event(std::vector<sycl::event> wait_list) : m_incomplete(std::move(wait_list)) {}

bool sycl_event::is_complete() const {
	const auto last_incomplete = std::remove_if(m_incomplete.begin(), m_incomplete.end(),
	    [](const sycl::event& evt) { return evt.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; });
	m_incomplete.erase(last_incomplete, m_incomplete.end());
	return m_incomplete.empty();
}

std::unique_ptr<event> launch_sycl_kernel(sycl::queue& queue, const sycl_kernel_launcher& launcher, const subrange<3>& execution_range) {
	auto event = queue.submit([&](sycl::handler& sycl_cgh) { launcher(sycl_cgh, execution_range); });
#if CELERITY_WORKAROUND(HIPSYCL)
	// hipSYCL does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
	// blocking the executor loop (see https://github.com/illuhad/hipSYCL/issues/599). Instead, we explicitly flush the queue to be able to continue
	// using our polling-based approach.
	queue.get_context().hipSYCL_runtime()->dag().flush_async();
#endif
	return std::make_unique<sycl_event>(std::vector{std::move(event)});
}

type get_type(const sycl::device& device) {
#if defined(__HIPSYCL__) && defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)
	if(device.get_backend() == sycl::backend::cuda) { return type::cuda; }
#endif
#if defined(__SYCL_COMPILER_VERSION) // DPC++ (TODO: This may break when using OpenSYCL w/ DPC++ as compiler)
	if(device.get_backend() == sycl::backend::ext_oneapi_cuda) { return type::cuda; }
#endif
	return type::unknown;
}

type get_effective_type(const sycl::device& device) {
	[[maybe_unused]] const auto b = get_type(device);

#if defined(CELERITY_DETAIL_BACKEND_CUDA_ENABLED)
	if(b == type::cuda) return b;
#endif

	return type::generic;
}

std::unique_ptr<queue> make_queue(type t, const std::vector<std::pair<device_id, sycl::device>>& devices) {
	assert(t != type::unknown);

#if defined(CELERITY_DETAIL_BACKEND_CUDA_ENABLED)
	if(t == type::cuda) return std::make_unique<cuda_queue>(devices);
#endif

	return std::make_unique<generic_queue>(devices);
}

} // namespace celerity::detail::backend
