#include "local_devices.h"

#include "device_selection.h"
#include "utils.h"

namespace celerity::detail {

void local_devices::init(const config& cfg, const devices_or_selector& dos) {
	assert(!m_is_initialized);
	m_is_initialized = true;

	const auto devices = std::visit([&cfg](const auto& value) { return pick_devices(cfg, value, sycl::platform::get_platforms()); }, dos);
	for(device_id did = 0; did < devices.size(); ++did) {
		m_device_queues.emplace_back(did, get_memory_id(did));
		m_device_queues.back().init(cfg, devices[did]);
	}
}

void local_devices::wait_all() {
	for(auto& dq : m_device_queues) {
		dq.wait();
	}
	m_host_queue.wait();
}

} // namespace celerity::detail