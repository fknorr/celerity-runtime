#pragma once

#include <memory>
#include <type_traits>

#include "device_queue.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {

struct [[deprecated("This tag type is no longer required to capture by reference")]] allow_by_ref_t{};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
inline constexpr allow_by_ref_t allow_by_ref{};
#pragma GCC diagnostic pop

class distr_queue {
  public:
	distr_queue() { init(detail::auto_select_devices{}); }

	/**
	 * @brief Creates a distr_queue and instructs it to use a particular set of devices.
	 *
	 * @param devices The devices to be used on the current node. This can vary between nodes.
	 *                If there are multiple nodes running on the same host, the list of devices must be the same across nodes on the same host.
	 */
	distr_queue(const std::vector<sycl::device>& devices) {
		if(detail::runtime::has_instance()) { throw std::runtime_error("Passing explicit device list not possible, runtime has already been initialized."); }
		init(devices);
	}

	/**
	 * @brief Creates a distr_queue and instructs it to use a particular set of devices.
	 *
	 * @param device_selector The device selector to be used on the current node. This can vary between nodes.
	 *                        If there are multiple nodes running on the same host, the selector must be the same across nodes on the same host.
	 */
	template <typename DeviceSelector>
	distr_queue(const DeviceSelector& device_selector) {
		if(detail::runtime::has_instance()) {
			throw std::runtime_error("Passing explicit device selector not possible, runtime has already been initialized.");
		}
		init(device_selector);
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 */
	template <typename CGF>
	[[deprecated("This overload is no longer required to capture by reference")]] void submit(allow_by_ref_t /* tag */, CGF cgf) {
		submit(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 */
	template <typename CGF>
	void submit(CGF cgf) { // NOLINT(readability-convert-member-functions-to-static)
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		detail::runtime::get_instance().get_task_manager().submit_command_group(std::move(cgf));
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes across the entire cluster. }
	 */
	void slow_full_sync() { detail::runtime::get_instance().sync(detail::epoch_action::barrier); } // NOLINT(readability-convert-member-functions-to-static)

  private:
	struct tracker {
		tracker() { detail::runtime::get_instance().create_queue(); }

		tracker(const tracker&) = delete;
		tracker(tracker&&) = delete;
		tracker& operator=(const tracker&) = delete;
		tracker& operator=(tracker&&) = delete;

		~tracker() {
			// The destructor of the last queue handle must wait for all submitted work to finish to guarantee implicit synchronization e.g. around host_task
			// ref-captures. Notifying the runtime of queue destruction might destroy the runtime instance itself, which will issue and wait on the shutdown
			// epoch, guaranteeing that all previously submitted work has completed.
			detail::runtime::get_instance().destroy_queue();

			// If any buffers or host objects outlive the queue, the runtime will delay its destruction (and thus the shutdown epoch) to still be able to
			// issue the appropriate instructions for buffer and host object deallocation. In that case, we insert and wait on another local epoch to guarantee
			// synchronization. Any later cleanup instructions will be inserted by the scheduler between this and the shutdown epoch.
			if(detail::runtime::has_instance()) { detail::runtime::get_instance().sync(detail::epoch_action::none); }
		}
	};

	std::shared_ptr<tracker> m_tracker;

	void init(const detail::devices_or_selector& devices_or_selector) {
		if(!detail::runtime::has_instance()) { detail::runtime::init(nullptr, nullptr, devices_or_selector); }
		m_tracker = std::make_shared<tracker>();
	}
};

} // namespace celerity
