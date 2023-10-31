#pragma once

#include <cstddef>
#include <optional>

namespace celerity {
namespace detail {

	struct host_config {
		size_t node_count;
		size_t local_rank;
	};

	class config {
		friend struct config_testspy;

	  public:
		/**
		 * Initializes the @p config by parsing environment variables and passed arguments.
		 */
		config(int* argc, char** argv[]);

		const host_config& get_host_config() const { return m_host_cfg; }

		std::optional<bool> get_enable_device_profiling() const { return m_enable_device_profiling; }
		bool is_dry_run() const { return m_dry_run_nodes > 0; }
		bool is_recording() const { return m_recording; }
		int get_dry_run_nodes() const { return m_dry_run_nodes; }
		std::optional<int> get_horizon_step() const { return m_horizon_step; }
		std::optional<int> get_horizon_max_parallelism() const { return m_horizon_max_parallelism; }

	  private:
		host_config m_host_cfg;
		std::optional<bool> m_enable_device_profiling;
		size_t m_dry_run_nodes = 0;
		bool m_recording = false;
		std::optional<int> m_horizon_step;
		std::optional<int> m_horizon_max_parallelism;
	};

} // namespace detail
} // namespace celerity
