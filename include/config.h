#pragma once

#include <cstddef>
#include <optional>

#include "log.h"

namespace celerity {
namespace detail {

	struct host_config {
		size_t node_count;
		size_t local_rank;
	};

	enum class tracy_mode { off, fast, full };

	class config {
	  public:
		/**
		 * Initializes the @p config by parsing environment variables and passed arguments.
		 */
		config(int* argc, char** argv[]);

		log_level get_log_level() const { return m_log_lvl; }

		const host_config& get_host_config() const { return m_host_cfg; }

		bool should_enable_device_profiling() const { return m_enable_device_profiling.value_or(m_tracy_mode == tracy_mode::full); }
		bool is_dry_run() const { return m_dry_run_nodes > 0; }
		bool should_print_graphs() const { return m_should_print_graphs; }
		bool should_record() const {
			// Currently only graph printing requires recording, but this might change in the future.
			return m_should_print_graphs;
		}
		int get_dry_run_nodes() const { return m_dry_run_nodes; }
		std::optional<int> get_horizon_step() const { return m_horizon_step; }
		std::optional<int> get_horizon_max_parallelism() const { return m_horizon_max_parallelism; }
		tracy_mode get_tracy_mode() const { return m_tracy_mode; }

	  private:
		log_level m_log_lvl;
		host_config m_host_cfg;
		std::optional<bool> m_enable_device_profiling;
		size_t m_dry_run_nodes = 0;
		bool m_should_print_graphs = false;
		std::optional<int> m_horizon_step;
		std::optional<int> m_horizon_max_parallelism;
		tracy_mode m_tracy_mode = tracy_mode::off;
	};

} // namespace detail
} // namespace celerity
