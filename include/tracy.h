#pragma once

#include <cassert>
#include <optional>

#include <spdlog/fmt/fmt.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

namespace celerity::detail {

// Workaround for https://github.com/wolfpld/tracy/issues/426
class tracy_async_lane {
  public:
	void initialize() {
		assert(!m_started);
		m_lane_id = get_free_lane();
		m_started = true;
	}

	void destroy() {
		assert(m_started);
		TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
		if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
		return_lane(m_lane_id);
		TracyFiberLeave;
		m_started = false;
	}

	void activate() {
		assert(m_started);
		TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
	}

	void deactivate() {
		assert(m_started);
		TracyFiberLeave;
	}

	void begin_phase(const std::string& name, const std::string& description, const tracy::Color::ColorType color) {
		assert(m_started);
		TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
		if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
		TracyCZone(t_ctx, true);
		TracyCZoneName(t_ctx, name.c_str(), name.size());
		TracyCZoneText(t_ctx, description.c_str(), description.size());
		TracyCZoneColor(t_ctx, color);
		TracyFiberLeave;
		m_current_zone = t_ctx;
	}

  private:
	bool m_started = false;
	size_t m_lane_id = -1;
	std::optional<TracyCZoneCtx> m_current_zone;

	struct lane_info {
		std::unique_ptr<std::string> name;
		bool is_free;
	};

	inline static std::vector<lane_info> tracy_lanes = {};

	static size_t get_free_lane() {
		for(size_t lane = 0; lane < tracy_lanes.size(); ++lane) {
			if(tracy_lanes[lane].is_free) {
				tracy_lanes[lane].is_free = false;
				return lane;
			}
		}
		tracy_lanes.push_back({std::make_unique<std::string>(fmt::format("celerity async {:02}", tracy_lanes.size())), false});
		return tracy_lanes.size() - 1;
	}

	static void return_lane(size_t lane_id) {
		assert(!tracy_lanes.at(lane_id).is_free);
		tracy_lanes[lane_id].is_free = true;
	}
};

} // namespace celerity::detail
