#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fmt/format.h>
#include <string_view>
#include <vector>

namespace celerity::detail {

struct dumb_profiler {
	enum class event : uint8_t { idle, busy };

	struct region {
		std::chrono::steady_clock::time_point start;
		std::chrono::steady_clock::time_point stop;
		std::string tag;
	};

	std::vector<region> log;
	std::optional<region> active_region;

	void busy(const std::string_view tag = {}) {
		if(active_region.has_value() && active_region->tag == tag) return;
		if(active_region.has_value()) { idle(); }
		active_region = region{.start = std::chrono::steady_clock::now(), .stop = {}, .tag = std::string(tag)};
	}

	void idle() {
		if(!active_region.has_value()) return;
		active_region->stop = std::chrono::steady_clock::now();
		log.push_back(*active_region);
		active_region.reset();
	}

	void dump(const char* path) {
		FILE* f = fopen(path, "w");
		fmt::print(f, "start,stop,tag\n");
		for(const auto& [start, stop, tag] : log) {
			fmt::print(f, "{},{},{}\n", start.time_since_epoch().count(), stop.time_since_epoch().count(), tag);
		}
		fclose(f);
	}
};

} // namespace celerity::detail
