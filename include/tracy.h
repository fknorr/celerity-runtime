#pragma once

#if CELERITY_TRACY_SUPPORT

#include "config.h"

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

namespace celerity::detail::tracy_detail {

// This is intentionally not an atomic, as parts of Celerity (= live_executor) expect it not to change after runtime startup
inline tracy_mode g_tracy_mode = tracy_mode::off;

/// Tracy is enabled via environment variable, either in fast or full mode.
inline bool is_enabled() { return g_tracy_mode != tracy_mode::off; }

/// Tracy is enabled via environment variable, in full mode.
inline bool is_enabled_full() { return g_tracy_mode == tracy_mode::full; }

template <typename Value>
struct plot {
	const char* identifier = nullptr;
	Value last_value = 0;

	explicit plot(const char* const identifier) : identifier(identifier) {
		TracyPlot(identifier, static_cast<Value>(0));
		TracyPlotConfig(identifier, tracy::PlotFormatType::Number, true /* step */, true /* fill*/, 0);
	}

	void update(const Value value_in) {
		const auto value = static_cast<Value>(value_in);
		if(value != last_value) {
			TracyPlot(identifier, value);
			last_value = value;
		}
	}
};

template <typename... FmtParams>
const char* make_thread_name(fmt::format_string<FmtParams...> fmt_string, const FmtParams&... fmt_args) {
	// Thread and fiber name pointers must remain valid for the duration of the program, so we intentionally leak them
	const auto size = fmt::formatted_size(fmt_string, fmt_args...);
	const auto name = static_cast<char*>(malloc(size + 1));
	fmt::format_to(name, fmt_string, fmt_args...);
	name[size] = 0;
	return name;
}

/// Helper to pass fmt::formatted strings to Tracy's (pointer, size) functions.
template <typename ApplyFn, typename... FmtParams, std::enable_if_t<(sizeof...(FmtParams) > 0), int> = 0>
void apply_string(const ApplyFn& apply, fmt::format_string<FmtParams...> fmt_string, const FmtParams&... fmt_args) {
	apply(fmt::format(fmt_string, fmt_args...));
}

template <typename ApplyFn, typename... FmtParams>
void apply_string(const ApplyFn& apply, std::string_view string) {
	apply(string);
}

} // namespace celerity::detail::tracy_detail

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...) __VA_ARGS__

#else

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...)

#endif


#define CELERITY_DETAIL_IF_TRACY_ENABLED(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled()) { __VA_ARGS__; })
#define CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled_full()) { __VA_ARGS__; })

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME, ...)                                                                                                \
	CELERITY_DETAIL_IF_TRACY_SUPPORTED(ZoneNamedNC(___tracy_scoped_zone, TAG, ::tracy::Color::COLOR_NAME, ::celerity::detail::tracy_detail::is_enabled()));    \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& n) { ZoneName(n.data(), n.size()); }, __VA_ARGS__))

#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)                                                                                                                   \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& t) { ZoneText(t.data(), t.size()); }, __VA_ARGS__))

#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME(NAME) CELERITY_DETAIL_IF_TRACY_ENABLED(::tracy::SetThreadName(NAME))
