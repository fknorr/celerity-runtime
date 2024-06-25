#pragma once

#if CELERITY_ENABLE_TRACY

#include <chrono>
#include <cstring>
#include <optional>

#include "print_utils.h"

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define CELERITY_DETAIL_TRACY_CAT_2(a, b) a##b
#define CELERITY_DETAIL_TRACY_CAT(a, b) CELERITY_DETAIL_TRACY_CAT_2(a, b)
#define CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tag) CELERITY_DETAIL_TRACY_CAT(tag, __COUNTER__)

#define CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE 128

#define CELERITY_DETAIL_TRACY_FORMAT(SCOPED_BUF, SCOPED_LENGTH, ...)                                                                                           \
	char SCOPED_BUF[CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE];                                                                                                   \
	::size_t SCOPED_LENGTH = ::fmt::format_to_n(SCOPED_BUF, CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE, __VA_ARGS__).size;                                         \
	if(SCOPED_LENGTH > CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE) {                                                                                               \
		::memset(SCOPED_BUF + CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE - 3, '.', 3);                                                                             \
		SCOPED_LENGTH = CELERITY_DETAIL_TRACY_TEXT_BUFFER_SIZE;                                                                                                \
	}

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED_2(SCOPED_NAME, SCOPED_NAME_LENGTH, TAG, COLOR_NAME, ...)                                                             \
	ZoneScopedNC(TAG, ::tracy::Color::COLOR_NAME);                                                                                                             \
	{                                                                                                                                                          \
		CELERITY_DETAIL_TRACY_FORMAT(SCOPED_NAME, SCOPED_NAME_LENGTH, __VA_ARGS__)                                                                             \
		ZoneName(SCOPED_NAME, SCOPED_NAME_LENGTH);                                                                                                             \
	}

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME, ...)                                                                                                \
	CELERITY_DETAIL_TRACY_ZONE_SCOPED_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_name_),                                                        \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_name_length_), TAG, COLOR_NAME, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ZONE_TEXT_2(SCOPED_NAME, SCOPED_NAME_LENGTH, ...)                                                                                \
	{                                                                                                                                                          \
		CELERITY_DETAIL_TRACY_FORMAT(SCOPED_NAME, SCOPED_NAME_LENGTH, __VA_ARGS__)                                                                             \
		ZoneText(SCOPED_NAME, SCOPED_NAME_LENGTH);                                                                                                             \
	}

#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)                                                                                                                   \
	CELERITY_DETAIL_TRACY_ZONE_TEXT_2(                                                                                                                         \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_text_), CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_text_length_), __VA_ARGS__)

namespace celerity::detail::tracy_detail {

struct async_fiber {
	const char* thread_name; // to call tracy_detail::get_thread from a release_lane
	size_t index;
	std::string fiber_name;
	std::optional<TracyCZoneCtx> current_zone;
	std::chrono::steady_clock::time_point current_zone_begin;
	async_fiber(const char* const thread_name, const size_t index)
	    : thread_name(thread_name), index(index), fiber_name(fmt::format("{} async ({})", thread_name, index)) {}
};

using async_lane = async_fiber*;

async_lane acquire_lane(const char* thread_name);
void release_lane(async_lane lane);

struct async_suspend_guard {
	async_suspend_guard() = default;

	async_suspend_guard(const async_suspend_guard&) = delete;
	async_suspend_guard(async_suspend_guard&&) = delete;
	async_suspend_guard& operator=(const async_suspend_guard&) = delete;
	async_suspend_guard& operator=(async_suspend_guard&&) = delete;

	~async_suspend_guard() { TracyFiberLeave; }
};

struct async_release_guard {
	explicit async_release_guard(const async_lane lane) : lane(lane) {}

	async_release_guard(const async_release_guard&) = delete;
	async_release_guard(async_release_guard&&) = delete;
	async_release_guard& operator=(const async_release_guard&) = delete;
	async_release_guard& operator=(async_release_guard&&) = delete;

	~async_release_guard() {
		if(lane->current_zone.has_value()) {
			TracyCZoneEnd(*lane->current_zone);
			lane->current_zone = ::std::nullopt;
		}
		TracyFiberLeave;
		::celerity::detail::tracy_detail::release_lane(lane);
	}

	async_lane lane;
};

#define CELERITY_DETAIL_TRACY_ASYNC_LANE(LANE) ::celerity::detail::tracy_detail::async_lane LANE = nullptr;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_START_SCOPED_2(SCOPED_CTX, SCOPED_NAME, SCOPED_NAME_LENGTH, SCOPED_GUARD, THREAD_NAME, LANE, TAG, COLOR_NAME, ...)    \
	const auto LANE = ::celerity::detail::tracy_detail::acquire_lane(THREAD_NAME);                                                                             \
	TracyFiberEnter((LANE)->fiber_name.c_str());                                                                                                               \
	if((LANE)->current_zone.has_value()) {                                                                                                                     \
		TracyCZoneEnd(*(LANE)->current_zone);                                                                                                                  \
		(LANE)->current_zone = ::std::nullopt;                                                                                                                 \
	}                                                                                                                                                          \
	TracyCZoneNC(SCOPED_CTX, (TAG), ::tracy::Color::COLOR_NAME, true);                                                                                         \
	{                                                                                                                                                          \
		CELERITY_DETAIL_TRACY_FORMAT(SCOPED_NAME, SCOPED_NAME_LENGTH, __VA_ARGS__)                                                                             \
		TracyCZoneName(SCOPED_CTX, SCOPED_NAME, SCOPED_NAME_LENGTH);                                                                                           \
	}                                                                                                                                                          \
	(LANE)->current_zone = SCOPED_CTX;                                                                                                                         \
	(LANE)->current_zone_begin = ::std::chrono::steady_clock::now();                                                                                           \
	::celerity::detail::tracy_detail::async_suspend_guard SCOPED_GUARD;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_START_SCOPED(OUT_LANE, THREAD_NAME, TAG, COLOR_NAME, ...)                                                             \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_START_SCOPED_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_ctx_),                                                  \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_name_), CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_name_length_),                 \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_suspend_guard_), THREAD_NAME, OUT_LANE, TAG, COLOR_NAME, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED_2(LANE, SCOPED_GUARD)                                                                                   \
	TracyFiberEnter((LANE)->fiber_name.c_str());                                                                                                               \
	::celerity::detail::tracy_detail::async_suspend_guard SCOPED_GUARD;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED(LANE)                                                                                                   \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED_2(LANE, CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_suspend_guard_))

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_FINISH_SCOPED_2(LANE, SCOPED_GUARD)                                                                                   \
	TracyFiberEnter((LANE)->fiber_name.c_str());                                                                                                               \
	::celerity::detail::tracy_detail::async_release_guard SCOPED_GUARD(LANE);

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_FINISH_SCOPED(LANE)                                                                                                   \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_FINISH_SCOPED_2(LANE, CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_release_guard_))

#define CELERITY_DETAIL_TRACY_ASYNC_ELAPSED_TIME_SECONDS(LANE)                                                                                                 \
	(::std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - (LANE)->current_zone_begin).count())

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT_2(SCOPED_TEXT, SCOPED_TEXT_LENGTH, LANE, ...)                                                                    \
	{                                                                                                                                                          \
		CELERITY_DETAIL_TRACY_FORMAT(SCOPED_TEXT, SCOPED_TEXT_LENGTH, __VA_ARGS__)                                                                             \
		TracyCZoneText((LANE)->current_zone.value(), SCOPED_TEXT, SCOPED_TEXT_LENGTH);                                                                         \
	}

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(LANE, ...)                                                                                                       \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_text_),                                                    \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_zone_text_length_), LANE, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME(NAME) ::tracy::SetThreadName(NAME);

#define CELERITY_DETAIL_IF_TRACY(...) __VA_ARGS__

} // namespace celerity::detail::tracy_detail

#else

#define CELERITY_DETAIL_TRACY_ASYNC_LANE(...)
#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_START_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_FINISH_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_ELAPSED_TIME_SECONDS(...) (0.0)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_END(...)
#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME(...)
#define CELERITY_DETAIL_IF_TRACY(...)

#endif
