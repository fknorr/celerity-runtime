#pragma once

#if CELERITY_ENABLE_TRACY

#include "print_utils.h"

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define CELERITY_DETAIL_TRACY_CAT_2(a, b) a##b
#define CELERITY_DETAIL_TRACY_CAT(a, b) CELERITY_DETAIL_TRACY_CAT_2(a, b)
#define CELERITY_DETAIL_TRACY_MAKE_SCOPED_NAME() CELERITY_DETAIL_TRACY_CAT(celerity_tracy_name_, __COUNTER__)

#define CELERITY_DETAIL_TRACY_SCOPED_ZONE_2(scoped_name, color, ...)                                                                                           \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	ZoneScopedC(color);                                                                                                                                        \
	ZoneName(scoped_name.data(), scoped_name.size());

#define CELERITY_DETAIL_TRACY_SCOPED_ZONE(color, ...) CELERITY_DETAIL_TRACY_SCOPED_ZONE_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_NAME(), color, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ZONE_TEXT_2(scoped_name, ...)                                                                                                    \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	ZoneText(scoped_name.data(), scoped_name.size());

#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...) CELERITY_DETAIL_TRACY_ZONE_TEXT_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_NAME(), __VA_ARGS__)

#else

#define CELERITY_DETAIL_TRACY_SCOPED_ZONE(color, ...)
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)

#endif
