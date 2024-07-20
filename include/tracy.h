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

#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME(NAME) ::tracy::SetThreadName(NAME);

#define CELERITY_DETAIL_IF_TRACY(...) __VA_ARGS__

#else

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)
#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME(...)
#define CELERITY_DETAIL_IF_TRACY(...)

#endif
