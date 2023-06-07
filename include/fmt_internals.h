#pragma once

#include <fmt/core.h>

namespace celerity::detail {

enum class instruction_backend;

}

namespace fmt {

template <>
struct formatter<celerity::detail::instruction_backend> {
	auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.begin(); }
	format_context::iterator format(celerity::detail::instruction_backend, format_context& ctx) const;
};

}