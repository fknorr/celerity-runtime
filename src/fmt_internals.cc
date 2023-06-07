#include "fmt_internals.h"

#include "instruction_backend.h"

using namespace celerity::detail;

namespace fmt {

format_context::iterator fmt::formatter<celerity::detail::instruction_backend>::format(
    const celerity::detail::instruction_backend backend, format_context& ctx) const {
	switch(backend) {
	case celerity::detail::instruction_backend::host: return format_to(ctx.out(), "Host");
	case celerity::detail::instruction_backend::sycl: return format_to(ctx.out(), "SYCL");
	case celerity::detail::instruction_backend::cuda: return format_to(ctx.out(), "CUDA");
	}
};

} // namespace fmt
