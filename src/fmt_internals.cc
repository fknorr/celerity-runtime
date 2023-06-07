#include "fmt_internals.h"

#include "instruction_backend.h"

using namespace celerity::detail;

namespace fmt {

format_context::iterator fmt::formatter<celerity::detail::instruction_backend>::format(
    const instruction_backend backend, format_context& ctx) const {
	switch(backend) {
	case instruction_backend::host: return format_to(ctx.out(), "Host");
	case instruction_backend::mpi: return format_to(ctx.out(), "MPI");
	case instruction_backend::sycl: return format_to(ctx.out(), "SYCL");
	case instruction_backend::cuda: return format_to(ctx.out(), "CUDA");
	}
};

} // namespace fmt
