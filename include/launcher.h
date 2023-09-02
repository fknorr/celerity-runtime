
#pragma once

#include "ranges.h"
#include "sycl_wrappers.h"

#include <functional>
#include <variant>

#include <mpi.h>

namespace celerity::detail {

using sycl_kernel_launcher = std::function<void(sycl::handler& sycl_cgh, const subrange<3>& execution_range)>;
using host_task_launcher = std::function<void(const subrange<3>& execution_range, const range<3>& global_range, MPI_Comm comm)>;
using command_group_launcher = std::variant<sycl_kernel_launcher, host_task_launcher>;

} // namespace celerity::detail