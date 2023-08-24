
#pragma once

#include "ranges.h"
#include "sycl_wrappers.h"

#include <functional>

#include <mpi.h>

namespace celerity::detail {

using sycl_kernel_launcher = std::function<void(sycl::handler&, const subrange<3>&)>;
using host_task_launcher = std::function<void(const subrange<3>&, const range<3>&, MPI_Comm)>;

} // namespace celerity::detail