
#pragma once

#include "host_queue.h"
#include "ranges.h"
#include "sycl_wrappers.h"

#include <functional>
#include <future>
#include <variant>

#include <mpi.h>

namespace celerity::detail {

using sycl_kernel_launcher = std::function<void(
    sycl::handler& sycl_cgh, const subrange<3>& execution_range, const std::vector<void*>& reduction_ptrs, const bool is_reduction_initializer)>;
using host_task_launcher = std::function<std::future<host_queue::execution_info>(host_queue& q, const subrange<3>& execution_range, MPI_Comm mpi_comm)>;
using command_group_launcher = std::variant<sycl_kernel_launcher, host_task_launcher>;

} // namespace celerity::detail