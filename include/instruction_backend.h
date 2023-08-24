#pragma once

#include <type_traits>

namespace celerity::detail {

enum class instruction_backend {
	host,
	mpi,
	sycl,
	cuda,
};

inline constexpr std::underlying_type_t<instruction_backend> num_instruction_backends = 4;

}
