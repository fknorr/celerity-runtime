#pragma once

namespace celerity::detail {

enum class instruction_backend {
	host,
	sycl,
	cuda,
};

}
