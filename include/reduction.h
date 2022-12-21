#pragma once

#include "buffer.h"
#include "buffer_manager.h"
#include "mpi_support.h"
#include "sycl_wrappers.h"

#include <type_traits>


namespace celerity {

using sycl::bit_and;
using sycl::bit_or;
using sycl::bit_xor;
using sycl::logical_and;
using sycl::logical_or;
using sycl::maximum;
using sycl::minimum;
using sycl::multiplies;
using sycl::plus;

template <typename Fn, typename T>
struct known_identity;

template <typename Fn, typename T>
inline constexpr T known_identity_v = known_identity<Fn, T>::value;

template <typename Fn, typename T>
struct has_known_identity;

template <typename Fn, typename T>
inline constexpr bool has_known_identity_v = has_known_identity<Fn, T>::value;

#define CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(fn, identity)                                                                                                   \
	template <typename U, typename T>                                                                                                                          \
	struct known_identity<fn<U>, T> {                                                                                                                          \
		inline constexpr static T value = identity;                                                                                                            \
	};                                                                                                                                                         \
	template <typename U, typename T>                                                                                                                          \
	struct has_known_identity<fn<U>, T> : std::true_type {};

#define CELERITY_DETAIL_KNOWN_IDENTITY(fn, t, identity)                                                                                                        \
	template <typename U>                                                                                                                                      \
	struct known_identity<fn<U>, t> {                                                                                                                          \
		inline constexpr static t value = identity;                                                                                                            \
	};                                                                                                                                                         \
	template <typename U>                                                                                                                                      \
	struct has_known_identity<fn<U>, t> : std::true_type {};

CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(plus, T(0))
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(multiplies, T(1))
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(bit_and, ~T{})
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(bit_or, T{})
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(bit_xor, T{})
CELERITY_DETAIL_KNOWN_IDENTITY(logical_and, bool, true);
CELERITY_DETAIL_KNOWN_IDENTITY(logical_or, bool, false);
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(maximum, std::numeric_limits<T>::min());
CELERITY_DETAIL_KNOWN_IDENTITY_GENERIC(minimum, std::numeric_limits<T>::max())
CELERITY_DETAIL_KNOWN_IDENTITY(maximum, float, -std::numeric_limits<float>::infinity());
CELERITY_DETAIL_KNOWN_IDENTITY(minimum, float, std::numeric_limits<float>::infinity())
CELERITY_DETAIL_KNOWN_IDENTITY(maximum, double, -std::numeric_limits<double>::infinity());
CELERITY_DETAIL_KNOWN_IDENTITY(minimum, double, std::numeric_limits<double>::infinity())
CELERITY_DETAIL_KNOWN_IDENTITY(maximum, long double, -std::numeric_limits<long double>::infinity());
CELERITY_DETAIL_KNOWN_IDENTITY(minimum, long double, std::numeric_limits<long double>::infinity())

} // namespace celerity

namespace celerity::detail {

template <typename Fn, typename T>
struct bind_function_object;

template <template <typename T> typename FnTemplate, typename T>
struct bind_function_object<FnTemplate<void>, T> {
	using type = FnTemplate<T>;
};

template <template <typename T> typename FnTemplate, typename T>
struct bind_function_object<FnTemplate<T>, T> {
	using type = FnTemplate<T>;
};

template <template <typename T> typename FnTemplate, typename U, typename T>
struct bind_function_object<FnTemplate<U>, T> {
	static_assert(constexpr_false<U>, "Operand type (e.g. int) does not match argument type of the function object (e.g. plus<float>).");
};

template <typename Fn, typename T>
using bind_function_object_v = typename bind_function_object<std::remove_cv_t<Fn>, T>::type;


template <typename, typename, int>
class reduction_init_kernel;

struct buffer_reduction_v2 {
	std::function<void(sycl::queue& q, buffer_manager& bm)> submit_init_kernel;
	MPI_Datatype mpi_data_type;
	MPI_Op mpi_op;

	template <typename Fn, typename T, int Dims>
	static buffer_reduction_v2 bind(const celerity::buffer<T, Dims>& buffer) {
		buffer_reduction_v2 br;
		br.submit_init_kernel = [buffer /* increment refcount */](sycl::queue& q, buffer_manager& bm, const subrange<3> celerity_sr) {
			const auto range = celerity_sr.range;
			const auto celerity_offset = celerity_sr.offset;
			const auto access_info =
			    bm.get_device_buffer<T, Dims>(detail::get_buffer_id(buffer), access_mode::discard_write, range_cast<3>(range), range_cast<3>(celerity_offset));
			const auto sycl_offset = {celerity_offset - access_info.offset};
			q.submit([&](sycl::handler& cgh) {
				sycl::accessor acc(buffer, cgh, range, sycl_offset);
				cgh.parallel_for<reduction_init_kernel<Fn, T, Dims>>(
				    range, [=](sycl::item<Dims> item) { acc[sycl_offset + item.get_id()] = known_identity_v<Fn, T>; });
			});
		};
		br.mpi_data_type = mpi_support::builtin_data_type<T>::value;
		br.mpi_op = mpi_support::builtin_op<Fn>::value;
		return br;
	}
};

} // namespace celerity::detail
