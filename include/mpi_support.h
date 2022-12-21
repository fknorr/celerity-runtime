#pragma once

#include <mpi.h>

#include "sycl_wrappers.h"

namespace celerity::detail::mpi_support {

constexpr int TAG_CMD = 0;
constexpr int TAG_DATA_TRANSFER = 1;
constexpr int TAG_TELEMETRY = 2;

class data_type {
  public:
	explicit data_type(MPI_Datatype dt) : m_dt(dt) {}
	data_type(const data_type&) = delete;
	data_type& operator=(const data_type&) = delete;
	~data_type() { MPI_Type_free(&m_dt); }

	operator MPI_Datatype() const { return m_dt; }

  private:
	MPI_Datatype m_dt;
};


template <typename T>
struct builtin_data_type;

#define CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(cpp_type, mpi_datatype)                                                                                  \
	template <>                                                                                                                                                \
	struct builtin_data_type<cpp_type> {                                                                                                                       \
		inline static const MPI_Datatype value = mpi_datatype;                                                                                                 \
	};

CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(std::byte, MPI_BYTE)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(char, MPI_CHAR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(wchar_t, MPI_WCHAR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(signed char, MPI_SIGNED_CHAR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(unsigned char, MPI_UNSIGNED_CHAR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(short, MPI_SHORT)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(unsigned short, MPI_UNSIGNED_SHORT)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(int, MPI_INT)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(unsigned, MPI_UNSIGNED)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(long, MPI_LONG)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(unsigned long, MPI_UNSIGNED_LONG)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(long long, MPI_LONG_LONG)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(float, MPI_FLOAT)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(double, MPI_DOUBLE)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_DATA_TYPE(long double, MPI_LONG_DOUBLE)


template <typename Fn>
struct builtin_op;

#define CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(fn_template, mpi_op)                                                                                            \
	template <typename T>                                                                                                                                      \
	struct builtin_op<fn_template<T>> {                                                                                                                        \
		inline static const MPI_Op value = mpi_op;                                                                                                             \
	};

CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::plus, MPI_SUM)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::multiplies, MPI_PROD)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::bit_and, MPI_BAND)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::bit_or, MPI_BOR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::bit_xor, MPI_BXOR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::logical_and, MPI_LAND)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::logical_or, MPI_LOR)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::minimum, MPI_MIN)
CELERITY_DETAIL_MPI_SUPPORT_BUILTIN_OP(sycl::maximum, MPI_MAX)

// TODO MPI has MPI_MINLOC, MPI_MAXLOC to implement argmin() / argmax() reductions - can we use that?


} // namespace celerity::detail::mpi_support
