#pragma once

#include <stdexcept>
#include <type_traits>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "ranges.h"

namespace celerity::experimental {

enum split_geometry { split, constant };
using split_geometry_map = std::vector<split_geometry>;

namespace access_geometry {
	struct other {
		friend bool operator==(other, other) { return true; }
		friend bool operator!=(other, other) { return false; }
	};

	struct one_to_one {
		int dimension;

		friend bool operator==(const one_to_one& lhs, const one_to_one& rhs) { return lhs.dimension == rhs.dimension; }
		friend bool operator!=(const one_to_one& lhs, const one_to_one& rhs) { return !(rhs == lhs); }
	};

	struct fixed {
		size_t offset;
		size_t range;

		friend bool operator==(const fixed& lhs, const fixed& rhs) { return lhs.offset == rhs.offset && lhs.range == rhs.range; }
		friend bool operator!=(const fixed& lhs, const fixed& rhs) { return !(rhs == lhs); }
	};
} // namespace access_geometry

using access_geometry_t = std::variant<access_geometry::other, access_geometry::one_to_one, access_geometry::fixed>;
using access_geometry_map = std::vector<access_geometry_t>;

struct range_mapper_properties {
	access_geometry_map access_geometry;
	bool is_constant = false;
	bool is_non_overlapping = false;
};

using buffer_geometry = std::vector<size_t>;

template <typename Functor>
struct range_mapper_traits {
	// by supplying a cv-qualified type here a user could accidentally not select the trait specializations below
	static_assert(!std::is_const_v<Functor> && !std::is_volatile_v<Functor>);

	static range_mapper_properties get_properties(const Functor&, const split_geometry_map&, const buffer_geometry&) { return {}; }
};
} // namespace celerity::experimental

namespace celerity {
namespace detail {

	template <typename T, typename Enable = void>
	struct is_equality_comparable : std::false_type {};

	template <typename T>
	struct is_equality_comparable<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>> : std::true_type {};

	template <typename T>
	constexpr bool is_equality_comparable_v = is_equality_comparable<T>::value;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_only = std::is_invocable_r_v<subrange<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_and_global_size =
	    std::is_invocable_r_v<subrange<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&, const range<BufferDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_kernel = is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims> //
	                                                      || is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>;

	template <typename Functor, int BufferDims>
	constexpr bool is_range_mapper_invocable = is_range_mapper_invocable_for_kernel<Functor, BufferDims, 1>    //
	                                           || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 2> //
	                                           || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 3>;

	[[noreturn]] inline void throw_invalid_range_mapper_args(int expect_kernel_dims, int expect_buffer_dims) {
		throw std::runtime_error(fmt::format("Invalid range mapper dimensionality: {0}-dimensional kernel submitted with a requirement whose range mapper "
		                                     "is neither invocable for chunk<{0}> nor (chunk<{0}>, range<{1}>) to produce subrange<{1}>",
		    expect_kernel_dims, expect_buffer_dims));
	}

	[[noreturn]] inline void throw_invalid_range_mapper_result(int expect_sr_dims, int actual_sr_dims, int kernel_dims) {
		throw std::runtime_error(fmt::format("Range mapper produces subrange of wrong dimensionality: Expecting subrange<{}>, got subrange<{}> for chunk<{}>",
		    expect_sr_dims, actual_sr_dims, kernel_dims));
	}

	template <int KernelDims, int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper_for_kernel(Functor&& fn, const celerity::chunk<KernelDims>& chunk, const range<BufferDims>& buffer_size) {
		static_assert(KernelDims >= 1 && KernelDims <= 3 && BufferDims >= 1 && BufferDims <= 3);
		if constexpr(is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk, buffer_size);
		} else if constexpr(is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk);
		} else {
			throw_invalid_range_mapper_args(KernelDims, BufferDims);
		}
	}

	template <int BufferDims>
	subrange<BufferDims> clamp_subrange_to_buffer_size(subrange<BufferDims> sr, range<BufferDims> buffer_size) {
		auto end = sr.offset + sr.range;
		if(BufferDims > 0 && end[0] > buffer_size[0]) { sr.range[0] = sr.offset[0] <= buffer_size[0] ? buffer_size[0] - sr.offset[0] : 0; }
		if(BufferDims > 1 && end[1] > buffer_size[1]) { sr.range[1] = sr.offset[1] <= buffer_size[1] ? buffer_size[1] - sr.offset[1] : 0; }
		if(BufferDims > 2 && end[2] > buffer_size[2]) { sr.range[2] = sr.offset[2] <= buffer_size[2] ? buffer_size[2] - sr.offset[2] : 0; }
		return sr;
	}

	template <int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper(int kernel_dims, Functor fn, const celerity::chunk<3>& chunk, const range<BufferDims>& buffer_size) {
		static_assert(is_range_mapper_invocable<Functor, BufferDims>);
		subrange<BufferDims> sr;
		switch(kernel_dims) {
		case 0:
			[[fallthrough]]; // range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
			                 // anyway, we require range mappers to take at least 1d subranges
		case 1: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<1>(chunk), buffer_size); break;
		case 2: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<2>(chunk), buffer_size); break;
		case 3: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<3>(chunk), buffer_size); break;
		default: assert(!"Unreachable"); return {};
		}
		return clamp_subrange_to_buffer_size(sr, buffer_size);
	}

	class range_mapper_base {
	  public:
		explicit range_mapper_base(cl::sycl::access::mode am) : m_access_mode(am) {}

		cl::sycl::access::mode get_access_mode() const { return m_access_mode; }

		virtual int get_buffer_dimensions() const = 0;

		virtual subrange<1> map_1(const chunk<1>& chnk) const = 0;
		virtual subrange<1> map_1(const chunk<2>& chnk) const = 0;
		virtual subrange<1> map_1(const chunk<3>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<1>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<2>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<3>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<1>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<2>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<3>& chnk) const = 0;

		virtual ~range_mapper_base() = default;

		virtual std::unique_ptr<range_mapper_base> clone_as(access_mode mode) const = 0;
		std::unique_ptr<range_mapper_base> clone() { return clone_as(m_access_mode); }

		virtual experimental::range_mapper_properties get_properties(const experimental::split_geometry_map& split) const = 0;

		virtual bool function_equals(const range_mapper_base& other) const = 0;

	  protected:
		range_mapper_base(const range_mapper_base& other) = default;
		range_mapper_base& operator=(const range_mapper_base& other) = default;

	  private:
		cl::sycl::access::mode m_access_mode;
	};

	template <int BufferDims, typename Functor>
	class range_mapper final : public range_mapper_base {
	  public:
		range_mapper(Functor rmfn, cl::sycl::access::mode am, range<BufferDims> buffer_size)
		    : range_mapper_base(am), m_rmfn(rmfn), m_buffer_size(buffer_size) {}

		int get_buffer_dimensions() const override { return BufferDims; }

		subrange<1> map_1(const chunk<1>& chnk) const override { return map<1>(chnk); }
		subrange<1> map_1(const chunk<2>& chnk) const override { return map<1>(chnk); }
		subrange<1> map_1(const chunk<3>& chnk) const override { return map<1>(chnk); }
		subrange<2> map_2(const chunk<1>& chnk) const override { return map<2>(chnk); }
		subrange<2> map_2(const chunk<2>& chnk) const override { return map<2>(chnk); }
		subrange<2> map_2(const chunk<3>& chnk) const override { return map<2>(chnk); }
		subrange<3> map_3(const chunk<1>& chnk) const override { return map<3>(chnk); }
		subrange<3> map_3(const chunk<2>& chnk) const override { return map<3>(chnk); }
		subrange<3> map_3(const chunk<3>& chnk) const override { return map<3>(chnk); }

		std::unique_ptr<range_mapper_base> clone_as(access_mode mode) const override { return std::make_unique<range_mapper>(m_rmfn, mode, m_buffer_size); }

		experimental::range_mapper_properties get_properties(const experimental::split_geometry_map& split) const override {
			std::vector<size_t> buffer_range(BufferDims);
			for(int d = 0; d < BufferDims; ++d) {
				buffer_range[d] = m_buffer_size[d];
			}
			return experimental::range_mapper_traits<Functor>::get_properties(m_rmfn, split, buffer_range);
		}

		bool function_equals(const range_mapper_base& other) const override {
			if constexpr(is_equality_comparable_v<Functor>) {
				if(typeid(other) == typeid(range_mapper)) { return m_rmfn == static_cast<const range_mapper&>(other).m_rmfn; }
			}
			return false;
		}

	  private:
		Functor m_rmfn;
		range<BufferDims> m_buffer_size;

		template <int OtherBufferDims, int KernelDims>
		subrange<OtherBufferDims> map(const chunk<KernelDims>& chnk) const {
			if constexpr(OtherBufferDims == BufferDims) {
				auto sr = invoke_range_mapper_for_kernel(m_rmfn, chnk, m_buffer_size);
				return clamp_subrange_to_buffer_size(sr, m_buffer_size);
			} else {
				throw_invalid_range_mapper_result(OtherBufferDims, BufferDims, KernelDims);
			}
		}
	};

	template <int KernelDims>
	subrange<3> apply_range_mapper(range_mapper_base const* rm, const chunk<KernelDims>& chnk) {
		switch(rm->get_buffer_dimensions()) {
		case 1: return subrange_cast<3>(rm->map_1(chnk));
		case 2: return subrange_cast<3>(rm->map_2(chnk));
		case 3: return rm->map_3(chnk);
		default: assert(false); abort();
		}
	}

	inline subrange<3> apply_range_mapper(range_mapper_base const* rm, const chunk<3>& chnk, int kernel_dims) {
		switch(kernel_dims) {
		case 0:
			[[fallthrough]]; // cl::sycl::range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
			                 // anyway, we require range mappers to take at least 1d subranges
		case 1: return apply_range_mapper<1>(rm, chunk_cast<1>(chnk));
		case 2: return apply_range_mapper<2>(rm, chunk_cast<2>(chnk));
		case 3: return apply_range_mapper<3>(rm, chunk_cast<3>(chnk));
		default: assert(!"Unreachable"); abort();
		}
		return {};
	}

	// std::is_permutation is C++20-only

	template <size_t, typename>
	struct prepend_to_index_sequence;

	template <size_t Index, size_t... Seq>
	struct prepend_to_index_sequence<Index, std::index_sequence<Seq...>> {
		using type = std::index_sequence<Index, Seq...>;
	};

	template <size_t Needle, typename Seq>
	struct find_in_index_sequence;

	template <size_t Needle, size_t Head, size_t... Tail>
	struct find_in_index_sequence<Needle, std::index_sequence<Head, Tail...>> {
		using next = find_in_index_sequence<Needle, std::index_sequence<Tail...>>;
		constexpr static bool found = next::found;
		using without = typename prepend_to_index_sequence<Head, typename next::without>::type;
		constexpr static size_t index = 1 + next::index;
	};

	template <size_t Needle, size_t... Tail>
	struct find_in_index_sequence<Needle, std::index_sequence<Needle, Tail...>> {
		constexpr static bool found = true;
		using without = std::index_sequence<Tail...>;
		constexpr static size_t index = 0;
	};

	template <size_t Needle>
	struct find_in_index_sequence<Needle, std::index_sequence<>> {
		constexpr static bool found = false;
		using without = std::index_sequence<>;
		constexpr static size_t index = 0;
	};

	template <size_t Min, size_t Max, typename Seq>
	struct is_range_permutation;

	template <size_t Min, size_t Max, size_t... Seq>
	struct is_range_permutation<Min, Max, std::index_sequence<Seq...>> {
		using find = find_in_index_sequence<Min, std::index_sequence<Seq...>>;
		constexpr static bool value = find::found && is_range_permutation<Min + 1, Max, typename find::without>::value;
	};

	template <size_t Min>
	struct is_range_permutation<Min, Min, std::index_sequence<Min>> : std::true_type {};

	template <size_t Min, size_t... Seq>
	struct is_range_permutation<Min, Min, std::index_sequence<Seq...>> : std::false_type {};

	template <size_t... Seq>
	struct is_index_permutation : is_range_permutation<0, sizeof...(Seq) - 1, std::index_sequence<Seq...>> {};

	template <>
	struct is_index_permutation<> : std::true_type {};

	template <size_t... Seq>
	constexpr bool is_index_permutation_v = is_index_permutation<Seq...>::value;

	static_assert(is_index_permutation_v<>);
	static_assert(is_index_permutation_v<0>);
	static_assert(!is_index_permutation_v<1>);
	static_assert(!is_index_permutation_v<2>);
	static_assert(is_index_permutation_v<0, 1>);
	static_assert(is_index_permutation_v<1, 0>);
	static_assert(!is_index_permutation_v<2, 0>);
	static_assert(!is_index_permutation_v<0, 0>);
	static_assert(is_index_permutation_v<2, 0, 1>);
	static_assert(!is_index_permutation_v<1, 1, 1>);

} // namespace detail


// --------------------------- Convenience range mappers ---------------------------

namespace access {

	template <int Dims = 0>
	struct one_to_one;

	template <>
	struct one_to_one<0> {
		template <int Dims>
		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			return chnk;
		}

		friend bool operator==(const one_to_one, const one_to_one) { return true; }
		friend bool operator!=(const one_to_one, const one_to_one) { return false; }
	};

	template <int Dims>
	struct [[deprecated("Explicitly-dimensioned range mappers are deprecated, remove template arguments from celerity::one_to_one")]] one_to_one
	    : one_to_one<0>{};

	one_to_one() -> one_to_one<>;

	template <int KernelDims, int BufferDims = KernelDims>
	struct fixed;

	template <int BufferDims>
	struct fixed<BufferDims, BufferDims> {
		fixed(const subrange<BufferDims>& sr) : m_sr(sr) {}

		const subrange<BufferDims>& get_subrange() const { return m_sr; }

		template <int KernelDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>&) const {
			return m_sr;
		}

		friend bool operator==(const fixed& lhs, const fixed& rhs) { return lhs.m_sr == rhs.m_sr; }
		friend bool operator!=(const fixed& lhs, const fixed& rhs) { return !(rhs == lhs); }

	  private:
		subrange<BufferDims> m_sr;
	};

	template <int KernelDims, int BufferDims>
	struct fixed : fixed<BufferDims, BufferDims> {
		[[deprecated("Explicitly-dimensioned range mappers are deprecated, remove first template argument from celerity::fixed")]] //
		fixed(const subrange<BufferDims>& sr)
		    : fixed<BufferDims, BufferDims>(sr) {}
	};

	template <int BufferDims>
	fixed(subrange<BufferDims>) -> fixed<BufferDims>;

	template <int Dims>
	struct slice {
		slice(size_t dim_idx) : m_dim_idx(dim_idx) { assert(dim_idx < Dims && "Invalid slice dimension index (starts at 0)"); }

		size_t get_slice_dimension() const { return m_dim_idx; }

		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			subrange<Dims> result = chnk;
			result.offset[m_dim_idx] = 0;
			// Since we don't know the range of the buffer, we just set it way too high and let it be clamped to the correct range
			result.range[m_dim_idx] = std::numeric_limits<size_t>::max();
			return result;
		}

		friend bool operator==(const slice& lhs, const slice& rhs) { return lhs.m_dim_idx == rhs.m_dim_idx; }
		friend bool operator!=(const slice& lhs, const slice& rhs) { return !(rhs == lhs); }

	  private:
		size_t m_dim_idx;
	};

	template <int KernelDims = 0, int BufferDims = KernelDims>
	struct all;

	template <>
	struct all<0, 0> {
		template <int KernelDims, int BufferDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>&, const range<BufferDims>& buffer_size) const {
			return {{}, buffer_size};
		}

		friend bool operator==(const all, const all) { return true; }
		friend bool operator!=(const all, const all) { return false; }
	};

	template <int KernelDims, int BufferDims>
	struct [[deprecated("Explicitly-dimensioned range mappers are deprecated, remove template arguments from celerity::all")]] all : all<0, 0>{};

	all() -> all<>;

	template <int Dims>
	struct neighborhood {
		neighborhood(size_t dim0) : m_dim0(dim0), m_dim1(0), m_dim2(0) {}

		template <int D = Dims, std::enable_if_t<D >= 2, void*>...>
		neighborhood(size_t dim0, size_t dim1) : m_dim0(dim0), m_dim1(dim1), m_dim2(0) {}

		template <int D = Dims, std::enable_if_t<D == 3, void*>...>
		neighborhood(size_t dim0, size_t dim1, size_t dim2) : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2) {}

		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			subrange<3> result = {celerity::detail::id_cast<3>(chnk.offset), celerity::detail::range_cast<3>(chnk.range)};
			const id<3> delta = {m_dim0 < result.offset[0] ? m_dim0 : result.offset[0], m_dim1 < result.offset[1] ? m_dim1 : result.offset[1],
			    m_dim2 < result.offset[2] ? m_dim2 : result.offset[2]};
			result.offset -= delta;
			result.range += range<3>{m_dim0 + delta[0], m_dim1 + delta[1], m_dim2 + delta[2]};
			return detail::subrange_cast<Dims>(result);
		}

		friend bool operator==(const neighborhood& lhs, const neighborhood& rhs) {
			return lhs.m_dim0 == rhs.m_dim0 && lhs.m_dim1 == rhs.m_dim1 && lhs.m_dim2 == rhs.m_dim2;
		}
		friend bool operator!=(const neighborhood& lhs, const neighborhood& rhs) { return !(rhs == lhs); }

	  private:
		size_t m_dim0, m_dim1, m_dim2;
	};

	neighborhood(size_t) -> neighborhood<1>;
	neighborhood(size_t, size_t) -> neighborhood<2>;
	neighborhood(size_t, size_t, size_t) -> neighborhood<3>;

} // namespace access

namespace experimental::access {

	/**
	 * For a 1D kernel, splits an nD-buffer evenly along its slowest dimension.
	 *
	 * This range mapper is unique in the sense that the chunk parameter (i.e. the iteration space) is unrelated to the buffer indices it maps to.
	 * It is designed to distribute a buffer in contiguous portions between nodes for collective host tasks, allowing each node to output its portion in
	 * I/O operations. See `accessor::get_allocation_window` on how to access the resulting host memory.
	 */
	template <int BufferDims>
	class even_split {
		static_assert(BufferDims > 0);

	  public:
		even_split() = default;
		explicit even_split(const range<BufferDims>& granularity) : m_granularity(granularity) {}

		subrange<BufferDims> operator()(const chunk<1>& chunk, const range<BufferDims>& buffer_size) const {
			if(chunk.global_size[0] == 0) { return {}; }

			// Equal splitting has edge cases when buffer_size is not a multiple of global_size * granularity. Splitting is performed in a manner that
			// distributes the remainder as equally as possible while adhering to granularity. In case buffer_size is not even a multiple of granularity,
			// only last chunk should be oddly sized so that only one node needs to deal with misaligned buffers.

			// 1. Each slice has at least buffer_size / global_size items, rounded down to the nearest multiple of the granularity.
			// 2. The first chunks in the range receive one additional granularity-sized block each to distribute most of the remainder
			// 3. The last chunk additionally receives the not-granularity-sized part of the remainder, if any.

			auto dim0_step = buffer_size[0] / (chunk.global_size[0] * m_granularity[0]) * m_granularity[0];
			auto dim0_remainder = buffer_size[0] - chunk.global_size[0] * dim0_step;
			auto dim0_range_in_this_chunk = chunk.range[0] * dim0_step;
			auto sum_dim0_remainder_in_prev_chunks = std::min(dim0_remainder / m_granularity[0] * m_granularity[0], chunk.offset[0] * m_granularity[0]);
			if(dim0_remainder > sum_dim0_remainder_in_prev_chunks) {
				dim0_range_in_this_chunk +=
				    std::min(chunk.range[0], (dim0_remainder - sum_dim0_remainder_in_prev_chunks) / m_granularity[0]) * m_granularity[0];
				if(chunk.offset[0] + chunk.range[0] == chunk.global_size[0]) { dim0_range_in_this_chunk += dim0_remainder % m_granularity[0]; }
			}
			auto dim0_offset_in_this_chunk = chunk.offset[0] * dim0_step + sum_dim0_remainder_in_prev_chunks;

			subrange<BufferDims> sr;
			sr.offset[0] = dim0_offset_in_this_chunk;
			sr.range = buffer_size;
			sr.range[0] = dim0_range_in_this_chunk;

			return sr;
		}

		friend bool operator==(const even_split& lhs, const even_split& rhs) { return lhs.m_granularity == rhs.m_granularity; }
		friend bool operator!=(const even_split& lhs, const even_split& rhs) { return !(rhs == lhs); }

	  private:
		range<BufferDims> m_granularity = detail::range_cast<BufferDims>(range<3>(1, 1, 1));
	};

	template <int... Permutation>
	struct transposed {
		static_assert(detail::is_index_permutation_v<Permutation...>);

		static constexpr int dimensions = sizeof...(Permutation);

		subrange<dimensions> operator()(const chunk<dimensions>& ck) const { return {{ck.offset[Permutation]...}, {ck.range[Permutation]...}}; }
	};
	// TODO what is the trait/query for this? "given split direction X we always have entire dimension Y"?

} // namespace experimental::access

} // namespace celerity

namespace celerity::experimental {

template <int Dims>
struct range_mapper_traits<celerity::access::one_to_one<Dims>> {
	static range_mapper_properties get_properties(
	    const celerity::access::one_to_one<Dims>&, const split_geometry_map& split, [[maybe_unused]] const buffer_geometry& buffer_range) {
		access_geometry_map map(split.size());
		for(int d = 0; d < static_cast<int>(split.size()); ++d) {
			map[d] = access_geometry::one_to_one{d};
		}
		return range_mapper_properties{std::move(map), /* is_constant= */ false, /* is_non_overlapping= */ true};
	}
};

template <int Dims>
struct range_mapper_traits<celerity::access::slice<Dims>> {
	static_assert(Dims > 0);

	static range_mapper_properties get_properties(
	    const celerity::access::slice<Dims>& rmfn, const split_geometry_map& split, const buffer_geometry& buffer_range) {
		assert(split.size() == Dims);
		assert(buffer_range.size() == Dims);

		access_geometry_map map(Dims);
		bool is_constant = true;
		for(int d = 0; d < Dims; ++d) {
			if(split[d] == split_geometry::split && d != static_cast<int>(rmfn.get_slice_dimension())) {
				map[d] = access_geometry::one_to_one{d};
				is_constant = false;
			} else {
				map[d] = access_geometry::fixed{0, buffer_range[d]};
			}
		}

		const bool is_non_overlapping = split[static_cast<int>(rmfn.get_slice_dimension())] != split_geometry::split;
		return range_mapper_properties{std::move(map), is_constant, is_non_overlapping};
	}
};

template <int KernelDims, int BufferDims>
struct range_mapper_traits<celerity::access::fixed<KernelDims, BufferDims>> {
	static range_mapper_properties get_properties(
	    const celerity::access::fixed<KernelDims, BufferDims>& rmfn, const split_geometry_map&, [[maybe_unused]] const buffer_geometry& buffer_range) {
		assert(buffer_range.size() == BufferDims);

		access_geometry_map map(BufferDims);
		for(int d = 0; d < BufferDims; ++d) {
			map[d] = access_geometry::fixed{rmfn.get_subrange().offset[d], rmfn.get_subrange().range[d]};
		}
		return range_mapper_properties{std::move(map), /* is_constant= */ true, /* is_non_overlapping= */ false};
	}
};

template <int KernelDims, int BufferDims>
struct range_mapper_traits<celerity::access::all<KernelDims, BufferDims>> {
	static range_mapper_properties get_properties(
	    const celerity::access::all<KernelDims, BufferDims>& rmfn, const split_geometry_map&, const buffer_geometry& buffer_range) {
		access_geometry_map map(buffer_range.size());
		for(size_t d = 0; d < buffer_range.size(); ++d) {
			map[d] = access_geometry::fixed{0, buffer_range[d]};
		}
		return range_mapper_properties{std::move(map), /* is_constant= */ true, /* is_non_overlapping= */ false};
	}
};

template <int... Permutation>
struct range_mapper_traits<celerity::experimental::access::transposed<Permutation...>> {
	static range_mapper_properties get_properties(const celerity::experimental::access::transposed<Permutation...>&,
	    [[maybe_unused]] const split_geometry_map& split, [[maybe_unused]] const buffer_geometry& buffer_range) {
		assert(split.size() == sizeof...(Permutation));
		assert(buffer_range.size() == sizeof...(Permutation));

		access_geometry_map map{access_geometry::one_to_one{Permutation}...};
		return range_mapper_properties{std::move(map), /* is_constant= */ false, /* is_non_overlapping= */ true};
	}
};

} // namespace celerity::experimental
