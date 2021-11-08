#pragma once

#include "ranges.h"

namespace celerity {

template <int Dims>
class item;
template <int Dims>
class group;
template <int Dims>
class nd_item;

namespace detail {

	template <int Dims>
	inline item<Dims> make_item(cl::sycl::id<Dims> absolute_global_id, cl::sycl::id<Dims> global_offset, cl::sycl::range<Dims> global_range) {
		return item<Dims>{absolute_global_id, global_offset, global_range};
	}

	template <int Dims>
	inline group<Dims> make_group(const cl::sycl::nd_item<Dims>& sycl_item, const cl::sycl::id<Dims>& group_id, const cl::sycl::range<Dims>& group_range) {
		return group<Dims>{sycl_item, group_id, group_range};
	}

	template <int Dims>
	nd_item<Dims> make_nd_item(const cl::sycl::nd_item<Dims>& sycl_item, const cl::sycl::range<Dims>& global_range, const cl::sycl::id<Dims>& global_offset,
	    const cl::sycl::id<Dims>& chunk_offset, const cl::sycl::range<Dims>& group_range, const cl::sycl::id<Dims>& group_offset) {
		return nd_item<Dims>{sycl_item, global_range, global_offset, chunk_offset, group_range, group_offset};
	}


	// SYCL 1.2.1: group::get_id(), SYCL 2020: group::get_group_id()
	// hipSYCL switched from the 1.2.1 interface between releases, so we currently don't have a reliable WORKAROUND macro for this.

	template <typename Group, typename Enable = void>
	struct sycl_group_has_get_group_id : public std::false_type {};

	template <typename Group>
	struct sycl_group_has_get_group_id<Group, std::void_t<decltype(std::declval<Group>().get_group_id())>> : public std::true_type {};

	template <int Dims>
	cl::sycl::id<Dims> get_sycl_group_id(const cl::sycl::group<Dims>& grp) {
		if constexpr(sycl_group_has_get_group_id<cl::sycl::group<Dims>>::value) {
			return grp.get_group_id();
		} else {
			return grp.get_id();
		}
	}


	template <int Dims>
	inline cl::sycl::nd_item<Dims>& get_sycl_item(nd_item<Dims>& nd_item) {
		return nd_item.sycl_item;
	}

	template <int Dims>
	inline const cl::sycl::nd_item<Dims>& get_sycl_item(const nd_item<Dims>& nd_item) {
		return nd_item.sycl_item;
	}

	template <int Dims>
	inline cl::sycl::nd_item<Dims>& get_sycl_item(group<Dims>& g) {
		return g.sycl_item;
	}

	template <int Dims>
	inline const cl::sycl::nd_item<Dims>& get_sycl_item(const group<Dims>& g) {
		return g.sycl_item;
	}

} // namespace detail

// We replace sycl::item with celerity::item to correctly expose the cluster global size instead of the chunk size to the user.
template <int Dims = 1>
class item {
  public:
	item() = delete;

	friend bool operator==(const item& lhs, const item& rhs) {
		return lhs.absolute_global_id == rhs.absolute_global_id && lhs.global_offset == rhs.global_offset && lhs.global_range == rhs.global_range;
	}

	friend bool operator!=(const item& lhs, const item& rhs) { return !(lhs == rhs); }

	cl::sycl::id<Dims> get_id() const { return absolute_global_id; }

	size_t get_id(int dimension) const { return absolute_global_id[dimension]; }

	operator cl::sycl::id<Dims>() const { return absolute_global_id; } // NOLINT(google-explicit-constructor)

	size_t operator[](int dimension) const { return absolute_global_id[dimension]; }

	cl::sycl::range<Dims> get_range() const { return global_range; }

	size_t get_range(int dimension) const { return global_range[dimension]; }

	size_t get_linear_id() const { return detail::get_linear_index(global_range, absolute_global_id - global_offset); }

	cl::sycl::id<Dims> get_offset() const { return global_offset; }

  private:
	template <int D>
	friend item<D> celerity::detail::make_item(cl::sycl::id<D>, cl::sycl::id<D>, cl::sycl::range<D>);

	cl::sycl::id<Dims> absolute_global_id;
	cl::sycl::id<Dims> global_offset;
	cl::sycl::range<Dims> global_range;

	explicit item(cl::sycl::id<Dims> absolute_global_id, cl::sycl::id<Dims> global_offset, cl::sycl::range<Dims> global_range)
	    : absolute_global_id(absolute_global_id), global_offset(global_offset), global_range(global_range) {}
};


template <int Dims = 1>
class group {
  public:
	using id_type = cl::sycl::id<Dims>;
	using range_type = cl::sycl::range<Dims>;
	using linear_id_type = size_t;
	static constexpr int dimensions = Dims;
	static constexpr memory_scope fence_scope = memory_scope_work_group;

	cl::sycl::id<Dims> get_group_id() const { return group_id; }

	size_t get_group_id(int dimension) const { return group_id[dimension]; }

	cl::sycl::id<Dims> get_local_id() const { return sycl_item.get_local_id(); }

	size_t get_local_id(int dimension) const { return sycl_item.get_local_id(dimension); }

	cl::sycl::range<Dims> get_local_range() const { return sycl_item.get_local_range(); }

	size_t get_local_range(int dimension) const { return sycl_item.get_local_range(dimension); }

	cl::sycl::range<Dims> get_group_range() const { return group_range; }

	size_t get_group_range(int dimension) const { return group_range[dimension]; }

	cl::sycl::range<Dims> get_max_local_range() const { return sycl_item.get_max_local_range(); }

	size_t operator[](int dimension) const { return group_id[dimension]; }

	size_t get_group_linear_id() const { return detail::get_linear_index(group_range, group_id); }

	size_t get_local_linear_id() const { return sycl_item.get_local_linear_id(); }

	size_t get_group_linear_range() const { return group_range.size(); }

	size_t get_local_linear_range() const { return sycl_item.get_local_range().size(); }

	bool leader() const { return sycl_item.get_local_id() == cl::sycl::id<Dims>{}; }

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_local_ptr<T> dest, decorated_global_ptr<T> src, size_t num_elements) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_global_ptr<T> dest, decorated_local_ptr<T> src, size_t num_elements) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_local_ptr<T> dest, decorated_global_ptr<T> src, size_t num_elements, size_t src_stride) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements, src_stride);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_global_ptr<T> dest, decorated_local_ptr<T> src, size_t num_elements, size_t dest_stride) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements, dest_stride);
	}

	template <typename... DeviceEvents>
	void wait_for(DeviceEvents... events) const {
		sycl_item.wait_for(events...);
	}

  private:
	// We capture SYCL `item` instead of `group` to provide celerity::group_barrier based on SYCL 1.2.1 nd_item.barrier()
	// TODO consider capturing `group` once ComputeCpp resolves this issue (if that benefits us e.g. wrt. struct size)
	cl::sycl::nd_item<Dims> sycl_item;
	cl::sycl::id<Dims> group_id;
	cl::sycl::range<Dims> group_range;

	template <int D>
	friend group<D> celerity::detail::make_group(const cl::sycl::nd_item<D>& sycl_item, const cl::sycl::id<D>& group_id, const cl::sycl::range<D>& group_range);

	template <int D>
	friend cl::sycl::nd_item<D>& celerity::detail::get_sycl_item(group<D>&);

	template <int D>
	friend const cl::sycl::nd_item<D>& celerity::detail::get_sycl_item(const group<D>&);

	explicit group(const cl::sycl::nd_item<Dims>& sycl_item, const cl::sycl::id<Dims>& group_id, const cl::sycl::range<Dims>& group_range)
	    : sycl_item(sycl_item), group_id(group_id), group_range(group_range) {}
};


// We replace sycl::nd_item with celerity::nd_item to correctly expose the cluster global size instead of the chunk size to the user.
template <int Dims = 1>
class nd_item {
  public:
	nd_item() = delete;

	cl::sycl::id<Dims> get_global_id() const { return global_id; }

	size_t get_global_id(int dimension) const { return global_id[dimension]; }

	size_t get_global_linear_id() const { return detail::get_linear_index(global_range, global_id); }

	cl::sycl::id<Dims> get_local_id() const { return sycl_item.get_local_id(); }

	size_t get_local_id(int dimension) const { return sycl_item.get_local_id(dimension); }

	size_t get_local_linear_id() const { return sycl_item.get_local_linear_id(); }

	group<Dims> get_group() const { return detail::make_group<Dims>(sycl_item, group_id, group_range); }

	size_t get_group(int dimension) const { return group_id[dimension]; }

	size_t get_group_linear_id() const { return detail::get_linear_index(group_range, group_id); }

	cl::sycl::range<Dims> get_group_range() const { return group_range; }

	size_t get_group_range(int dimension) const { return group_range[dimension]; }

#if !WORKAROUND_COMPUTECPP && !WORKAROUND_DPCPP // no sub_group support
	cl::sycl::sub_group get_sub_group() const { return sycl_item.get_sub_group(); }
#endif

	cl::sycl::range<Dims> get_global_range() const { return global_range; }

	size_t get_global_range(int dimension) const { return global_range[dimension]; }

	cl::sycl::range<Dims> get_local_range() const { return sycl_item.get_local_range(); }

	size_t get_local_range(int dimension) const { return sycl_item.get_local_range(dimension); }

	cl::sycl::id<Dims> get_offset() const { return global_offset; }

	celerity::nd_range<Dims> get_nd_range() const { return celerity::nd_range<Dims>{global_range, sycl_item.get_local_range(), global_offset}; }

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_local_ptr<T> dest, decorated_global_ptr<T> src, size_t num_elements) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_global_ptr<T> dest, decorated_local_ptr<T> src, size_t num_elements) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_local_ptr<T> dest, decorated_global_ptr<T> src, size_t num_elements, size_t src_stride) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements, src_stride);
	}

	template <typename T>
	cl::sycl::device_event async_work_group_copy(decorated_global_ptr<T> dest, decorated_local_ptr<T> src, size_t num_elements, size_t dest_stride) const {
		return sycl_item.async_work_group_copy(dest, src, num_elements, dest_stride);
	}

	template <typename... DeviceEvents>
	void wait_for(DeviceEvents... events) const {
		sycl_item.wait_for(events...);
	}

  private:
	cl::sycl::nd_item<Dims> sycl_item;
	cl::sycl::id<Dims> global_id;
	cl::sycl::id<Dims> global_offset;
	cl::sycl::range<Dims> global_range;
	cl::sycl::id<Dims> group_id;
	cl::sycl::range<Dims> group_range;

	template <int D>
	friend nd_item<D> celerity::detail::make_nd_item(const cl::sycl::nd_item<D>&, const cl::sycl::range<D>&, const cl::sycl::id<D>&, const cl::sycl::id<D>&,
	    const cl::sycl::range<D>&, const cl::sycl::id<D>&);

	template <int D>
	friend cl::sycl::nd_item<D>& celerity::detail::get_sycl_item(group<D>& nd_item);

	template <int D>
	friend const cl::sycl::nd_item<D>& celerity::detail::get_sycl_item(const group<D>& nd_item);

	explicit nd_item(const cl::sycl::nd_item<Dims>& sycl_item, const cl::sycl::range<Dims>& global_range, const cl::sycl::id<Dims>& global_offset,
	    const cl::sycl::id<Dims>& chunk_offset, const cl::sycl::range<Dims>& group_range, const cl::sycl::id<Dims>& group_offset)
	    : sycl_item(sycl_item), global_id(chunk_offset + sycl_item.get_global_id()), global_offset(global_offset), global_range(global_range),
	      group_id(group_offset + detail::get_sycl_group_id(sycl_item.get_group())), group_range(group_range) {}
};


#if !WORKAROUND_COMPUTECPP
using cl::sycl::group_barrier;
#endif

template <int Dims>
void group_barrier(const group<Dims>& g, memory_scope scope = memory_scope_work_group) {
#if WORKAROUND_COMPUTECPP
	auto space = scope > memory_scope_work_group ? cl::sycl::access::fence_space::global_and_local : cl::sycl::access::fence_space::local_space;
	detail::get_sycl_item(g).barrier(space);
#else
	return cl::sycl::group_barrier(detail::get_sycl_item(g).get_group(), static_cast<cl::sycl::memory_scope>(scope)); // identical representation
#endif
}


#if !WORKAROUND_COMPUTECPP // no group primitives

using cl::sycl::group_broadcast;

template <int Dims, typename T>
inline T group_broadcast(const group<Dims>& g, T x) {
	return cl::sycl::group_broadcast(detail::get_sycl_item(g).get_group(), x);
}

template <int Dims, typename T>
inline T group_broadcast(const group<Dims>& g, T x, size_t local_linear_id) {
	return cl::sycl::group_broadcast(detail::get_sycl_item(g).get_group(), x, local_linear_id);
}

template <int Dims, typename T>
inline T group_broadcast(const group<Dims>& g, T x, const cl::sycl::id<Dims>& local_id) {
	return cl::sycl::group_broadcast(detail::get_sycl_item(g).get_group(), x, local_id);
};


using cl::sycl::joint_any_of;

template <int Dims, typename Ptr, typename Predicate>
bool joint_any_of(const group<Dims>& g, Ptr first, Ptr last, Predicate pred) {
	return cl::sycl::joint_any_of(detail::get_sycl_item(g).get_group(), first, last, pred);
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::any_of_group;
#endif

template <int Dims, typename T, typename Predicate>
bool any_of_group(const group<Dims>& g, T x, Predicate pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_any_of(detail::get_sycl_item(g).get_group(), x, pred);
#else
	return cl::sycl::any_of_group(detail::get_sycl_item(g).get_group(), x, pred);
#endif
}

template <int Dims>
bool any_of_group(const group<Dims>& g, bool pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_any_of(detail::get_sycl_item(g).get_group(), pred);
#else
	return cl::sycl::any_of_group(detail::get_sycl_item(g).get_group(), pred);
#endif
}


using cl::sycl::joint_all_of;

template <int Dims, typename Ptr, typename Predicate>
bool joint_all_of(const group<Dims>& g, Ptr first, Ptr last, Predicate pred) {
	return cl::sycl::joint_all_of(detail::get_sycl_item(g).get_group(), first, last, pred);
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::all_of_group;
#endif

template <int Dims, typename T, typename Predicate>
bool all_of_group(const group<Dims>& g, T x, Predicate pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_all_of(detail::get_sycl_item(g).get_group(), x, pred);
#else
	return cl::sycl::all_of_group(detail::get_sycl_item(g).get_group(), x, pred);
#endif
}

template <int Dims>
bool all_of_group(const group<Dims>& g, bool pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_all_of(detail::get_sycl_item(g).get_group(), pred);
#else
	return cl::sycl::all_of_group(detail::get_sycl_item(g).get_group(), pred);
#endif
}


using cl::sycl::joint_none_of;

template <int Dims, typename Ptr, typename Predicate>
bool joint_none_of(const group<Dims>& g, Ptr first, Ptr last, Predicate pred) {
	return cl::sycl::joint_none_of(detail::get_sycl_item(g).get_group(), first, last, pred);
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::none_of_group;
#endif

template <int Dims, typename T, typename Predicate>
bool none_of_group(const group<Dims>& g, T x, Predicate pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_none_of(detail::get_sycl_item(g).get_group(), x, pred);
#else
	return cl::sycl::none_of_group(detail::get_sycl_item(g).get_group(), x, pred);
#endif
}

template <int Dims>
bool none_of_group(const group<Dims>& g, bool pred) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_none_of(detail::get_sycl_item(g).get_group(), pred);
#else
	return cl::sycl::none_of_group(detail::get_sycl_item(g).get_group(), pred);
#endif
}


#if !WORKAROUND_HIPSYCL // Not available in hipSYCL 0.9.1, but in the newest upstream version - TODO add feature detection?

using cl::sycl::permute_group_by_xor;
using cl::sycl::shift_group_left;
using cl::sycl::shift_group_right;

template <int Dims, typename T>
T shift_group_left(const group<Dims>& g, T x, size_t delta = 1) {
	return cl::sycl::shift_group_left(detail::get_sycl_item(g).get_group(), x, delta);
}

template <int Dims, typename T>
T shift_group_right(const group<Dims>& g, T x, size_t delta = 1) {
	return cl::sycl::shift_group_right(detail::get_sycl_item(g).get_group(), x, delta);
}

template <int Dims, typename T>
T permute_group_by_xor(const group<Dims>& g, T x, size_t mask) {
	return cl::sycl::permute_group_by_xor(detail::get_sycl_item(g).get_group(), x, mask);
}


using cl::sycl::select_from_group;

template <int Dims, typename T>
T select_from_group(const group<Dims>& g, T x, size_t remote_local_id) {
	return cl::sycl::select_from_group(detail::get_sycl_item(g).get_group(), x, remote_local_id);
}

#endif


using cl::sycl::joint_reduce;

template <int Dims, typename Ptr, typename BinaryOperation>
typename std::iterator_traits<Ptr>::value_type joint_reduce(const group<Dims>& g, Ptr first, Ptr last, BinaryOperation binary_op) {
	return cl::sycl::joint_reduce(detail::get_sycl_item(g).get_group(), first, last, binary_op);
}

template <int Dims, typename Ptr, typename T, typename BinaryOperation>
T joint_reduce(const group<Dims>& g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
	return cl::sycl::joint_reduce(detail::get_sycl_item(g).get_group(), first, last, init, binary_op);
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::reduce_over_group;
#endif

template <int Dims, typename T, typename BinaryOperation>
T reduce_over_group(const group<Dims>& g, T x, BinaryOperation binary_op) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_reduce(detail::get_sycl_item(g).get_group(), x, binary_op);
#else
	return cl::sycl::reduce_over_group(detail::get_sycl_item(g).get_group(), x, binary_op);
#endif
}

template <int Dims, typename V, typename T, typename BinaryOperation>
T reduce_over_group(const group<Dims>& g, V x, T init, BinaryOperation binary_op) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_reduce(detail::get_sycl_item(g).get_group(), x, init, binary_op);
#else
	return cl::sycl::reduce_over_group(detail::get_sycl_item(g).get_group(), x, init, binary_op);
#endif
}


using cl::sycl::joint_exclusive_scan;

template <int Dims, typename InPtr, typename OutPtr, typename BinaryOperation>
OutPtr joint_exclusive_scan(const group<Dims>& g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op) {
	return cl::sycl::joint_exclusive_scan(detail::get_sycl_item(g).get_group(), first, last, result, binary_op);
}

template <int Dims, typename InPtr, typename OutPtr, typename T, typename BinaryOperation>
T joint_exclusive_scan(const group<Dims>& g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
	return cl::sycl::joint_exclusive_scan(detail::get_sycl_item(g).get_group(), first, last, result, init, binary_op);
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::exclusive_scan_over_group;
#endif

template <int Dims, typename T, typename BinaryOperation>
T exclusive_scan_over_group(const group<Dims>& g, T x, BinaryOperation binary_op) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_exclusive_scan(detail::get_sycl_item(g).get_group(), x, binary_op);
#else
	return cl::sycl::exclusive_scan_over_group(detail::get_sycl_item(g).get_group(), x, binary_op);
#endif
}

template <int Dims, typename V, typename T, typename BinaryOperation>
T exclusive_scan_over_group(const group<Dims>& g, V x, T init, BinaryOperation binary_op) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_exclusive_scan(detail::get_sycl_item(g).get_group(), x, init, binary_op);
#else
	return cl::sycl::exclusive_scan_over_group(detail::get_sycl_item(g).get_group(), x, init, binary_op);
#endif
}


using cl::sycl::joint_inclusive_scan;

template <int Dims, typename InPtr, typename OutPtr, typename BinaryOperation>
OutPtr joint_inclusive_scan(const group<Dims>& g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op) {
	return cl::sycl::joint_inclusive_scan(detail::get_sycl_item(g).get_group(), first, last, result, binary_op);
}

template <int Dims, typename InPtr, typename OutPtr, typename T, typename BinaryOperation>
T joint_inclusive_scan(const group<Dims>& g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op, T init) {
	return cl::sycl::joint_inclusive_scan(detail::get_sycl_item(g).get_group(), first, last, result, binary_op, init);
}

template <int Dims, typename T, typename BinaryOperation>
T inclusive_scan_over_group(const group<Dims>& g, T x, BinaryOperation binary_op) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_inclusive_scan(detail::get_sycl_item(g).get_group(), x, binary_op);
#else
	return cl::sycl::inclusive_scan_over_group(detail::get_sycl_item(g).get_group(), x, binary_op);
#endif
}


#if !WORKAROUND_HIPSYCL
using cl::sycl::inclusive_scan_over_group;
#endif

template <int Dims, typename V, typename T, typename BinaryOperation>
T inclusive_scan_over_group(const group<Dims>& g, V x, BinaryOperation binary_op, T init) {
#if WORKAROUND_HIPSYCL
	return cl::sycl::group_inclusive_scan(detail::get_sycl_item(g).get_group(), x, binary_op, init);
#else
	return cl::sycl::inclusive_scan_over_group(detail::get_sycl_item(g).get_group(), x, binary_op, init);
#endif
}

#endif // !WORKAROUND_COMPUTECPP

} // namespace celerity