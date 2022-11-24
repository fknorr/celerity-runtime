#pragma once

#include <regex>
#include <type_traits>
#include <typeinfo>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "buffer.h"
#include "closure_hydrator.h"
#include "device_queue.h"
#include "host_queue.h"
#include "item.h"
#include "range_mapper.h"
#include "ranges.h"
#include "reduction_manager.h"
#include "task.h"
#include "types.h"
#include "workaround.h"

#if !defined(_MSC_VER)
// Required for kernel name demangling in Clang
#include <cxxabi.h>
#endif

namespace celerity {

class handler;

namespace experimental {
	template <int Dims>
	void constrain_split(handler& cgh, const range<Dims>& constraint);

	template <typename Hint>
	void hint(handler& cgh, Hint&& h);
} // namespace experimental

namespace detail {
	class device_queue;
	class task_manager;

	handler make_command_group_handler(const task_id tid, const size_t num_collective_nodes);
	std::unique_ptr<task> into_task(handler&& cgh);
	closure_object_id add_requirement(handler& cgh, const buffer_id bid, std::unique_ptr<range_mapper_base> rm);
	void add_requirement(handler& cgh, const host_object_id hoid, const experimental::side_effect_order order);
	void add_reduction(handler& cgh, const reduction_info& rinfo);

	template <typename Name>
	std::string kernel_debug_name() {
		std::string name = typeid(Name*).name();
#if !defined(_MSC_VER)
		const std::unique_ptr<char, void (*)(void*)> demangled(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr), std::free);
		const std::string demangled_s(demangled.get());
		if(size_t lastc; (lastc = demangled_s.rfind(":")) != std::string::npos) {
			name = demangled_s.substr(lastc + 1, demangled_s.length() - lastc - 1);
		} else {
			name = demangled_s;
		}
#elif defined(_MSC_VER)
		if(size_t lastc, id_end; (lastc = name.rfind(":")) != std::string::npos && (id_end = name.find(" ", lastc)) != std::string::npos) {
			name = name.substr(lastc + 1, id_end - lastc);
		}
#endif
		return name.substr(0, name.length() - 1);
	}

	struct unnamed_kernel {};

	template <typename KernelName>
	constexpr bool is_unnamed_kernel = std::is_same_v<KernelName, unnamed_kernel>;

#if CELERITY_DETAIL_IS_OLD_COMPUTECPP_COMPILER
	template <typename KernelName>
	struct kernel_name_wrapper;
#endif

	template <typename KernelName>
	struct bound_kernel_name {
		static_assert(!is_unnamed_kernel<KernelName>);
#if CELERITY_DETAIL_IS_OLD_COMPUTECPP_COMPILER
		using type = kernel_name_wrapper<KernelName>; // Suppress -Rsycl-kernel-naming diagnostic for local types
#else
		using type = KernelName;
#endif
	};

	template <typename KernelName>
	using bind_kernel_name = typename bound_kernel_name<KernelName>::type;

	struct simple_kernel_flavor {};
	struct nd_range_kernel_flavor {};

	template <typename Flavor, int Dims>
	struct kernel_flavor_traits;

	struct no_local_size {};

	template <int Dims>
	struct kernel_flavor_traits<simple_kernel_flavor, Dims> {
		inline static constexpr bool has_local_size = false;
		using local_size_type = no_local_size;
	};

	template <int Dims>
	struct kernel_flavor_traits<nd_range_kernel_flavor, Dims> {
		inline static constexpr bool has_local_size = true;
		using local_size_type = range<Dims>;
	};
} // namespace detail

/**
 * Tag type marking a `handler::host_task` as a master-node task. Do not construct this type directly, but use `celerity::on_master_node`.
 */
class on_master_node_tag {};

/**
 * Pass to `handler::host_task` to select the master-node task overload.
 */
inline constexpr on_master_node_tag on_master_node;

namespace experimental {
	class collective_tag_factory;

	/**
	 * Each collective host task is executed within a collective group. If multiple host tasks are scheduled within the same collective group, they are
	 * guaranteed to execute in the same order on every node and within a single thread per node. Each group has its own MPI communicator spanning all
	 * participating nodes, so MPI operations the user invokes from different collective groups do not race.
	 */
	class collective_group {
	  public:
		/// Creates a new collective group with a globally unique id. This must only be called from the main thread.
		collective_group() noexcept : m_cgid(next_cgid++) {}

	  private:
		friend class collective_tag_factory;
		detail::collective_group_id m_cgid;
		inline static size_t next_cgid = 1;
	};

	/**
	 * Tag type marking a `handler::host_task` as a collective task. Do not construct this type directly, but use `celerity::experimental::collective`
	 * or `celerity::experimental::collective(group)`.
	 */
	class collective_tag {
	  private:
		friend class collective_tag_factory;
		friend class celerity::handler;
		collective_tag(detail::collective_group_id cgid) : m_cgid(cgid) {}
		detail::collective_group_id m_cgid;
	};

	/**
	 * The collective group used in collective host tasks when no group is specified explicitly.
	 */
	inline const collective_group default_collective_group;

	/**
	 * Tag type construction helper. Do not construct this type directly, use `celerity::experimental::collective` instead.
	 */
	class collective_tag_factory {
	  public:
		operator experimental::collective_tag() const { return default_collective_group.m_cgid; }
		experimental::collective_tag operator()(experimental::collective_group cg) const { return cg.m_cgid; }
	};

	/**
	 * Pass to `handler::host_task` to select the collective host task overload.
	 *
	 * Either as a value to schedule with the `default_collective_group`:
	 * ```c++
	 * cgh.host_task(celerity::experimental::collective, []...);
	 * ```
	 *
	 * Or by specifying a collective group explicitly:
	 * ```c++
	 * celerity::experimental::collective_group my_group;
	 * ...
	 * cgh.host_task(celerity::experimental::collective(my_group), []...);
	 * ```
	 */
	inline constexpr collective_tag_factory collective;
} // namespace experimental

namespace detail {
	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel_with_celerity_item(const Kernel& kernel, const id<Dims>& s_id, const range<Dims>& global_range, const id<Dims>& global_offset,
	    const id<Dims>& chunk_offset, Reducers&... reducers) {
		kernel(make_item<Dims>(s_id + chunk_offset, global_offset, global_range), reducers...);
	}

	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel_with_celerity_nd_item(const Kernel& kernel, const cl::sycl::nd_item<Dims>& s_item, const range<Dims>& global_range,
	    const id<Dims>& global_offset, const id<Dims>& chunk_offset, const range<Dims>& group_range, const id<Dims>& group_offset, Reducers&... reducers) {
		kernel(make_nd_item<Dims>(s_item, global_range, global_offset, chunk_offset, group_range, group_offset), reducers...);
	}

	template <typename Kernel, int Dims, typename... Reducers>
	[[deprecated("Support for kernels receiving cl::sycl::item<Dims> will be removed in the future, change parameter type to celerity::item<Dims>")]] //
	inline void
	invoke_kernel_with_sycl_item(const Kernel& kernel, const cl::sycl::item<Dims>& s_item, Reducers&... reducers) {
		kernel(s_item, reducers...);
	}

	template <typename Kernel, int Dims>
	auto bind_simple_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims>& chunk_offset) {
		// NOCOMMIT UPDATE DESCRIPTION
		// The current mechanism for hydrating the SYCL placeholder accessors inside Celerity accessors requires that the kernel functor
		// capturing those accessors is copied at least once during submission (see also live_pass_device_handler::submit_to_sycl).
		// As of SYCL 2020 kernel functors are passed as const references, so we explicitly capture by value here.
		return [=](auto s_item_or_id, auto&... reducers) {
			if constexpr(std::is_invocable_v<Kernel, celerity::item<Dims>, decltype(reducers)...>) {
				if constexpr(CELERITY_WORKAROUND(DPCPP) && std::is_same_v<id<Dims>, decltype(s_item_or_id)>) {
					// CELERITY_WORKAROUND_LESS_OR_EQUAL: DPC++ passes a sycl::id instead of a sycl::item to kernels alongside reductions
					invoke_kernel_with_celerity_item(kernel, s_item_or_id, global_range, global_offset, chunk_offset, reducers...);
				} else {
					// Explicit item constructor: ComputeCpp does not pass a sycl::item, but an implicitly convertible sycl::item_base (?) which does not have
					// `sycl::id<> get_id()`
					invoke_kernel_with_celerity_item(
					    kernel, cl::sycl::item<Dims>{s_item_or_id}.get_id(), global_range, global_offset, chunk_offset, reducers...);
				}
			} else if constexpr(std::is_invocable_v<Kernel, cl::sycl::item<Dims>, decltype(reducers)...>) {
				invoke_kernel_with_sycl_item(kernel, cl::sycl::item<Dims>{s_item_or_id}, reducers...);
			} else {
				static_assert(constexpr_false<decltype(reducers)...>, "Kernel function must be invocable with celerity::item<Dims> (or cl::sycl::item<Dims>, "
				                                                      "deprecated) and as many reducer objects as reductions passed to parallel_for");
			}
		};
	}

	template <typename Kernel, int Dims>
	auto bind_nd_range_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims> chunk_offset,
	    const range<Dims>& group_range, const id<Dims>& group_offset) {
		return [=](cl::sycl::nd_item<Dims> s_item, auto&... reducers) {
			static_assert(std::is_invocable_v<Kernel, celerity::nd_item<Dims>, decltype(reducers)...>,
			    "Kernel function must be invocable with celerity::nd_item<Dims> or and as many reducer objects as reductions passed to parallel_for");
			invoke_kernel_with_celerity_nd_item(kernel, s_item, global_range, global_offset, chunk_offset, group_range, group_offset, reducers...);
		};
	}

	template <typename KernelName, typename... Params>
	inline void invoke_sycl_parallel_for(cl::sycl::handler& cgh, Params&&... args) {
		static_assert(CELERITY_FEATURE_UNNAMED_KERNELS || !is_unnamed_kernel<KernelName>,
		    "Your SYCL implementation does not support unnamed kernels, add a kernel name template parameter to this parallel_for invocation");
		if constexpr(detail::is_unnamed_kernel<KernelName>) {
#if CELERITY_FEATURE_UNNAMED_KERNELS // see static_assert above
			cgh.parallel_for(std::forward<Params>(args)...);
#endif
		} else {
			cgh.parallel_for<detail::bind_kernel_name<KernelName>>(std::forward<Params>(args)...);
		}
	}

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	class reduction_descriptor;

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	auto make_sycl_reduction(const reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>& d) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		cl::sycl::property_list props;
		if(!d.m_include_current_buffer_value) { props = {cl::sycl::property::reduction::initialize_to_identity{}}; }
		if constexpr(WithExplicitIdentity) {
			return sycl::reduction(d.m_device_ptr, d.m_identity, d.m_op, props);
		} else {
			return sycl::reduction(d.m_device_ptr, d.m_op, props);
		}
#endif
	}

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, false /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT /* identity */, bool include_current_buffer_value, DataT* device_ptr)
		    : m_bid(bid), m_op(combiner), m_include_current_buffer_value(include_current_buffer_value), m_device_ptr(device_ptr) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, false>(const reduction_descriptor&);

		buffer_id m_bid;
		BinaryOperation m_op;
		bool m_include_current_buffer_value;
		DataT* m_device_ptr;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, true /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT identity, bool include_current_buffer_value, DataT* device_ptr)
		    : m_bid(bid), m_op(combiner), m_identity(identity), m_include_current_buffer_value(include_current_buffer_value), m_device_ptr(device_ptr) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, true>(const reduction_descriptor&);

		buffer_id m_bid;
		BinaryOperation m_op;
		DataT m_identity{};
		bool m_include_current_buffer_value;
		DataT* m_device_ptr;
	};

	template <bool WithExplicitIdentity, typename DataT, int Dims, typename BinaryOperation>
	auto make_reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation op, DataT identity, const cl::sycl::property_list& prop_list) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		if(vars.get_range().size() != 1) {
			// Like SYCL 2020, Celerity only supports reductions to unit-sized buffers. This allows us to avoid tracking different parts of the buffer
			// as distributed_state and pending_reduction_state.
			throw std::runtime_error("Only unit-sized buffers can be reduction targets");
		}

		// NOCOMMIT Also need hydration mechanism for this

		// auto bid = detail::get_buffer_id(vars);
		// auto include_current_buffer_value = !prop_list.has_property<celerity::property::reduction::initialize_to_identity>();
		// DataT* device_ptr = nullptr;

		// if(detail::is_prepass_handler(cgh)) {
		// 	auto rid = detail::runtime::get_instance().get_reduction_manager().create_reduction<DataT, Dims>(bid, op, identity);
		// 	static_cast<detail::prepass_handler&>(cgh).add_reduction(reduction_info{rid, bid, include_current_buffer_value});
		// } else {
		// 	auto& device_handler = static_cast<detail::live_pass_device_handler&>(cgh);
		// 	include_current_buffer_value &= device_handler.is_reduction_initializer();

		// 	auto mode = cl::sycl::access_mode::discard_write;
		// 	if(include_current_buffer_value) { mode = cl::sycl::access_mode::read_write; }
		// 	device_ptr = static_cast<DataT*>(runtime::get_instance()
		// 	                                     .get_buffer_manager()
		// 	                                     .access_device_buffer<DataT, Dims>(device_handler.get_memory_id(), bid, mode, range<3>{1, 1, 1}, id<3>{})
		// 	                                     .ptr);
		// }
		// return detail::reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>{bid, op, identity, include_current_buffer_value, device_ptr};
#endif
	}

} // namespace detail

class handler {
  public:
	virtual ~handler() = default;

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(
		    global_range, id<Dims>(), detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, id<Dims> global_offset, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(
		    global_range, global_offset, detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(celerity::nd_range<Dims> execution_range, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::nd_range_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(execution_range.get_global_range(),
		    execution_range.get_offset(), execution_range.get_local_range(), std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{},
		    reductions_and_kernel...);
	}

	/**
	 * Schedules `kernel` to execute on the master node only. Call via `cgh.host_task(celerity::on_master_node, []...)`. The kernel is assumed to be invocable
	 * with the signature `void(const celerity::partition<0> &)` or `void()`.
	 *
	 * The kernel is executed in a background thread pool and multiple master node tasks may be executed concurrently if they are independent in the
	 * task graph, so proper synchronization must be ensured.
	 *
	 * **Compatibility note:** This replaces master-access tasks from Celerity 0.1 which were executed on the master node's main thread, so this implementation
	 * may require different lifetimes for captures. See `celerity::allow_by_ref` for more information on this topic.
	 */
	template <typename Functor>
	void host_task(on_master_node_tag, Functor kernel);

	/**
	 * Schedules `kernel` to be executed collectively on all nodes participating in the specified collective group. Call via
	 * `cgh.host_task(celerity::experimental::collective, []...)` or  `cgh.host_task(celerity::experimental::collective(group), []...)`.
	 * The kernel is assumed to be invocable with the signature `void(const celerity::experimental::collective_partition&)`
	 * or `void(const celerity::partition<1>&)`.
	 *
	 * This provides framework to use arbitrary collective MPI operations in a host task, such as performing collective I/O with parallel HDF5.
	 * The local node id,t the number of participating nodes as well as the group MPI communicator can be obtained from the `collective_partition` passed into
	 * the kernel.
	 *
	 * All collective tasks within a collective group are guaranteed to be executed in the same order on all nodes, additionally, all internal MPI operations
	 * and all host kernel invocations are executed in a single thread on each host.
	 */
	template <typename Functor>
	void host_task(experimental::collective_tag tag, Functor kernel);

	/**
	 * Schedules a distributed execution of `kernel` by splitting the iteration space in a runtime-defined manner. The kernel is assumed to be invocable
	 * with the signature `void(const celerity::partition<Dims>&)`.
	 *
	 * The kernel is executed in a background thread pool with multiple host tasks being run concurrently if they are independent in the task graph,
	 * so proper synchronization must be ensured. The partition passed into the kernel describes the split each host receives. It may be used with accessors
	 * to obtain the per-node portion of a buffer en-bloc, see `celerity::accessor::get_allocation_window` for details.
	 *
	 * There are no guarantees with respect to the split size and the order in which host tasks are re-orered between nodes other than
	 * the restrictions imposed by dependencies in the task graph. Also, the kernel may be invoked multiple times on one node and not be scheduled on
	 * another node. If you need guarantees about execution order
	 */
	template <int Dims, typename Functor>
	void host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel);

	/**
	 * Like `host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel)`, but with a `global_offset` of zero.
	 */
	template <int Dims, typename Functor>
	void host_task(range<Dims> global_range, Functor task) {
		host_task(global_range, {}, task);
	}

  private:
	// friend detail::execution_target detail::get_handler_execution_target(const handler& cgh); // NOCOMMIT For what do we use this?
	friend handler detail::make_command_group_handler(const detail::task_id tid, const size_t num_collective_nodes);
	friend std::unique_ptr<detail::task> detail::into_task(handler&& cgh);
	friend detail::closure_object_id detail::add_requirement(handler& cgh, const detail::buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);
	friend void detail::add_requirement(handler& cgh, const detail::host_object_id hoid, const experimental::side_effect_order order);
	friend void detail::add_reduction(handler& cgh, const detail::reduction_info& rinfo);
	template <int Dims>
	friend void constrain_split(handler& cgh, const range<Dims>& constraint);
	template <typename Hint>
	friend void experimental::hint(handler& cgh, Hint&& h);

	detail::task_id m_tid;
	detail::buffer_access_map m_access_map;
	detail::side_effect_map m_side_effects;
	detail::reduction_set m_reductions;
	std::unique_ptr<detail::command_launcher_storage_base> m_launcher; // NOCOMMIT Get rid of this again (no need to have this be a member)
	std::unique_ptr<detail::task> m_task = nullptr;
	size_t m_num_collective_nodes;
	detail::closure_object_id m_next_closure_object_id = 0;
	range<3> m_split_constraint = detail::unit_range;
	std::vector<std::unique_ptr<detail::hint_base>> m_hints;

	handler(detail::task_id tid, size_t num_collective_nodes) : m_tid(tid), m_num_collective_nodes(num_collective_nodes) {}

	template <typename KernelFlavor, typename KernelName, int Dims, typename... ReductionsAndKernel, size_t... ReductionIndices>
	void parallel_for_reductions_and_kernel(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_size, std::index_sequence<ReductionIndices...> indices,
	    ReductionsAndKernel&... kernel_and_reductions) {
		auto args_tuple = std::forward_as_tuple(kernel_and_reductions...);
		auto& kernel = std::get<sizeof...(kernel_and_reductions) - 1>(args_tuple);
		parallel_for_kernel_and_reductions<KernelFlavor, KernelName>(
		    global_range, global_offset, local_size, kernel, std::get<ReductionIndices>(args_tuple)...);
	}

	template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
	void parallel_for_kernel_and_reductions(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel& kernel, Reductions&... reductions);

	[[nodiscard]] detail::closure_object_id add_requirement(const detail::buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
		assert(m_task == nullptr);
		m_access_map.add_access(bid, std::move(rm));
		return m_next_closure_object_id++;
	}

	void add_requirement(const detail::host_object_id hoid, const experimental::side_effect_order order) {
		assert(m_task == nullptr);
		m_side_effects.add_side_effect(hoid, order);
	}

	void add_reduction(const detail::reduction_info& rinfo) { m_reductions.push_back(rinfo); }

	template <int Dims>
	void experimental_constrain_split(const range<Dims>& constraint) {
		assert(m_task == nullptr);
		m_split_constraint = detail::range_cast<3>(constraint);
	}

	template <typename Hint>
	void experimental_hint(Hint&& h) {
		static_assert(std::is_base_of_v<detail::hint_base, Hint>, "Incompatible hint");
		m_hints.emplace_back(std::make_unique<Hint>(std::forward<Hint>(h)));
	}

	range<3> get_constrained_granularity(const range<3>& granularity) const {
		range<3> result = granularity;
		for(size_t i = 0; i < 3; ++i) {
			const auto lcm = std::lcm(granularity[i], m_split_constraint[i]);
			if(lcm == 0) { throw std::runtime_error("Split constraint cannot be 0"); }
			result[i] = lcm;
		}
		return result;
	}

	void create_host_compute_task(detail::task_geometry geometry) {
		assert(m_task == nullptr);
		assert(m_launcher != nullptr);
		if(geometry.global_size.size() == 0) {
			// TODO this can be easily supported by not creating a task in case the execution range is empty
			throw std::runtime_error{"The execution range of distributed host tasks must have at least one item"};
		}
		m_task = detail::task::make_host_compute(
		    m_tid, geometry, std::move(m_launcher), std::move(m_access_map), std::move(m_side_effects), std::move(m_reductions));
	}

	void create_device_compute_task(detail::task_geometry geometry, std::string debug_name) {
		assert(m_task == nullptr);
		assert(m_launcher != nullptr);
		if(geometry.global_size.size() == 0) {
			// TODO unless reductions are involved, this can be easily supported by not creating a task in case the execution range is empty.
			// Edge case: If the task includes reductions that specify property::reduction::initialize_to_identity, we need to create a task that sets
			// the buffer state to an empty pending_reduction_state in the graph_generator. This will cause a trivial reduction_command to be generated on
			// each node that reads from the reduction output buffer, initializing it to the identity value locally.
			throw std::runtime_error{"The execution range of device tasks must have at least one item"};
		}
		if(!m_side_effects.empty()) { throw std::runtime_error{"Side effects cannot be used in device kernels"}; }
		m_task =
		    detail::task::make_device_compute(m_tid, geometry, std::move(m_launcher), std::move(m_access_map), std::move(m_reductions), std::move(debug_name));
	}

	void create_collective_task(detail::collective_group_id cgid) {
		assert(m_task == nullptr);
		assert(m_launcher != nullptr);
		m_task = detail::task::make_collective(m_tid, cgid, m_num_collective_nodes, std::move(m_launcher), std::move(m_access_map), std::move(m_side_effects));
	}

	void create_master_node_task() {
		assert(m_task == nullptr);
		assert(m_launcher != nullptr);
		m_task = detail::task::make_master_node(m_tid, std::move(m_launcher), std::move(m_access_map), std::move(m_side_effects));
	}

	// NOCOMMIT Get rid of this, just do it in create_device_compute_task?
	template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
	void make_device_kernel_launcher(const range<Dims>& global_range, const id<Dims>& global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel kernel, Reductions... reductions) {
		assert(m_launcher == nullptr);

		auto fn = [=](detail::device_queue& q, const subrange<3> execution_sr) {
			return q.submit([&](sycl::handler& cgh) {
				auto hydrated_kernel = detail::closure_hydrator::get_instance().hydrate_local_accessors(kernel, cgh);

				if constexpr(std::is_same_v<KernelFlavor, detail::simple_kernel_flavor>) {
					detail::invoke_sycl_parallel_for<KernelName>(cgh, detail::range_cast<Dims>(execution_sr.range), detail::make_sycl_reduction(reductions)...,
					    detail::bind_simple_kernel(hydrated_kernel, global_range, global_offset, detail::id_cast<Dims>(execution_sr.offset)));
				} else if constexpr(std::is_same_v<KernelFlavor, detail::nd_range_kernel_flavor>) {
					detail::invoke_sycl_parallel_for<KernelName>(cgh, cl::sycl::nd_range{detail::range_cast<Dims>(execution_sr.range), local_range},
					    detail::make_sycl_reduction(reductions)...,
					    detail::bind_nd_range_kernel(hydrated_kernel, global_range, global_offset, detail::id_cast<Dims>(execution_sr.offset),
					        global_range / local_range, detail::id_cast<Dims>(execution_sr.offset) / local_range));
				} else {
					static_assert(detail::constexpr_false<KernelFlavor>);
				}
			});
		};

		m_launcher = std::make_unique<detail::command_launcher_storage<decltype(fn)>>(std::move(fn));
	}

	template <int Dims, bool Collective, typename Kernel>
	void make_host_task_launcher(Kernel kernel) {
		static_assert(Dims >= 0);
		assert(m_launcher == nullptr);

		auto fn = [kernel](detail::host_queue& q, const detail::collective_group_id cgid, const range<3>& global_size, const subrange<3>& sr) {
			return q.submit(cgid, [kernel, global_size, sr](MPI_Comm comm) {
				(void)global_size;
				if constexpr(Dims > 0) {
					if constexpr(Collective) {
						static_assert(Dims == 1);
						const auto part = detail::make_collective_partition(detail::range_cast<1>(global_size), detail::subrange_cast<1>(sr), comm);
						kernel(part);
					} else {
						const auto part = detail::make_partition<Dims>(detail::range_cast<Dims>(global_size), detail::subrange_cast<Dims>(sr));
						kernel(part);
					}
				} else if constexpr(std::is_invocable_v<Kernel, const partition<0>&>) {
					(void)sr;
					const auto part = detail::make_0d_partition();
					kernel(part);
				} else {
					(void)sr;
					kernel();
				}
			});
		};

		m_launcher = std::make_unique<detail::command_launcher_storage<decltype(fn)>>(std::move(fn));
	}

	std::unique_ptr<detail::task> into_task() && {
		assert(m_task != nullptr);
		for(auto& h : m_hints) {
			m_task->add_hint(std::move(h));
		}
		return std::move(m_task);
	}
};

namespace detail {

	// inline execution_target get_handler_execution_target(const handler& cgh) { return cgh.get_task().get_execution_target(); }

	inline handler make_command_group_handler(const detail::task_id tid, const size_t num_collective_nodes) { return handler(tid, num_collective_nodes); }

	inline std::unique_ptr<detail::task> into_task(handler&& cgh) { return std::move(cgh).into_task(); }

	[[nodiscard]] inline closure_object_id add_requirement(handler& cgh, const buffer_id bid, std::unique_ptr<range_mapper_base> rm) {
		return cgh.add_requirement(bid, std::move(rm));
	}

	inline void add_requirement(handler& cgh, const host_object_id hoid, const experimental::side_effect_order order) {
		return cgh.add_requirement(hoid, order);
	}

	inline void add_reduction(handler& cgh, const detail::reduction_info& rinfo) { cgh.add_reduction(rinfo); }

	// class prepass_handler final : public handler {
	//   public:
	// 	explicit prepass_handler(task_id tid, size_t num_collective_nodes) : m_tid(tid), m_num_collective_nodes(num_collective_nodes) {}

	//   protected:
	// 	const class task& get_task() const override {
	// 		assert(m_task != nullptr);
	// 		return *m_task;
	// 	}

	//   private:
	// };

	// class live_pass_handler : public handler {
	//   public:
	// 	const class task& get_task() const final { return *task; }

	// 	template <int BufferDims, typename RangeMapper>
	// 	subrange<BufferDims> apply_range_mapper(RangeMapper rm, const range<BufferDims>& buffer_range) const {
	// 		return invoke_range_mapper(task->get_dimensions(), rm, chunk{sr.offset, sr.range, task->get_global_size()}, buffer_range);
	// 	}

	// 	subrange<3> get_iteration_range() { return sr; }

	// 	bool is_reduction_initializer() const { return initialize_reductions; }

	//   protected:
	// 	live_pass_handler(const class task* task, subrange<3> sr, bool initialize_reductions)
	// 	    : task(std::move(task)), sr(sr), initialize_reductions(initialize_reductions) {}

	// 	const class task* task = nullptr;

	// 	// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
	// 	subrange<3> sr;

	// 	bool initialize_reductions;
	// };

	// class live_pass_host_handler final : public live_pass_handler {
	//   public:
	// 	live_pass_host_handler(const class task* task, subrange<3> sr, bool initialize_reductions, host_queue& queue)
	// 	    : live_pass_handler(task, sr, initialize_reductions), m_queue(&queue) {}

	// 	template <int Dims, typename Kernel>
	// 	void schedule(Kernel kernel) {
	// 		static_assert(Dims >= 0);
	// 		m_future = m_queue->submit(task->get_collective_group_id(), [kernel, global_size = task->get_global_size(), sr = sr](MPI_Comm) {
	// 			if constexpr(Dims > 0) {
	// 				const auto part = make_partition<Dims>(range_cast<Dims>(global_size), subrange_cast<Dims>(sr));
	// 				kernel(part);
	// 			} else if constexpr(std::is_invocable_v<Kernel, const partition<0>&>) {
	// 				(void)sr;
	// 				const auto part = make_0d_partition();
	// 				kernel(part);
	// 			} else {
	// 				(void)sr;
	// 				kernel();
	// 			}
	// 		});
	// 	}

	// 	template <typename Kernel>
	// 	void schedule_collective(Kernel kernel) {
	// 		m_future = m_queue->submit(task->get_collective_group_id(), [kernel, global_size = task->get_global_size(), sr = sr](MPI_Comm comm) {
	// 			const auto part = make_collective_partition(range_cast<1>(global_size), subrange_cast<1>(sr), comm);
	// 			kernel(part);
	// 		});
	// 	}

	// 	std::future<host_queue::execution_info> into_future() { return std::move(m_future); }

	//   private:
	// 	host_queue* m_queue;
	// 	std::future<host_queue::execution_info> m_future;
	// };

	// class live_pass_device_handler final : public live_pass_handler {
	//   public:
	// 	live_pass_device_handler(const class task* task, subrange<3> sr, bool initialize_reductions, device_queue& d_queue)
	// 	    : live_pass_handler(task, sr, initialize_reductions), m_d_queue(&d_queue) {}

	// 	template <typename CGF>
	// 	void submit_to_sycl(CGF&& cgf) {
	// 		m_event = m_d_queue->submit([&](cl::sycl::handler& cgh) {
	// 			this->m_eventual_cgh = &cgh;
	// 			std::forward<CGF>(cgf)(cgh);
	// 			this->m_eventual_cgh = nullptr;
	// 		});
	// 	}

	// 	cl::sycl::event get_submission_event() const { return m_event; }

	// 	cl::sycl::handler* const* get_eventual_sycl_cgh() const { return &m_eventual_cgh; }

	// 	// For now we assume that all buffers reside in the same memory (the main memory of the device),
	// 	// which is why this function isn't parameterized with a buffer id.
	// 	memory_id get_memory_id() const { return m_d_queue->get_memory_id(); }

	//   private:
	// 	device_queue* m_d_queue;
	// 	cl::sycl::handler* m_eventual_cgh = nullptr;
	// 	cl::sycl::event m_event;
	// };


} // namespace detail

// NOCOMMIT Can we move up?
template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
void handler::parallel_for_kernel_and_reductions(range<Dims> global_range, id<Dims> global_offset,
    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel& kernel, Reductions&... reductions) {
	if constexpr(!CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS && sizeof...(reductions) > 0) {
		static_assert(detail::constexpr_false<Kernel>, "Reductions are not supported by your SYCL implementation");
	} else if constexpr(!CELERITY_FEATURE_SCALAR_REDUCTIONS && sizeof...(reductions) > 1) {
		static_assert(detail::constexpr_false<Kernel>, "DPC++ currently does not support more than one reduction variable per kernel");
	}

	range<3> granularity = {1, 1, 1};
	if constexpr(detail::kernel_flavor_traits<KernelFlavor, Dims>::has_local_size) {
		for(int d = 0; d < Dims; ++d) {
			granularity[d] = local_range[d];
		}
	}
	const detail::task_geometry geometry{
	    Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), get_constrained_granularity(granularity)};
	make_device_kernel_launcher<KernelFlavor, KernelName, Dims>(global_range, global_offset, local_range, kernel, reductions...);
	create_device_compute_task(geometry, detail::kernel_debug_name<KernelName>());
}

// NOCOMMIT Can we move these up?
template <typename Functor>
void handler::host_task(on_master_node_tag, Functor kernel) {
	make_host_task_launcher<0, false>(kernel); // NOCOMMIT Just do this in create_master_node_task?
	create_master_node_task();
}

template <typename Functor>
void handler::host_task(experimental::collective_tag tag, Functor kernel) {
	make_host_task_launcher<1, true>(kernel);
	create_collective_task(tag.m_cgid);
}

template <int Dims, typename Functor>
void handler::host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel) {
	const detail::task_geometry geometry{Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), get_constrained_granularity({1, 1, 1})};
	make_host_task_launcher<Dims, false>(kernel);
	create_host_compute_task(geometry);
}

// NOCOMMIT (later): Can we move reduction stuff into separate header?

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const cl::sycl::property_list& prop_list = {}) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
#if CELERITY_WORKAROUND(DPCPP)
	static_assert(Dims == 1, "DPC++ currently does not support reductions to buffers with dimensionality != 1");
#endif
	static_assert(cl::sycl::has_known_identity_v<BinaryOperation, DataT>,
	    "Celerity does not currently support reductions without an identity. Either specialize "
	    "cl::sycl::known_identity or use the reduction() overload taking an identity at runtime");
	return detail::make_reduction<false>(vars, cgh, combiner, cl::sycl::known_identity_v<BinaryOperation, DataT>, prop_list);
#endif
}

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(const buffer<DataT, Dims>& vars, handler& cgh, const DataT identity, BinaryOperation combiner, const cl::sycl::property_list& prop_list = {}) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
	static_assert(!cl::sycl::has_known_identity_v<BinaryOperation, DataT>, "Identity is known to SYCL, remove the identity parameter from reduction()");
	return detail::make_reduction<true>(vars, cgh, combiner, identity, prop_list);
#endif
}

namespace experimental {
	template <int Dims>
	void constrain_split(handler& cgh, const range<Dims>& constraint) {
		cgh.experimental_constrain_split(constraint);
	}

	template <typename Hint>
	void hint(handler& cgh, Hint&& h) {
		cgh.experimental_hint(std::forward<Hint>(h));
	}
} // namespace experimental

} // namespace celerity
