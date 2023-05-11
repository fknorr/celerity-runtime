#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "allscale/api/user/data/grid.h"
#include "device_queue.h"
#include "grid.h"
#include "host_queue.h" // NOCOMMIT Do we really need to include this..?
#include "intrusive_graph.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

class handler;

namespace detail {
	class hint_base {
	  public:
		virtual ~hint_base() = default;
	};
} // namespace detail

// TODO: Move elsewhere
namespace experimental::hints {
	class oversubscribe : public detail::hint_base {
	  public:
		oversubscribe(size_t factor) : m_factor(factor) {}

		[[nodiscard]] size_t get_factor() const { return m_factor; }

	  private:
		size_t m_factor;
	};

	class tiled_split : public detail::hint_base {};

}; // namespace experimental::hints

namespace detail {

	enum class task_type {
		epoch,
		host_compute,    ///< host task with explicit global size and celerity-defined split
		device_compute,  ///< device compute task
		host_collective, ///< host task with implicit 1d global size = #ranks and fixed split
		master_node,     ///< zero-dimensional host task
		horizon,         ///< task horizon
		forward,
	};

	enum class execution_target {
		none,
		host,
		device,
	};

	enum class epoch_action {
		none,
		barrier,
		shutdown,
	};

	using sycl_kernel_launcher = std::function<void(sycl::handler&, const subrange<3>&)>;
	using host_task_launcher = std::function<void(const subrange<3>&, const range<3>&, MPI_Comm)>;

	// TODO(IDAG) remove
	class command_launcher_storage_base {
	  public:
		virtual sycl::event operator()(device_queue& q, const subrange<3> execution_sr) const = 0;
		virtual std::future<host_queue::execution_info> operator()(
		    host_queue& q, const collective_group_id cgid, const range<3>& global_size, const subrange<3>& sr) const = 0;
		virtual ~command_launcher_storage_base() = default;
	};

	// TODO(IDAG) remove
	template <typename Functor>
	class command_launcher_storage : public command_launcher_storage_base {
	  public:
		command_launcher_storage(Functor fun) : m_fun(std::move(fun)) {}

		sycl::event operator()(device_queue& q, const subrange<3> execution_sr) const override { return invoke<sycl::event>(q, execution_sr); }

		std::future<host_queue::execution_info> operator()(
		    host_queue& q, const collective_group_id cgid, const range<3>& global_size, const subrange<3>& sr) const override {
			return invoke<std::future<host_queue::execution_info>>(q, cgid, global_size, sr);
		}

	  private:
		Functor m_fun;

		template <typename Ret, typename... Args>
		Ret invoke(Args&&... args) const {
			if constexpr(std::is_invocable_v<Functor, Args...>) {
				// Copy functor once to hydrate captured accessors
				// NOCOMMIT This is too far removed from the hydrator. Tie them together somehow.
				auto fun = m_fun;
				return fun(args...);
			} else {
				throw std::runtime_error("Cannot launch command function with provided arguments");
			}
		}
	};

	class contiguous_box_set : private std::vector<GridBox<3>> {
	  private:
		using vector = std::vector<GridBox<3>>;

	  public:
		using typename vector::const_iterator;
		using typename vector::value_type;
		using iterator = const_iterator;

		contiguous_box_set() = default;

		using vector::empty;
		using vector::size;
		using vector::swap;

		iterator begin() const { return vector::begin(); } // only export const overload
		iterator end() const { return vector::end(); }     // only export const overload

		void insert(const GridBox<3>& box);

		template <typename Iterator>
		void insert(const Iterator first, const Iterator last) {
			while(first != last) {
				insert(*first++);
			}
		}

		vector into_vector() && { return std::move(*this); }
	};

	class buffer_access_map {
	  public:
		void add_access(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { m_accesses.emplace_back(bid, std::move(rm)); }

		std::unordered_set<buffer_id> get_accessed_buffers() const;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const;
		size_t get_num_accesses() const { return m_accesses.size(); }
		std::pair<buffer_id, access_mode> get_nth_access(const size_t n) const {
			const auto& [bid, rm] = m_accesses[n];
			return {bid, rm->get_access_mode()};
		}

		/**
		 * @brief Computes the combined access-region for a given buffer, mode and subrange.
		 *
		 * @param bid
		 * @param mode
		 * @param sr The subrange to be passed to the range mappers (extended to a chunk using the global size of the task)
		 *
		 * @returns The region obtained by merging the results of all range-mappers for this buffer and mode
		 */
		GridRegion<3> get_mode_requirements(
		    const buffer_id bid, const access_mode mode, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

		GridBox<3> get_requirements_for_nth_access(const size_t n, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

		std::vector<const range_mapper_base*> get_range_mappers(const buffer_id bid) const {
			std::vector<const range_mapper_base*> rms;
			for(const auto& [a_bid, a_rm] : m_accesses) {
				if(a_bid == bid) { rms.push_back(a_rm.get()); }
			}
			return rms;
		}

		contiguous_box_set get_required_contiguous_boxes(const buffer_id bid, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

	  private:
		std::vector<std::pair<buffer_id, std::unique_ptr<range_mapper_base>>> m_accesses;
	};

	using reduction_set = std::vector<reduction_info>;

	class side_effect_map : private std::unordered_map<host_object_id, experimental::side_effect_order> {
	  private:
		using map_base = std::unordered_map<host_object_id, experimental::side_effect_order>;

	  public:
		using typename map_base::const_iterator, map_base::value_type, map_base::key_type, map_base::mapped_type, map_base::const_reference,
		    map_base::const_pointer;
		using iterator = const_iterator;
		using reference = const_reference;
		using pointer = const_pointer;

		using map_base::size, map_base::count, map_base::empty, map_base::cbegin, map_base::cend, map_base::at;

		iterator begin() const { return cbegin(); }
		iterator end() const { return cend(); }
		iterator find(host_object_id key) const { return map_base::find(key); }

		void add_side_effect(host_object_id hoid, experimental::side_effect_order order);
	};

	struct task_geometry {
		int dimensions = 0;
		cl::sycl::range<3> global_size{0, 0, 0};
		cl::sycl::id<3> global_offset{};
		cl::sycl::range<3> granularity{1, 1, 1};

		friend bool operator==(const task_geometry& lhs, const task_geometry& rhs) {
			return lhs.dimensions == rhs.dimensions          //
			       && lhs.global_size == rhs.global_size     //
			       && lhs.global_offset == rhs.global_offset //
			       && lhs.granularity == rhs.granularity;
		}

		friend bool operator!=(const task_geometry& lhs, const task_geometry& rhs) { return !(rhs == lhs); }
	};

	class task : public intrusive_graph_node<task> {
	  public:
		virtual ~task() = 0;

		task_type get_type() const { return m_type; }

		task_id get_id() const { return m_tid; }

		virtual execution_target get_execution_target() const { return execution_target::none; }

	  protected:
		explicit task(const task_id tid, const task_type type) : m_tid(tid), m_type(type) {}

		// derived classes are copy/move-constructible and assignable, but this base class is not
		task(const task&) = default;
		task(task&&) = default;
		task& operator=(const task&) = default;
		task& operator=(task&&) = default;

	  private:
		task_id m_tid;
		task_type m_type;
	};

	inline task::~task() = default;

	class epoch_task final : public task {
	  public:
		explicit epoch_task(const task_id tid, const detail::epoch_action epoch_action) : task(tid, task_type::epoch), m_epoch_action(epoch_action) {}

		epoch_action get_epoch_action() const { return m_epoch_action; }

	  private:
		detail::epoch_action m_epoch_action;
	};

	class horizon_task final : public task {
	  public:
		explicit horizon_task(const task_id tid) : task(tid, task_type::horizon) {}
	};

	struct split_constraints {
		split_constraints(const task_geometry& task_geometry, const bool require_tiled_split)
		    : m_task_geometry(task_geometry), m_tiled_split(require_tiled_split), m_split_geometry(task_geometry.dimensions) {
			if(task_geometry.dimensions > 0) { m_split_geometry[0] = experimental::split_geometry::split; }
			if(task_geometry.dimensions > 1) { m_split_geometry[1] = require_tiled_split ? experimental::split_geometry::split : experimental::constant; }
			if(task_geometry.dimensions > 2) { m_split_geometry[2] = experimental::constant; }
			m_is_splittable = (task_geometry.global_size > task_geometry.granularity) != sycl::id<3>(false, false, false);
		}

		friend bool operator==(const split_constraints& lhs, const split_constraints& rhs) {
			return lhs.m_task_geometry == rhs.m_task_geometry //
			       && lhs.m_tiled_split == rhs.m_tiled_split;
		}
		friend bool operator!=(const split_constraints& lhs, const split_constraints& rhs) { return !(rhs == lhs); }

		task_geometry get_task_geometry() const { return m_task_geometry; }

		const experimental::split_geometry_map& get_split_geometry() const { return m_split_geometry; }

		bool is_splittable() const { return m_is_splittable; }

		bool requires_tiled_split() const { return m_tiled_split; }

	  private:
		// v-- inputs
		detail::task_geometry m_task_geometry;
		bool m_tiled_split;
		// oversubscription is not a concern because it is applied after the initial internode split

		// v-- generated
		experimental::split_geometry_map m_split_geometry;
		bool m_is_splittable;
	};

	class command_group_task final : public task {
	  public:
		using launcher = std::variant<std::monostate, sycl_kernel_launcher, host_task_launcher>;

		collective_group_id get_collective_group_id() const { return m_cgid; }

		const buffer_access_map& get_buffer_access_map() const { return m_access_map; }

		const side_effect_map& get_side_effect_map() const { return m_side_effects; }

		const task_geometry& get_geometry() const { return m_geometry; }

		int get_dimensions() const { return m_geometry.dimensions; }

		cl::sycl::range<3> get_global_size() const { return m_geometry.global_size; }

		cl::sycl::id<3> get_global_offset() const { return m_geometry.global_offset; }

		cl::sycl::range<3> get_granularity() const { return m_geometry.granularity; }

		const std::string& get_debug_name() const { return m_debug_name; }

		// TODO this is currently used to determine whether a task can be split at all, but it reports false for collective-host-tasks, where it used to mean
		//  that naive_split_transformer cannot be applied on top of the "natural" one-item-per-node split for these tasks
		bool has_variable_split() const {
			return (get_type() == task_type::host_compute || get_type() == task_type::device_compute)
			       && (m_geometry.global_size.size() > m_geometry.granularity.size());
		}

		execution_target get_execution_target() const {
			switch(get_type()) {
			case task_type::device_compute: return execution_target::device;
			case task_type::host_compute:
			case task_type::host_collective:
			case task_type::master_node: return execution_target::host;
			default: assert(!"Unhandled task type"); return execution_target::none;
			}
		}

		const reduction_set& get_reductions() const { return m_reductions; }

		// NOCOMMIT TODO We can do better with typings here...
		template <typename... Args>
		auto launch(Args&&... args) const {
			return (*m_c_launcher)(std::forward<Args>(args)...);
		}

		const launcher& get_launcher() const { return m_launcher; }

		// TODO: What happens when the same hint is added several times?
		// TODO: Are there combinations of hints that aren't allowed? Where would that be validated?
		void add_hint(std::unique_ptr<hint_base>&& h) { m_hints.emplace_back(std::move(h)); }

		// TODO: Should we be able to get hints at all, or should they manifest in property changes on the task itself?
		template <typename Hint>
		const Hint* get_hint() const {
			static_assert(std::is_base_of_v<hint_base, Hint>, "Hint must extend hint_base");
			for(auto& h : m_hints) {
				if(auto* ptr = dynamic_cast<Hint*>(h.get()); ptr != nullptr) { return ptr; }
			}
			return nullptr;
		}

		split_constraints get_split_constraints() const;

		static std::unique_ptr<command_group_task> make_host_compute(task_id tid, task_geometry geometry,
		    std::unique_ptr<command_launcher_storage_base> c_launcher, host_task_launcher launcher, buffer_access_map access_map,
		    side_effect_map side_effect_map, reduction_set reductions) {
			return std::unique_ptr<command_group_task>(new command_group_task(tid, task_type::host_compute, non_collective, geometry, std::move(c_launcher),
			    std::move(launcher), std::move(access_map), std::move(side_effect_map), std::move(reductions), {}));
		}

		static std::unique_ptr<command_group_task> make_device_compute(task_id tid, task_geometry geometry,
		    std::unique_ptr<command_launcher_storage_base> c_launcher, sycl_kernel_launcher launcher, buffer_access_map access_map, reduction_set reductions,
		    std::string debug_name) {
			return std::unique_ptr<command_group_task>(new command_group_task(tid, task_type::device_compute, non_collective, geometry, std::move(c_launcher),
			    std::move(launcher), std::move(access_map), {}, std::move(reductions), std::move(debug_name)));
		}

		static std::unique_ptr<command_group_task> make_host_collective(task_id tid, collective_group_id cgid, size_t num_collective_nodes,
		    std::unique_ptr<command_launcher_storage_base> c_launcher, host_task_launcher launcher, buffer_access_map access_map,
		    side_effect_map side_effect_map) {
			const task_geometry geometry{1, detail::range_cast<3>(cl::sycl::range<1>{num_collective_nodes}), {}, {1, 1, 1}};
			return std::unique_ptr<command_group_task>(new command_group_task(tid, task_type::host_collective, cgid, geometry, std::move(c_launcher),
			    std::move(launcher), std::move(access_map), std::move(side_effect_map), {}, {}));
		}

		static std::unique_ptr<command_group_task> make_master_node(task_id tid, std::unique_ptr<command_launcher_storage_base> c_launcher,
		    host_task_launcher launcher, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<command_group_task>(new command_group_task(tid, task_type::master_node, non_collective, task_geometry{},
			    std::move(c_launcher), std::move(launcher), std::move(access_map), std::move(side_effect_map), {}, {}));
		}

	  private:
		collective_group_id m_cgid;
		task_geometry m_geometry;
		std::unique_ptr<command_launcher_storage_base> m_c_launcher; // TODO remove
		std::variant<std::monostate, sycl_kernel_launcher, host_task_launcher> m_launcher;
		buffer_access_map m_access_map;
		detail::side_effect_map m_side_effects;
		reduction_set m_reductions;
		std::string m_debug_name;
		std::vector<std::unique_ptr<hint_base>> m_hints;

		command_group_task(task_id tid, task_type type, collective_group_id cgid, task_geometry geometry,
		    std::unique_ptr<command_launcher_storage_base> c_launcher, launcher launcher, buffer_access_map access_map, detail::side_effect_map side_effects,
		    reduction_set reductions, std::string debug_name)
		    : task(tid, type), m_cgid(cgid), m_geometry(geometry), m_launcher(std::move(launcher)), m_access_map(std::move(access_map)),
		      m_side_effects(std::move(side_effects)), m_reductions(std::move(reductions)), m_debug_name(std::move(debug_name)) {
			assert(type == task_type::host_compute || type == task_type::device_compute || get_granularity().size() == 1);
			// Only host tasks can have side effects
			assert(this->m_side_effects.empty() || type == task_type::host_compute || type == task_type::host_collective || type == task_type::master_node);
		}
	};

	class forward_task final : public task {
	  public:
		struct access {
			split_constraints constraints;
			std::vector<const range_mapper_base*> range_mappers;
		};

		forward_task(const task_id tid, const buffer_id bid, GridRegion<3> region, access producer, access consumer)
		    : task(tid, task_type::forward), m_bid(bid), m_region(std::move(region)), m_producer(std::move(producer)), m_consumer(std::move(consumer)) {}

		const buffer_id& get_bid() const { return m_bid; }
		const GridRegion<3>& get_region() const { return m_region; }
		const access& get_producer() const { return m_producer; }
		const access& get_consumer() const { return m_consumer; }

	  private:
		buffer_id m_bid;
		GridRegion<3> m_region;
		access m_producer;
		access m_consumer;
	};

	inline subrange<3> apply_range_mapper(range_mapper_base const* rm, const task_geometry& geometry) {
		return apply_range_mapper(rm, chunk<3>{geometry.global_offset, geometry.global_size, geometry.global_size}, geometry.dimensions);
	}

} // namespace detail
} // namespace celerity
