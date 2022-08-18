#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grid.h"
#include "intrusive_graph.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

class handler;

namespace detail {

	enum class task_type {
		epoch,
		host_compute,   ///< host task with explicit global size and celerity-defined split
		device_compute, ///< device compute task
		collective,     ///< host task with implicit 1d global size = #ranks and fixed split
		master_node,    ///< zero-dimensional host task
		horizon,        ///< task horizon
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

	struct command_group_storage_base {
		virtual void operator()(handler& cgh) const = 0;

		virtual ~command_group_storage_base() = default;
	};

	template <typename Functor>
	struct command_group_storage : command_group_storage_base {
		Functor fun;

		command_group_storage(Functor fun) : fun(fun) {}
		void operator()(handler& cgh) const override { fun(cgh); }
	};

	class buffer_access_map {
	  public:
		void add_access(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { m_map.emplace(bid, std::move(rm)); }

		std::unordered_set<buffer_id> get_accessed_buffers() const;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const;

		/**
		 * @brief Computes the combined access-region for a given buffer, mode and subrange.
		 *
		 * @param bid
		 * @param mode
		 * @param sr The subrange to be passed to the range mappers (extended to a chunk using the global size of the task)
		 *
		 * @returns The region obtained by merging the results of all range-mappers for this buffer and mode
		 */
		GridRegion<3> get_requirements_for_access(
		    buffer_id bid, cl::sycl::access::mode mode, int kernel_dims, const subrange<3>& sr, const cl::sycl::range<3>& global_size) const;

	  private:
		std::unordered_multimap<buffer_id, std::unique_ptr<range_mapper_base>> m_map;
	};

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
	};

	class task : public intrusive_graph_node<task> {
	  protected:
		explicit task(const task_id tid, std::string debug_name, buffer_access_map buffer_accesses, side_effect_map side_effects)
		    : m_tid(tid), m_debug_name(std::move(debug_name)), m_buffer_accesses(std::move(buffer_accesses)), m_side_effects(std::move(side_effects)) {}

	  public:
		virtual ~task() = default;

		virtual task_type get_type() const = 0;

		task_id get_id() const { return m_tid; }

		const buffer_access_map& get_buffer_access_map() const { return m_buffer_accesses; }

		const side_effect_map& get_side_effect_map() const { return m_side_effects; }

		const std::string& get_debug_name() const { return m_debug_name; }

		virtual execution_target get_execution_target() const = 0;

	  private:
		task_id m_tid;
		std::string m_debug_name;
		buffer_access_map m_buffer_accesses;
		side_effect_map m_side_effects;
	};

	class horizon_task final : public task {
	  public:
		explicit horizon_task(const task_id tid, std::string debug_name) : task(tid, std::move(debug_name), {}, {}) {}

		task_type get_type() const override { return task_type::horizon; }

		execution_target get_execution_target() const override { return execution_target::none; }
	};

	class epoch_task final : public task {
	  public:
		explicit epoch_task(
		    const task_id tid, std::string debug_name, buffer_access_map buffer_accesses, side_effect_map side_effects, const epoch_action action)
		    : task(tid, std::move(debug_name), std::move(buffer_accesses), std::move(side_effects)), m_action(action) {}

		task_type get_type() const override { return task_type::epoch; }

		execution_target get_execution_target() const override { return execution_target::none; }

		epoch_action get_epoch_action() const { return m_action; }

	  private:
		epoch_action m_action;
	};

	class execution_task : public task {
	  public:
		explicit execution_task(const task_id tid, std::string debug_name, buffer_access_map buffer_accesses, side_effect_map side_effects,
		    std::unique_ptr<command_group_storage_base> cgf)
		    : task(tid, std::move(debug_name), std::move(buffer_accesses), std::move(side_effects)), m_cgf(std::move(cgf)) {}

		const command_group_storage_base& get_command_group() const { return *m_cgf; }

	  private:
		std::unique_ptr<command_group_storage_base> m_cgf;
	};

	class compute_task : public execution_task {
	  public:
		explicit compute_task(const task_id tid, std::string debug_name, buffer_access_map buffer_accesses, side_effect_map side_effects,
		    std::unique_ptr<command_group_storage_base> cgf, task_geometry geometry, std::vector<reduction_id> reductions)
		    : execution_task(tid, std::move(debug_name), std::move(buffer_accesses), std::move(side_effects), std::move(cgf)), m_geometry(geometry),
		      m_reductions(std::move(reductions)) {}

		const task_geometry& get_geometry() const { return m_geometry; }

		int get_dimensions() const { return m_geometry.dimensions; }

		cl::sycl::range<3> get_global_size() const { return m_geometry.global_size; }

		cl::sycl::id<3> get_global_offset() const { return m_geometry.global_offset; }

		cl::sycl::range<3> get_granularity() const { return m_geometry.granularity; }

		const std::vector<reduction_id>& get_reductions() const { return m_reductions; }

	  private:
		task_geometry m_geometry;
		std::vector<reduction_id> m_reductions;
	};

	class device_compute_task final : public compute_task {
	  public:
		using compute_task::compute_task;

		task_type get_type() const override { return task_type::device_compute; }

		execution_target get_execution_target() const override { return execution_target::device; }
	};

	class host_compute_task final : public compute_task {
	  public:
		using compute_task::compute_task;

		task_type get_type() const override { return task_type::host_compute; }

		execution_target get_execution_target() const override { return execution_target::host; }
	};

	class collective_host_task final : public execution_task {
	  public:
		explicit collective_host_task(const task_id tid, std::string debug_name, buffer_access_map buffer_accesses, side_effect_map side_effects,
		    std::unique_ptr<command_group_storage_base> cgf, const collective_group_id cgid)
		    : execution_task(tid, std::move(debug_name), std::move(buffer_accesses), std::move(side_effects), std::move(cgf)), m_cgid(cgid) {}

		task_type get_type() const override { return task_type::collective; }

		collective_group_id get_collective_group_id() const { return m_cgid; }

		execution_target get_execution_target() const override { return execution_target::host; }

	  private:
		collective_group_id m_cgid;
	};

	class master_node_host_task final : public execution_task {
	  public:
		using execution_task::execution_task;

		task_type get_type() const override { return task_type::master_node; }

		execution_target get_execution_target() const override { return execution_target::host; }
	};

} // namespace detail
} // namespace celerity
