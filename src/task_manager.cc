#include "task_manager.h"

#include "access_modes.h"
#include "print_graph.h"

namespace celerity {
namespace detail {

	task_manager::task_manager(size_t num_collective_nodes, host_queue* queue) : m_num_collective_nodes(num_collective_nodes), m_queue(queue) {
		// We manually generate the initial epoch task, which we treat as if it has been reached immediately.
		auto reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
		m_task_buffer.put(std::move(reserve), task::make_epoch(initial_epoch_task, epoch_action::none));
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		m_buffers_last_writers.emplace(bid, range);
		if(host_initialized) { m_buffers_last_writers.at(bid).update_region(subrange_to_grid_box(subrange<3>({}, range)), m_epoch_for_new_tasks); }
	}

	const task* task_manager::find_task(task_id tid) const { return m_task_buffer.find_task(tid); }

	bool task_manager::has_task(task_id tid) const { return m_task_buffer.has_task(tid); }

	// Note that we assume tasks are not modified after their initial creation, which is why
	// we don't need to worry about thread-safety after returning the task pointer.
	const task* task_manager::get_task(task_id tid) const { return m_task_buffer.get_task(tid); }

	std::optional<std::string> task_manager::print_graph(size_t max_nodes) const {
		if(m_task_buffer.get_current_task_count() <= max_nodes) { return detail::print_task_graph(m_task_buffer, nullptr); }
		return std::nullopt;
	}

	void task_manager::notify_horizon_reached(task_id horizon_tid) {
		// m_latest_horizon_reached does not need synchronization (see definition), all other accesses are implicitly synchronized.

		assert(m_task_buffer.get_task(horizon_tid)->get_type() == task_type::horizon);
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < horizon_tid);
		assert(m_latest_epoch_reached.get() < horizon_tid);

		if(m_latest_horizon_reached) { m_latest_epoch_reached.set(*m_latest_horizon_reached); }

		m_latest_horizon_reached = horizon_tid;
	}

	void task_manager::notify_epoch_reached(task_id epoch_tid) {
		// m_latest_horizon_reached does not need synchronization (see definition), all other accesses are implicitly synchronized.

		assert(get_task(epoch_tid)->get_type() == task_type::epoch);
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < epoch_tid);
		assert(m_latest_epoch_reached.get() < epoch_tid);

		m_latest_epoch_reached.set(epoch_tid);
		m_latest_horizon_reached = std::nullopt; // Any non-applied horizon is now behind the epoch and will therefore never become an epoch itself
	}

	void task_manager::await_epoch(task_id epoch) { m_latest_epoch_reached.await(epoch); }

	GridRegion<3> get_requirements(task const& tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		const auto& access_map = tsk.get_buffer_access_map();
		const subrange<3> full_range{tsk.get_global_offset(), tsk.get_global_size()};
		GridRegion<3> result;
		for(auto m : modes) {
			result = GridRegion<3>::merge(result, access_map.get_mode_requirements(bid, m, tsk.get_dimensions(), full_range, tsk.get_global_size()));
		}
		return result;
	}

	void task_manager::compute_dependencies(task& tsk) {
		using namespace cl::sycl::access;

		const auto& access_map = tsk.get_buffer_access_map();

		auto buffers = access_map.get_accessed_buffers();
		for(const auto& reduction : tsk.get_reductions()) {
			buffers.emplace(reduction.bid);
		}

		for(const auto bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);

			std::optional<reduction_info> reduction;
			for(const auto& maybe_reduction : tsk.get_reductions()) {
				if(maybe_reduction.bid == bid) {
					if(reduction) { throw std::runtime_error(fmt::format("Multiple reductions attempt to write buffer {} in task {}", bid, tsk.get_id())); }
					reduction = maybe_reduction;
				}
			}

			if(reduction && !modes.empty()) {
				throw std::runtime_error(
				    fmt::format("Buffer {} is both required through an accessor and used as a reduction output in task {}", bid, tsk.get_id()));
			}

			// Determine reader dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_consumer) || (reduction.has_value() && reduction->init_from_buffer)) {
				auto read_requirements = get_requirements(tsk, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
				if(reduction.has_value()) { read_requirements = GridRegion<3>::merge(read_requirements, GridRegion<3>{{1, 1, 1}}); }
				const auto last_writers = m_buffers_last_writers.at(bid).get_region_values(read_requirements);

				for(auto& p : last_writers) {
					// This indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(p.second == std::nullopt) continue;
					const task_id last_writer = *p.second;
					add_dependency(tsk, *m_task_buffer.get_task(last_writer), dependency_kind::true_dep, dependency_origin::dataflow);
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_producer) || reduction.has_value()) {
				auto write_requirements = get_requirements(tsk, bid, {detail::access::producer_modes.cbegin(), detail::access::producer_modes.cend()});
				if(reduction.has_value()) { write_requirements = GridRegion<3>::merge(write_requirements, GridRegion<3>{{1, 1, 1}}); }
				if(write_requirements.empty()) continue;

				const auto last_writers = m_buffers_last_writers.at(bid).get_region_values(write_requirements);
				for(auto& p : last_writers) {
					if(p.second == std::nullopt) continue;
					task* last_writer = m_task_buffer.get_task(*p.second);

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(auto dependent : last_writer->get_dependents()) {
						if(dependent.node->get_id() == tsk.get_id()) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements =
						    get_requirements(*dependent.node, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
							add_dependency(tsk, *dependent.node, dependency_kind::anti_dep, dependency_origin::dataflow);
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tsk, *last_writer, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}

				m_buffers_last_writers.at(bid).update_region(write_requirements, tsk.get_id());
			}
		}

		for(const auto& side_effect : tsk.get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			if(const auto last_effect = m_host_object_last_effects.find(hoid); last_effect != m_host_object_last_effects.end()) {
				add_dependency(tsk, *m_task_buffer.get_task(last_effect->second), dependency_kind::true_dep, dependency_origin::dataflow);
			}
			m_host_object_last_effects.insert_or_assign(hoid, tsk.get_id());
		}

		if(auto cgid = tsk.get_collective_group_id(); cgid != 0) {
			if(auto prev = m_last_collective_tasks.find(cgid); prev != m_last_collective_tasks.end()) {
				add_dependency(tsk, *m_task_buffer.get_task(prev->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				m_last_collective_tasks.erase(prev);
			}
			m_last_collective_tasks.emplace(cgid, tsk.get_id());
		}

		// Tasks without any other true-dependency must depend on the last epoch to ensure they cannot be re-ordered before the epoch
		if(const auto deps = tsk.get_dependencies();
		    std::none_of(deps.begin(), deps.end(), [](const task::dependency d) { return d.kind == dependency_kind::true_dep; })) {
			add_dependency(tsk, *m_task_buffer.get_task(m_epoch_for_new_tasks), dependency_kind::true_dep, dependency_origin::last_epoch);
		}
	}

	task& task_manager::register_task_internal(task_ring_buffer::reservation&& reserve, std::unique_ptr<task> task) {
		auto& task_ref = *task;
		assert(task != nullptr);
		m_task_buffer.put(std::move(reserve), std::move(task));
		m_execution_front.insert(&task_ref);
		return task_ref;
	}

	void task_manager::invoke_callbacks(const task* tsk) const {
		for(const auto& cb : m_task_callbacks) {
			cb(tsk);
		}
	}

	void task_manager::add_dependency(task& depender, task& dependee, dependency_kind kind, dependency_origin origin) {
		assert(&depender != &dependee);
		depender.add_dependency({&dependee, kind, origin});
		m_execution_front.erase(&dependee);
		m_max_pseudo_critical_path_length = std::max(m_max_pseudo_critical_path_length, depender.get_pseudo_critical_path_length());
	}

	task& task_manager::reduce_execution_front(task_ring_buffer::reservation&& reserve, std::unique_ptr<task> new_front) {
		// add dependencies from a copy of the front to this task
		const auto current_front = m_execution_front;
		for(task* front_task : current_front) {
			add_dependency(*new_front, *front_task, dependency_kind::true_dep, dependency_origin::execution_front);
		}
		assert(m_execution_front.empty());
		return register_task_internal(std::move(reserve), std::move(new_front));
	}

	void task_manager::set_epoch_for_new_tasks(const task_id epoch) {
		// apply the new epoch to buffers_last_writers and last_collective_tasks data structs
		for(auto& [_, buffer_region_map] : m_buffers_last_writers) {
			buffer_region_map.apply_to_values([epoch](const std::optional<task_id> tid) -> std::optional<task_id> {
				if(!tid) return tid;
				return {std::max(epoch, *tid)};
			});
		}
		for(auto& [cgid, tid] : m_last_collective_tasks) {
			tid = std::max(epoch, tid);
		}
		for(auto& [hoid, tid] : m_host_object_last_effects) {
			tid = std::max(epoch, tid);
		}

		m_epoch_for_new_tasks = epoch;
	}

	task_id task_manager::generate_horizon_task() {
		auto reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
		const auto tid = reserve.get_tid();

		m_current_horizon_critical_path_length = m_max_pseudo_critical_path_length;
		const auto previous_horizon = m_current_horizon;
		m_current_horizon = tid;

		task& new_horizon = reduce_execution_front(std::move(reserve), task::make_horizon_task(*m_current_horizon));
		if(previous_horizon) { set_epoch_for_new_tasks(*previous_horizon); }

		invoke_callbacks(&new_horizon);
		return tid;
	}

	task_id task_manager::generate_epoch_task(epoch_action action) {
		auto reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
		const auto tid = reserve.get_tid();

		task& new_epoch = reduce_execution_front(std::move(reserve), task::make_epoch(tid, action));
		compute_dependencies(new_epoch);
		set_epoch_for_new_tasks(tid);

		m_current_horizon = std::nullopt; // this horizon is now behind the epoch_for_new_tasks, so it will never become an epoch itself
		m_current_horizon_critical_path_length = m_max_pseudo_critical_path_length; // the explicit epoch resets the need to create horizons

		invoke_callbacks(&new_epoch);
		return tid;
	}

	task_id task_manager::get_first_in_flight_epoch() const {
		task_id current_horizon = 0;
		task_id latest_epoch = m_latest_epoch_reached.get();
		// we need either one epoch or two horizons that have yet to be executed
		// so that it is possible for task slots to be freed in the future
		for(const auto& tsk : m_task_buffer) {
			if(tsk->get_id() <= latest_epoch) continue;
			if(tsk->get_type() == task_type::epoch) {
				return tsk->get_id();
			} else if(tsk->get_type() == task_type::horizon) {
				if(current_horizon) return current_horizon;
				current_horizon = tsk->get_id();
			}
		}
		return latest_epoch;
	}

	task_ring_buffer::wait_callback task_manager::await_free_task_slot_callback() {
		return [&](task_id previous_free_tid) {
			if(get_first_in_flight_epoch() == m_latest_epoch_reached.get()) {
				// verify that the epoch didn't get reached between the invocation of the callback and the in flight check
				if(m_latest_epoch_reached.get() < previous_free_tid + 1) {
					throw std::runtime_error("Exhausted task slots with no horizons or epochs in flight."
					                         "\nLikely due to generating a very large number of tasks with no dependencies.");
				}
			}
			task_id reached_epoch = m_latest_epoch_reached.await(previous_free_tid + 1);
			m_task_buffer.delete_up_to(reached_epoch);
		};
	}

	void task_manager::infer_collective_data_requirements(task& tsk) {
		struct gather_dependency {
			buffer_id bid;
			subrange<3> consumed_sr;
			const range_mapper_base* constant_consumer_rm; // not redundant with consumed_sr because RM is dimensionality-erased
			task* identity_producer;
			const range_mapper_base* identity_producer_rm;
		};
		std::vector<gather_dependency> pending_gathers;

		for(const auto bid : tsk.get_buffer_access_map().get_accessed_buffers()) {
			task* const consumer = &tsk;
			const range_mapper_base* consumer_rm = nullptr;
			size_t num_consumer_accesses = 0;
			for(const auto* rm : consumer->get_buffer_access_map().get_range_mappers(bid)) {
				if(detail::access::mode_traits::is_consumer(rm->get_access_mode())) {
					consumer_rm = rm;
					num_consumer_accesses += 1;
				}
			}

			// TODO allow multiple non-overlapping consumers of the same buffer.
			if(num_consumer_accesses != 1 || !consumer_rm->is_constant()) continue;

			const auto& consumer_geometry = consumer->get_geometry();
			const chunk<3> consumer_chunk{consumer_geometry.global_offset, consumer_geometry.global_size, consumer_geometry.global_size};
			const auto consumed_sr = apply_range_mapper(consumer_rm, consumer_chunk, consumer_geometry.dimensions);

			task* part_producer = nullptr;
			task* identity_producer = nullptr;
			const range_mapper_base* identity_producer_rm = nullptr;
			for(const auto& dep : consumer->get_dependencies()) {
				// Don't get confused by epoch serialization dependencies or similar
				if(dep.kind != dependency_kind::true_dep || dep.origin != dependency_origin::dataflow) continue;

				for(const auto* rm : dep.node->get_buffer_access_map().get_range_mappers(bid)) {
					if(!detail::access::mode_traits::is_producer(rm->get_access_mode())) continue;

					task* const buffer_producer = dep.node;
					const auto& producer_geometry = buffer_producer->get_geometry();
					const chunk<3> producer_chunk{producer_geometry.global_offset, producer_geometry.global_size, producer_geometry.global_size};
					const auto produced_sr = apply_range_mapper(rm, producer_chunk, producer_geometry.dimensions);

					// There can be redundant true-dependencies in the graph, so we skip the buffer if we see multiple producers for our subrange
					const auto producer_box = subrange_to_grid_box(produced_sr);
					const auto consumer_box = subrange_to_grid_box(consumed_sr);
					if(producer_box.intersectsWith(consumer_box) && part_producer == nullptr) { part_producer = buffer_producer; }

					// TODO also require that producer is splittable
					if(part_producer == buffer_producer && rm->is_identity()) {
						assert(identity_producer == nullptr);
						identity_producer = part_producer;
						identity_producer_rm = rm;
					}
				}
			}

			if(identity_producer == nullptr) continue;

			// We stage our changes in this vector to avoid working on a partially-modified TDAG later in subsequent iterations of the loop
			pending_gathers.push_back({bid, consumed_sr, consumer_rm, identity_producer, identity_producer_rm});
		}

		for(const auto& gather : pending_gathers) {
			task* const consumer = &tsk;

			const auto& producer_geometry = gather.identity_producer->get_geometry();
			const auto buffer_dimensions = producer_geometry.dimensions; // since we require a one-to-one access
			const task_geometry geometry{buffer_dimensions, gather.consumed_sr.range, gather.consumed_sr.offset};

			// TODO is it enough to define the gather through buffer requirements only? Do we need to specify a task geometry?
			buffer_access_map access_map;
			access_map.add_access(gather.bid, gather.identity_producer_rm->clone_as(access_mode::read));
			access_map.add_access(gather.bid, gather.constant_consumer_rm->clone_as(access_mode::discard_write));

			auto gather_reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
			auto gather_task_ptr = task::make_gather(gather_reserve.get_tid(), geometry, std::move(access_map));
			auto& gather_task = *gather_task_ptr;
			m_task_buffer.put(std::move(gather_reserve), std::move(gather_task_ptr));

			// TODO investigate if this has any consequence for PCPL and horizon generation
			consumer->remove_dependency(gather.identity_producer);
			add_dependency(*consumer, gather_task, dependency_kind::true_dep, dependency_origin::dataflow);
			add_dependency(gather_task, *gather.identity_producer, dependency_kind::true_dep, dependency_origin::dataflow);

			m_buffers_last_writers.at(gather.bid).update_region(subrange_to_grid_box(gather.consumed_sr), gather_task.get_id());

			invoke_callbacks(&gather_task);
			// We never want to generate additional horizons for nodes inserted in a graph transformation like this
		}
	}
} // namespace detail
} // namespace celerity
