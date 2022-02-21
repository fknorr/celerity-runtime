#include "task_manager.h"

#include "access_modes.h"
#include "print_graph.h"

namespace celerity {
namespace detail {

	task_manager::task_manager(size_t num_collective_nodes, host_queue* queue, reduction_manager* reduction_mgr)
	    : num_collective_nodes(num_collective_nodes), queue(queue), reduction_mngr(reduction_mgr) {
		// We manually generate the initial epoch task, which we treat as immediately-completed.
		task_map.emplace(initial_epoch_task, task::make_epoch(initial_epoch_task, {}, {}, epoch_action::none));
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		std::lock_guard<std::mutex> lock(task_mutex);
		buffers_last_writers.emplace(bid, range);
		if(host_initialized) { buffers_last_writers.at(bid).update_region(subrange_to_grid_box(subrange<3>({}, range)), current_epoch); }
	}

	// Note that we assume tasks are not modified after their initial creation, which is why
	// we don't need to worry about thread-safety after returning the task pointer.
	const task* task_manager::find_task(const task_id tid) const {
		const std::lock_guard lock{task_mutex};
		if(const auto it = task_map.find(tid); it != task_map.end()) {
			return it->second.get();
		} else {
			return nullptr;
		}
	}

	bool task_manager::has_task(const task_id tid) const { return find_task(tid) != nullptr; }

	const task* task_manager::get_task(const task_id tid) const {
		const auto tsk = find_task(tid);
		assert(tsk != nullptr);
		return tsk;
	}

	std::optional<std::string> task_manager::print_graph(size_t max_nodes) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		if(task_map.size() <= max_nodes) { return detail::print_task_graph(task_map); }
		return std::nullopt;
	}

	void task_manager::notify_horizon_completed(task_id horizon_tid) {
		// This method is called from the executor thread, but does not lock task_mutex to avoid lock-step execution with the main thread.
		// last_complete_horizon does not need synchronization (see definition), all other accesses are implicitly synchronized.

		assert(get_task(horizon_tid)->get_type() == task_type::HORIZON);
		assert(!last_completed_horizon || *last_completed_horizon < horizon_tid);
		assert(last_completed_epoch.get() < horizon_tid);

		if(last_completed_horizon) {
			last_completed_epoch.set(*last_completed_horizon); // The next call to create_task() will prune all tasks before the last completed epoch
		}
		last_completed_horizon = horizon_tid;
	}

	void task_manager::notify_epoch_completed(task_id epoch_tid) {
		// This method is called from the executor thread, but does not lock task_mutex to avoid lock-step execution with the main thread.
		// last_complete_horizon does not need synchronization (see definition), all other accesses are implicitly synchronized.

		assert(get_task(epoch_tid)->get_type() == task_type::EPOCH);
		assert(!last_completed_horizon || *last_completed_horizon < epoch_tid);
		assert(last_completed_epoch.get() < epoch_tid);

		last_completed_epoch.set(epoch_tid);   // The next call to create_task() will prune all tasks before the last completed epoch
		last_completed_horizon = std::nullopt; // The last completed horizon is now behind the current epoch and will therefore never become an epoch itself
	}

	void task_manager::await_epoch_completed(task_id epoch) { last_completed_epoch.await(epoch); }

	GridRegion<3> get_requirements(task const* tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		const auto& access_map = tsk->get_buffer_access_map();
		const subrange<3> full_range{tsk->get_global_offset(), tsk->get_global_size()};
		GridRegion<3> result;
		for(auto m : modes) {
			result = GridRegion<3>::merge(result, access_map.get_requirements_for_access(bid, m, tsk->get_dimensions(), full_range, tsk->get_global_size()));
		}
		return result;
	}

	void task_manager::compute_dependencies(task_id tid) {
		using namespace cl::sycl::access;

		const auto& tsk = task_map[tid];
		const auto& access_map = tsk->get_buffer_access_map();

		auto buffers = access_map.get_accessed_buffers();
		for(auto rid : tsk->get_reductions()) {
			assert(reduction_mngr != nullptr);
			buffers.emplace(reduction_mngr->get_reduction(rid).output_buffer_id);
		}

		for(const auto bid : buffers) {
			auto modes = access_map.get_access_modes(bid);
			if(tsk->get_buffer_capture_map().reads(bid)) { modes.insert(access_mode::read); }

			std::optional<reduction_info> reduction;
			for(auto maybe_rid : tsk->get_reductions()) {
				auto maybe_reduction = reduction_mngr->get_reduction(maybe_rid);
				if(maybe_reduction.output_buffer_id == bid) {
					if(reduction) { throw std::runtime_error(fmt::format("Multiple reductions attempt to write buffer {} in task {}", bid, tid)); }
					reduction = maybe_reduction;
				}
			}

			if(reduction && !modes.empty()) {
				throw std::runtime_error(fmt::format("Buffer {} is both required through an accessor and used as a reduction output in task {}", bid, tid));
			}

			// Determine reader dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_consumer)
			    || (reduction.has_value() && reduction->initialize_from_buffer)) {
				auto read_requirements = get_requirements(tsk.get(), bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
				if(reduction.has_value()) { read_requirements = GridRegion<3>::merge(read_requirements, GridRegion<3>{{1, 1, 1}}); }
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(read_requirements);

				for(auto& p : last_writers) {
					// This indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(p.second == std::nullopt) continue;
					const task_id last_writer = *p.second;
					assert(task_map.count(last_writer) == 1);
					add_dependency(tsk.get(), task_map[last_writer].get(), dependency_kind::TRUE_DEP, dependency_origin::dataflow);
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_producer) || reduction.has_value()) {
				auto write_requirements = get_requirements(tsk.get(), bid, {detail::access::producer_modes.cbegin(), detail::access::producer_modes.cend()});
				if(reduction.has_value()) { write_requirements = GridRegion<3>::merge(write_requirements, GridRegion<3>{{1, 1, 1}}); }
				if(write_requirements.empty()) continue;

				const auto last_writers = buffers_last_writers.at(bid).get_region_values(write_requirements);
				for(auto& p : last_writers) {
					if(p.second == std::nullopt) continue;
					assert(task_map.count(*p.second) == 1);
					auto& last_writer = *task_map[*p.second];

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(auto dependent : last_writer.get_dependents()) {
						if(dependent.node->get_id() == tid) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements =
						    get_requirements(dependent.node, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
							add_dependency(tsk.get(), dependent.node, dependency_kind::ANTI_DEP, dependency_origin::dataflow);
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tsk.get(), &last_writer, dependency_kind::ANTI_DEP, dependency_origin::dataflow);
					}
				}

				buffers_last_writers.at(bid).update_region(write_requirements, tid);
			}
		}

		for(const auto& side_effect : tsk->get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			if(const auto last_effect = host_object_last_effects.find(hoid); last_effect != host_object_last_effects.end()) {
				add_dependency(tsk.get(), task_map.at(last_effect->second).get(), dependency_kind::TRUE_DEP, dependency_origin::dataflow);
			}
			host_object_last_effects.insert_or_assign(hoid, tid);
		}

		if(auto cgid = tsk->get_collective_group_id(); cgid != 0) {
			if(auto prev = last_collective_tasks.find(cgid); prev != last_collective_tasks.end()) {
				add_dependency(tsk.get(), task_map.at(prev->second).get(), dependency_kind::TRUE_DEP, dependency_origin::collective_group_serialization);
				last_collective_tasks.erase(prev);
			}
			last_collective_tasks.emplace(cgid, tid);
		}

		// Tasks without any other true-dependency must depend on the current epoch to ensure they cannot be re-ordered before the epoch
		if(const auto deps = tsk->get_dependencies();
		    std::none_of(deps.begin(), deps.end(), [](const task::dependency d) { return d.kind == dependency_kind::TRUE_DEP; })) {
			add_dependency(tsk.get(), task_map.at(current_epoch).get(), dependency_kind::TRUE_DEP, dependency_origin::current_epoch);
		}
	}

	task& task_manager::register_task_internal(std::unique_ptr<task> task) {
		auto& task_ref = *task;
		assert(task != nullptr);
		task_map.emplace(task->get_id(), std::move(task));
		execution_front.insert(&task_ref);
		return task_ref;
	}

	void task_manager::invoke_callbacks(const task_id tid) {
		for(const auto& cb : task_callbacks) {
			cb(tid);
		}
	}

	void task_manager::add_dependency(task* depender, task* dependee, dependency_kind kind, dependency_origin origin) {
		assert(depender != dependee);
		assert(depender != nullptr && dependee != nullptr);
		depender->add_dependency({dependee, kind, origin});
		execution_front.erase(dependee);
		max_pseudo_critical_path_length = std::max(max_pseudo_critical_path_length, depender->get_pseudo_critical_path_length());
	}

	task_id task_manager::collect_execution_front(std::unique_ptr<task> reducer) {
		// add dependencies from a copy of the front to this task
		const auto current_front = execution_front;
		for(task* front_task : current_front) {
			add_dependency(reducer.get(), front_task, dependency_kind::TRUE_DEP, dependency_origin::execution_front);
		}
		assert(execution_front.empty());
		return register_task_internal(std::move(reducer)).get_id();
	}

	void task_manager::set_current_epoch(const task_id epoch) {
		// apply the new epoch to buffers_last_writers and last_collective_tasks data structs
		for(auto& [_, buffer_region_map] : buffers_last_writers) {
			buffer_region_map.apply_to_values([epoch](const std::optional<task_id> tid) -> std::optional<task_id> {
				if(!tid) return tid;
				return {std::max(epoch, *tid)};
			});
		}
		for(auto& [cgid, tid] : last_collective_tasks) {
			tid = std::max(epoch, tid);
		}
		for(auto& [hoid, tid] : host_object_last_effects) {
			tid = std::max(epoch, tid);
		}

		current_epoch = epoch;
	}

	void task_manager::generate_task_horizon() {
		// we are probably overzealous in locking here
		{
			std::lock_guard lock(task_mutex);
			current_horizon_critical_path_length = max_pseudo_critical_path_length;
			const auto previous_horizon = current_horizon;
			current_horizon = collect_execution_front(task::make_horizon_task(get_new_tid()));
			if(previous_horizon) { set_current_epoch(*previous_horizon); }
		}

		// it's important that we don't hold the lock while doing this
		invoke_callbacks(*current_horizon);
	}

	task_id task_manager::finish_epoch(epoch_action action, buffer_capture_map capture_map, side_effect_map side_effect_map) {
		// we are probably overzealous in locking here
		task_id tid;
		{
			std::lock_guard lock(task_mutex);
			tid = get_new_tid();
			const auto new_epoch = collect_execution_front(task::make_epoch(tid, std::move(capture_map), std::move(side_effect_map), action));
			compute_dependencies(new_epoch);
			set_current_epoch(new_epoch);
			current_horizon = std::nullopt; // if there is a current horizon, it is now behind the new current epoch, so it will become an current_epoch itself
			current_horizon_critical_path_length = max_pseudo_critical_path_length; // the explicit epoch resets the need to create horizons
		}

		// it's important that we don't hold the lock while doing this
		invoke_callbacks(tid);
		return tid;
	}

	void task_manager::prune_completed_tasks() {
		const auto prune_before = last_completed_epoch.get();
		if(prune_before > last_pruned_before) {
			for(auto iter = task_map.begin(); iter != task_map.end();) {
				if(iter->first < prune_before) {
					iter = task_map.erase(iter);
				} else {
					++iter;
				}
			}
		}
		last_pruned_before = prune_before;
	}

} // namespace detail
} // namespace celerity
