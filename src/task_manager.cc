#include "task_manager.h"

#include "access_modes.h"
#include "print_graph.h"

namespace celerity {
namespace detail {

	task_manager::task_manager(size_t num_collective_nodes, host_queue* queue) : m_num_collective_nodes(num_collective_nodes), m_queue(queue) {
		// We manually generate the initial epoch task, which we treat as if it has been reached immediately.
		auto reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
		m_task_buffer.put(std::move(reserve), std::make_unique<epoch_task>(initial_epoch_task, epoch_action::none));
	}

	void task_manager::add_buffer(buffer_id bid, int dimensions, const cl::sycl::range<3>& range, bool host_initialized) {
		m_buffer_info.emplace(bid, buffer_info{dimensions, range});
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

	GridRegion<3> get_requirements(command_group_task const& tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		const auto& access_map = tsk.get_buffer_access_map();
		const subrange<3> full_range{tsk.get_global_offset(), tsk.get_global_size()};
		GridRegion<3> result;
		for(auto m : modes) {
			result = GridRegion<3>::merge(result, access_map.get_mode_requirements(bid, m, tsk.get_dimensions(), full_range, tsk.get_global_size()));
		}
		return result;
	}

	std::vector<const range_mapper_base*> get_participating_range_mappers(
	    const command_group_task* const cgtsk, const buffer_id bid, const GridRegion<3>& region, bool (*const mode_filter)(access_mode)) {
		std::vector<const range_mapper_base*> participating_rms;
		for(const auto rm : cgtsk->get_buffer_access_map().get_range_mappers(bid)) {
			if(!mode_filter(rm->get_access_mode())) continue;
			const auto& geometry = cgtsk->get_geometry();
			const auto rm_region =
			    subrange_to_grid_box(apply_range_mapper(rm, chunk<3>(geometry.global_offset, geometry.global_size, geometry.global_size), geometry.dimensions));
			if(GridRegion<3>::intersect(region, rm_region).empty()) continue;
			participating_rms.push_back(rm);
		}
		return participating_rms;
	}

	bool is_communication_free_dataflow(const forward_task::access& producer, const forward_task::access& consumer) {
		// (2) if producer and consumer are unsplittable, both will be mapped to node 0 and no data transfer will take place
		if(!producer.split.variable && !consumer.split.variable) {
			// TODO variable-split is currently interpreted as "splittable"
			return true;
		}

		// (3) if producer and consumer have same split-constraints and equal range-mappers, no data transfer will take place
		if(producer.split != consumer.split) return false;
		if(producer.range_mappers.size() != consumer.range_mappers.size()) return false;

		const auto num_rms = producer.range_mappers.size();
		assert(num_rms > 0); // otherwise how would this forward task come to be?

		size_t first_rm_mismatch = num_rms;
		for(size_t i = 0; i < num_rms; ++i) {
			if(!producer.range_mappers[i]->function_equals(*consumer.range_mappers[i])) {
				first_rm_mismatch = i;
				break;
			}
		}
		if(first_rm_mismatch == num_rms) return true;      // no mismatch
		if(first_rm_mismatch == num_rms - 1) return false; // single mismatch - order independent

		// mismatches might be due to ordering: linear-search a corresponding consumer for every producer
		std::vector<const range_mapper_base*> remaining_consumer_rms(consumer.range_mappers.begin() + first_rm_mismatch, consumer.range_mappers.end());
		for(auto prm_it = producer.range_mappers.begin() + first_rm_mismatch; prm_it != producer.range_mappers.end(); ++prm_it) {
			const auto crm_it = std::find_if(remaining_consumer_rms.begin(), remaining_consumer_rms.end(),
			    [prm = *prm_it](const range_mapper_base* const crm) { return crm->function_equals(*prm); });
			if(crm_it != remaining_consumer_rms.end()) {
				remaining_consumer_rms.erase(crm_it);
			} else {
				return false;
			}
		}
		assert(remaining_consumer_rms.empty());
		return true;
	}

	bool is_potential_collective(const forward_task::access& producer, const forward_task::access& consumer) {
		if(!"NOCOMMIT too restrictive for alltoalls - we have to exclude communication-free RM pairs first. But is it even worth removing those forwards?") {
			// (1) if consumers are neither constant nor non-overlapping, distributed_graph_generator will not be able to detect a pattern

			bool all_constant = true, all_non_overlapping = true;
			for(const auto rm : consumer.range_mappers) {
				const auto props = rm->get_properties(consumer.split.get_geometry_map());
				all_constant &= props.is_constant;
				all_non_overlapping &= props.is_non_overlapping;
			}
			if(!all_constant && !all_non_overlapping) return false;
		}

		return !is_communication_free_dataflow(producer, consumer);
	}

	void task_manager::generate_forward_tasks(const command_group_task* consumer) {
		// Allow disabling collectives at runtime using env var
		// TODO NOCOMMIT (at least the env var handling)
		if(nullptr != getenv("CELERITY_NO_COLLECTIVES")) return;

		struct dataflow {
			command_group_task* producer;
			buffer_id bid;
			GridRegion<3> region;
		};
		std::vector<dataflow> dataflows;

		for(const auto bid : consumer->get_buffer_access_map().get_accessed_buffers()) {
			const auto consumer_reads = get_requirements(*consumer, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});

			std::unordered_map<command_group_task*, GridRegion<3>> contributing_writes;
			for(const auto& [box, producer_tid] : m_buffers_last_writers.at(bid).get_region_values(consumer_reads)) {
				if(producer_tid == std::nullopt || box.empty()) continue;
				const auto producer = m_task_buffer.get_task(*producer_tid);
				if(const auto cg_producer = dynamic_cast<command_group_task*>(producer)) {
					auto& writes = contributing_writes[cg_producer]; // allow default-insert
					writes = GridRegion<3>::merge(writes, box);
				}
			}

			for(const auto& [producer, contributed_region] : contributing_writes) {
				GridRegion<3> unconsumed_writes = contributed_region;
				for(const auto potential_earlier_consumer : producer->get_dependent_nodes()) {
					if(const auto earlier_cg = dynamic_cast<command_group_task*>(potential_earlier_consumer)) {
						const auto earlier_read =
						    get_requirements(*earlier_cg, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});

						if(const auto earlier_region = GridRegion<3>::intersect(unconsumed_writes, earlier_read); !earlier_region.empty()) {
							forward_task::access earlier_producer_acc{producer->get_split_constraints(),
							    get_participating_range_mappers(producer, bid, earlier_region, access::mode_traits::is_producer)};
							forward_task::access earlier_consumer_acc{earlier_cg->get_split_constraints(),
							    get_participating_range_mappers(earlier_cg, bid, earlier_region, access::mode_traits::is_consumer)};
							if(!is_communication_free_dataflow(earlier_producer_acc, earlier_consumer_acc)) {
								unconsumed_writes = GridRegion<3>::difference(unconsumed_writes, earlier_region);
							}
						}
					} else if(const auto earlier_forward = dynamic_cast<forward_task*>(potential_earlier_consumer)) {
						if(earlier_forward->get_bid() == bid) {
							unconsumed_writes = GridRegion<3>::difference(unconsumed_writes, earlier_forward->get_region());
						}
					} else /* epoch or horizon */ {
						// if the dependent were an epoch we would not have been able to reach it via the last_writers map
						assert(potential_earlier_consumer->get_type() == task_type::horizon);
						// this is fine - horizon *successors* never obscure the last-writers relationship
					}
				}
				if(unconsumed_writes.empty()) continue;

				// TODO filter trivial / impossible forwards
				dataflows.push_back(dataflow{producer, bid, unconsumed_writes});
			}
		}

		for(const auto& flow : dataflows) {
			auto fwd_reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
			forward_task::access producer_acc{flow.producer->get_split_constraints(),
			    get_participating_range_mappers(flow.producer, flow.bid, flow.region, access::mode_traits::is_producer)};
			forward_task::access consumer_acc{
			    consumer->get_split_constraints(), get_participating_range_mappers(consumer, flow.bid, flow.region, access::mode_traits::is_consumer)};
			if(!is_potential_collective(producer_acc, consumer_acc)) continue;

			auto& fwd = static_cast<forward_task&>(register_task_internal(std::move(fwd_reserve),
			    std::make_unique<forward_task>(fwd_reserve.get_tid(), flow.bid, flow.region, std::move(producer_acc), std::move(consumer_acc))));

			add_dependency(fwd, *flow.producer, dependency_kind::true_dep, dependency_origin::dataflow);
			m_buffers_last_writers.at(flow.bid).update_region(flow.region, fwd.get_id());
			invoke_callbacks(&fwd);
			// never automatically insert horizons here as they could break the dependency chain
		}
	}

	void task_manager::compute_command_group_dependencies(command_group_task& tsk) {
		using namespace cl::sycl::access;

		const auto& access_map = tsk.get_buffer_access_map();

		auto all_accessed_buffers = access_map.get_accessed_buffers();
		for(const auto& reduction : tsk.get_reductions()) {
			all_accessed_buffers.emplace(reduction.bid);
		}

		for(const auto bid : all_accessed_buffers) {
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

				for(auto& [box, last_writer] : last_writers) {
					// A null value indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(last_writer != std::nullopt) {
						add_dependency(tsk, *m_task_buffer.get_task(*last_writer), dependency_kind::true_dep, dependency_origin::dataflow);
					}
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

					for(const auto dependent : last_writer->get_dependent_nodes()) {
						if(dependent->get_id() == tsk.get_id()) {
							// This can happen
							// - if a task writes to two or more all_accessed_buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						if(const auto cg_dependent = dynamic_cast<command_group_task*>(dependent)) {
							const auto dependent_read_requirements =
							    get_requirements(*cg_dependent, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
							// Only add an anti-dependency if we are really writing over the region read by this task
							if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
								add_dependency(tsk, *cg_dependent, dependency_kind::anti_dep, dependency_origin::dataflow);
								has_anti_dependents = true;
							}
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
	}

	void task_manager::compute_dependencies(task& tsk) {
		if(const auto cgtsk = dynamic_cast<command_group_task*>(&tsk)) { compute_command_group_dependencies(*cgtsk); }

		// Tasks without any other true-dependency must depend on the last epoch to ensure they cannot be re-ordered before the epoch
		if(const auto deps = tsk.get_dependencies();
		    std::none_of(deps.begin(), deps.end(), [](const task::dependency d) { return d.kind == dependency_kind::true_dep; })) {
			add_dependency(tsk, *m_task_buffer.get_task(m_epoch_for_new_tasks), dependency_kind::true_dep, dependency_origin::last_epoch);
		}
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

		task& new_horizon = reduce_execution_front(std::move(reserve), std::make_unique<horizon_task>(*m_current_horizon));
		if(previous_horizon) { set_epoch_for_new_tasks(*previous_horizon); }

		invoke_callbacks(&new_horizon);
		return tid;
	}

	task_id task_manager::generate_epoch_task(epoch_action action) {
		auto reserve = m_task_buffer.reserve_task_entry(await_free_task_slot_callback());
		const auto tid = reserve.get_tid();

		task& new_epoch = reduce_execution_front(std::move(reserve), std::make_unique<epoch_task>(tid, action));
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

} // namespace detail
} // namespace celerity
