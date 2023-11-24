#include "instruction_executor.h"
#include "buffer_storage.h" // for memcpy_strided_host
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "mpi_communicator.h" // TODO
#include "print_utils.h"
#include "receive_arbiter.h"
#include "tracy.h"
#include "types.h"

#include <queue>

#include <matchbox.hh>

namespace celerity::detail {

// TODO --v move into PIMPL?

struct instruction_executor::pending_instruction_info {
	size_t n_unmet_dependencies;
};

struct instruction_executor::active_instruction_info {
	async_event operation;
	CELERITY_DETAIL_TRACY_DECLARE_ASYNC_LANE(tracy_lane)
};

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, std::unique_ptr<communicator> comm, delegate* dlg)
    : m_delegate(dlg), m_communicator(std::move(comm)), m_backend_queue(std::move(backend_queue)), m_recv_arbiter(*m_communicator),
      m_thread(&instruction_executor::loop, this), m_alloc_pool(4) {
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

instruction_executor::~instruction_executor() { wait(); }

void instruction_executor::wait() {
	if(m_thread.joinable()) { m_thread.join(); }
}

void instruction_executor::submit_instruction(const instruction& instr) { m_submission_queue.push_back(&instr); }

void instruction_executor::submit_pilot(const outbound_pilot& pilot) { m_submission_queue.push_back(pilot); }

void instruction_executor::announce_buffer_user_pointer(const buffer_id bid, const void* const ptr) {
	m_submission_queue.push_back(buffer_user_pointer_announcement{bid, ptr});
}

void instruction_executor::announce_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push_back(host_object_instance_announcement{hoid, std::move(instance)});
}

void instruction_executor::announce_reduction(const reduction_id rid, std::unique_ptr<runtime_reduction> reduction) {
	assert(reduction != nullptr);
	m_submission_queue.push_back(reduction_announcement{rid, std::move(reduction)});
}

void instruction_executor::thread_main() {
	CELERITY_DETAIL_TRACY_SET_CURRENT_THREAD_NAME("cy-executor");
	try {
		loop();
	} catch(const std::exception& e) {
		CELERITY_CRITICAL("[executor] {}", e.what());
		std::abort();
	}
}

// Heuristic: prioritize instructions with small launch overhead and long async completion times
int instruction_priority(const instruction* instr) {
	// - execute horizons first to unblock task generation, otherwise they are postponed indefinitely unless new alloc_instructions explicitly depend on them
	// - execute free_instructions last because nothing ever depends on them (TODO except horizons!)
	// TODO consider increasing instruction priority with age (where age = the instruction's epoch?)
	return matchbox::match(
	    *instr,                                             //
	    [](const horizon_instruction&) { return 5; },       //
	    [](const epoch_instruction&) { return 5; },         //
	    [](const device_kernel_instruction&) { return 4; }, //
	    [](const host_task_instruction&) { return 4; },     //
	    [](const fence_instruction&) { return 3; },         //
	    [](const send_instruction&) { return 2; },          //
	    [](const receive_instruction&) { return 2; },       //
	    [](const split_receive_instruction&) { return 2; }, //
	    [](const await_receive_instruction&) { return 2; }, //
	    [](const copy_instruction&) { return 2; },          //
	    [](const alloc_instruction&) { return 1; },         //
	    [](const auto&) { return 0; },                      //
	    [](const free_instruction&) { return -1; }          //
	);
}

struct instruction_priority_less {
	bool operator()(const instruction* lhs, const instruction* rhs) const { return instruction_priority(lhs) < instruction_priority(rhs); }
};

struct alloc_result {
	memory_id mid;
	allocation_id aid;
	void* ptr;
};

// TODO dupe of host_queue::future_event
template <typename Result = void>
class future_event final : public async_event_base {
  public:
	future_event(std::future<Result> future) : m_future(std::move(future)) {}

	bool is_complete() const override { return m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

	std::any take_result() override {
		if constexpr(!std::is_void_v<Result>) { return m_future.get(); }
		return std::any();
	}

  private:
	std::future<Result> m_future;
};


void instruction_executor::loop() {
	m_backend_queue->init();

	closure_hydrator::make_available();

	m_allocations.emplace(null_allocation_id, nullptr);
	m_collective_groups.emplace(root_collective_group_id, m_communicator->get_collective_root());
	m_host_queue.require_collective_group(root_collective_group_id);

	std::vector<submission> loop_submission_queue;
	std::unordered_map<const instruction*, pending_instruction_info> pending_instructions;
	std::priority_queue<const instruction*, std::vector<const instruction*>, instruction_priority_less> ready_instructions;
	std::unordered_map<const instruction*, active_instruction_info> active_instructions;
	std::unordered_set<const instruction*> incomplete_instructions;
	std::optional<std::chrono::steady_clock::time_point> last_progress_timestamp;
	bool progress_warning_emitted = false;
	while(m_expecting_more_submissions || !incomplete_instructions.empty()) {
		m_recv_arbiter.poll_communicator();

		bool made_progress = false;

		for(auto active_it = active_instructions.begin(); active_it != active_instructions.end();) {
			auto& [active_instr, active_info] = *active_it;
			if(active_info.operation.is_complete()) {
				CELERITY_DETAIL_TRACY_ASYNC_ZONE_END(active_info.tracy_lane);
				CELERITY_DEBUG("[executor] completed I{}", active_instr->get_id());

				const auto result = active_info.operation.take_result();
				if(const auto alloc = std::any_cast<alloc_result>(&result)) {
					m_allocations.emplace(alloc->aid, alloc->ptr);
					CELERITY_DEBUG("[executor] M{}.A{} allocated as {}", alloc->mid, alloc->aid, alloc->ptr);
				}

				for(const auto& dep : active_instr->get_dependents()) {
					if(const auto pending_it = pending_instructions.find(dep.node); pending_it != pending_instructions.end()) {
						auto& [pending_instr, pending_info] = *pending_it;
						assert(pending_info.n_unmet_dependencies > 0);
						pending_info.n_unmet_dependencies -= 1;
						if(pending_info.n_unmet_dependencies == 0) {
							ready_instructions.push(pending_instr);
							pending_instructions.erase(pending_it);
						}
					}
				}
				made_progress = true;
				incomplete_instructions.erase(active_instr);
				active_it = active_instructions.erase(active_it);
			} else {
				++active_it;
			}
		}

		if(m_submission_queue.swap_if_nonempty(loop_submission_queue)) {
			CELERITY_DETAIL_TRACY_SCOPED_ZONE(Gray, "process submissions");
			for(auto& submission : loop_submission_queue) {
				matchbox::match(
				    submission,
				    [&](const instruction* incoming_instr) {
					    const auto n_unmet_dependencies =
					        static_cast<size_t>(std::count_if(incoming_instr->get_dependencies().begin(), incoming_instr->get_dependencies().end(),
					            [&](const instruction::dependency& dep) { return incomplete_instructions.count(dep.node) != 0; }));
					    incomplete_instructions.insert(incoming_instr);
					    if(n_unmet_dependencies > 0) {
						    pending_instructions.emplace(incoming_instr, pending_instruction_info{n_unmet_dependencies});
					    } else {
						    ready_instructions.push(incoming_instr);
					    }
				    },
				    [&](const outbound_pilot& pilot) { //
					    m_communicator->send_outbound_pilot(pilot);
				    },
				    [&](const buffer_user_pointer_announcement& ann) {
					    assert(m_buffer_user_pointers.count(ann.bid) == 0);
					    m_buffer_user_pointers.emplace(ann.bid, ann.ptr);
				    },
				    [&](host_object_instance_announcement& ann) {
					    assert(m_host_object_instances.count(ann.hoid) == 0);
					    m_host_object_instances.emplace(ann.hoid, std::move(ann.instance));
				    },
				    [&](reduction_announcement& ann) {
					    assert(m_reductions.count(ann.rid) == 0);
					    m_reductions.emplace(ann.rid, std::move(ann.reduction));
				    });
			}
			loop_submission_queue.clear();
		}

		if(!ready_instructions.empty()) {
			const auto ready_instr = ready_instructions.top();
			ready_instructions.pop();
			active_instructions.emplace(ready_instr, begin_executing(*ready_instr));
			made_progress = true;
		}

		// TODO consider rate-limiting this (e.g. with an overflow counter) if steady_clock::now() turns out to have measurable latency
		if(made_progress) {
			last_progress_timestamp = std::chrono::steady_clock::now();
			progress_warning_emitted = false;
		} else if(last_progress_timestamp.has_value()) {
			const auto assume_stuck_after = std::chrono::seconds(3);
			const auto elapsed_since_last_progress = std::chrono::steady_clock::now() - *last_progress_timestamp;
			if(elapsed_since_last_progress > assume_stuck_after && !progress_warning_emitted) {
				std::string instr_list;
				for(auto& [instr, _] : active_instructions) {
					if(!instr_list.empty()) instr_list += ", ";
					fmt::format_to(std::back_inserter(instr_list), "I{}", instr->get_id());
				}
				CELERITY_WARN("[executor] no progress for {:.3f} seconds, potentially stuck. Active instructions: {}",
				    std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_since_last_progress).count(),
				    active_instructions.empty() ? "none" : instr_list);
				progress_warning_emitted = true;
			}
		}
	}

	assert(m_allocations.size() == 1); // null allocation
	assert(m_host_object_instances.empty());
}

void instruction_executor::prepare_accessor_hydration(
    target target, const buffer_access_allocation_map& amap CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, const boundary_check_info& oob_info)) {
	std::vector<closure_hydrator::accessor_info> accessor_infos;
	accessor_infos.reserve(amap.size());
	for(size_t i = 0; i < amap.size(); ++i) {
		const auto ptr = m_allocations.at(amap[i].allocation_id);
		accessor_infos.push_back(closure_hydrator::accessor_info{ptr, amap[i].allocated_box_in_buffer,
		    amap[i].accessed_box_in_buffer CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, &oob_info.illegal_access_bounding_boxes[i])});
	}

	closure_hydrator::get_instance().arm(target, std::move(accessor_infos));
}

#if CELERITY_ACCESSOR_BOUNDARY_CHECK

instruction_executor::boundary_check_info instruction_executor::prepare_accessor_boundary_check(
    const buffer_access_allocation_map& amap, const task_id tid, const std::string& task_name, const target target) {
	boundary_check_info info;
	if(!amap.empty()) {
		info.illegal_access_bounding_boxes =
		    static_cast<oob_bounding_box*>(m_backend_queue->malloc(host_memory_id, amap.size() * sizeof(oob_bounding_box), alignof(oob_bounding_box)));
		std::uninitialized_default_construct_n(info.illegal_access_bounding_boxes, amap.size());
	}
	for(size_t i = 0; i < amap.size(); ++i) {
		info.accessors.push_back({amap[i].oob_buffer_id, amap[i].oob_buffer_name, amap[i].accessed_box_in_buffer});
	}
	info.task_id = tid;
	info.task_name = task_name;
	info.target = target;
	return info;
}

bool instruction_executor::boundary_checked_event::is_complete() const {
	if(!m_state.has_value()) return true; // we clear `state` completion to make is_complete() idempotent
	if(!m_state->launch_event.is_complete()) return false;

	const auto& info = m_state->oob_info;
	for(size_t i = 0; i < info.accessors.size(); ++i) {
		if(const auto oob_box = info.illegal_access_bounding_boxes[i].into_box(); !oob_box.empty()) {
			const auto& accessor_info = info.accessors[i];
			CELERITY_ERROR(
			    "Out-of-bounds access detected in {} T{}{}: accessor {} attempted to access buffer {} indicies between {} and outside the declared range {}.",
			    info.target == target::device ? "device kernel" : "host task", info.task_id,
			    (!info.task_name.empty() ? fmt::format(" \"{}\"", info.task_name) : ""), i,
			    get_buffer_label(accessor_info.buffer_id, accessor_info.buffer_name), oob_box, accessor_info.accessible_box);
		}
	}

	if(info.illegal_access_bounding_boxes != nullptr /* i.e. there is at least one accessor */) {
		m_state->executor->m_backend_queue->free(host_memory_id, info.illegal_access_bounding_boxes);
	}

	m_state = {}; // make is_complete() idempotent
	return true;
}

#endif // CELERITY_ACCESSOR_BOUNDARY_CHECK

instruction_executor::active_instruction_info instruction_executor::begin_executing(const instruction& instr) {
	static constexpr auto log_accesses = [](const buffer_access_allocation_map& map) {
		std::string acc_log;
		for(size_t i = 0; i < map.size(); ++i) {
			auto& aa = map[i];
			const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
			    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
			fmt::format_to(std::back_inserter(acc_log), "{} A{} {}", i == 0 ? ", accessing" : ",", aa.allocation_id, accessed_box_in_allocation);
		}
		return acc_log;
	};

	active_instruction_info active_instruction;
	active_instruction.operation = matchbox::match<async_event>(
	    instr,
	    [&](const clone_collective_group_instruction& ccginstr) {
		    const auto new_cgid = ccginstr.get_new_collective_group_id();
		    const auto origin_cgid = ccginstr.get_original_collective_group_id();

		    CELERITY_DEBUG("[executor] I{}: clone collective group CG{} -> CG{}", ccginstr.get_id(), origin_cgid, new_cgid);
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Brown, "I{} clone collective", ccginstr.get_id());

		    assert(m_collective_groups.count(new_cgid) == 0);
		    const auto new_group = m_collective_groups.at(origin_cgid)->clone();
		    m_collective_groups.emplace(new_cgid, new_group);
		    m_host_queue.require_collective_group(new_cgid);
		    return make_complete_event();
	    },
	    [&](const alloc_instruction& ainstr) {
		    CELERITY_DEBUG("[executor] I{}: alloc M{}.A{}, {}%{} bytes", ainstr.get_id(), ainstr.get_memory_id(), ainstr.get_allocation_id(),
		        ainstr.get_size_bytes(), ainstr.get_alignment_bytes());

		    void* ptr;
		    {
			    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Turquoise, "I{} alloc", ainstr.get_id());
			    ptr = m_backend_queue->malloc(ainstr.get_memory_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
		    }

		    CELERITY_DEBUG("[executor] M{}.A{} allocated as {}", ainstr.get_memory_id(), ainstr.get_allocation_id(), ptr);
		    m_allocations.emplace(ainstr.get_allocation_id(), ptr);
		    return make_complete_event();
	    },
	    [&](const free_instruction& finstr) {
		    const auto it = m_allocations.find(finstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto ptr = it->second;
		    m_allocations.erase(it);

		    CELERITY_DEBUG("[executor] I{}: free M{}.A{}", finstr.get_id(), finstr.get_memory_id(), finstr.get_allocation_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Turquoise, "I{} free", finstr.get_id());

		    m_backend_queue->free(finstr.get_memory_id(), ptr);
		    return make_complete_event();
	    },
	    [&](const init_buffer_instruction& ibinstr) {
		    CELERITY_DEBUG("[executor] I{}: init B{} as M0.A{}, {} bytes", ibinstr.get_id(), ibinstr.get_buffer_id(), ibinstr.get_host_allocation_id(),
		        ibinstr.get_size_bytes());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Lime, "I{} init buffer", ibinstr.get_id());

		    const auto user_ptr = m_buffer_user_pointers.at(ibinstr.get_buffer_id());
		    const auto host_ptr = m_allocations.at(ibinstr.get_host_allocation_id());
		    memcpy(host_ptr, user_ptr, ibinstr.get_size_bytes());
		    return make_complete_event();
	    },
	    [&](const export_instruction& einstr) {
		    CELERITY_DEBUG("[executor] I{}: export M0.A{} ({})+{}, {}x{} bytes", einstr.get_id(), einstr.get_host_allocation_id(),
		        einstr.get_allocation_range(), einstr.get_offset_in_allocation(), einstr.get_copy_range(), einstr.get_element_size());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Lime, "I{} export", einstr.get_id());

		    const auto dest_ptr = einstr.get_out_pointer(); // TODO very naughty
		    const auto source_base_ptr = m_allocations.at(einstr.get_host_allocation_id());

		    // TODO copy-pasted,but export_instruction will be removed anyway
		    const auto dispatch_copy = [&](const auto dims) {
			    memcpy_strided_host(source_base_ptr, dest_ptr, einstr.get_element_size(), range_cast<dims.value>(einstr.get_allocation_range()),
			        id_cast<dims.value>(einstr.get_offset_in_allocation()), range_cast<dims.value>(einstr.get_copy_range()), id<dims.value>(),
			        range_cast<dims.value>(einstr.get_copy_range()));
		    };
		    switch(einstr.get_dimensions()) {
		    case 0: dispatch_copy(std::integral_constant<int, 0>()); break;
		    case 1: dispatch_copy(std::integral_constant<int, 1>()); break;
		    case 2: dispatch_copy(std::integral_constant<int, 2>()); break;
		    case 3: dispatch_copy(std::integral_constant<int, 3>()); break;
		    default: abort();
		    }
		    return make_complete_event();
	    },
	    [&](const copy_instruction& cinstr) {
		    CELERITY_DEBUG("[executor] I{}: copy M{}.A{}+{} -> M{}.A{}+{}, {}x{} bytes", cinstr.get_id(), cinstr.get_source_memory_id(),
		        cinstr.get_source_allocation_id(), cinstr.get_offset_in_source(), cinstr.get_dest_memory_id(), cinstr.get_dest_allocation_id(),
		        cinstr.get_offset_in_dest(), cinstr.get_copy_range(), cinstr.get_element_size());

		    const auto source_base_ptr = m_allocations.at(cinstr.get_source_allocation_id());
		    const auto dest_base_ptr = m_allocations.at(cinstr.get_dest_allocation_id());
		    if(cinstr.get_source_memory_id() == host_memory_id && cinstr.get_dest_memory_id() == host_memory_id) {
			    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Lime, "I{} copy", cinstr.get_id());
			    const auto dispatch_copy = [&](const auto dims) {
				    memcpy_strided_host(source_base_ptr, dest_base_ptr, cinstr.get_element_size(), range_cast<dims.value>(cinstr.get_source_range()),
				        id_cast<dims.value>(cinstr.get_offset_in_source()), range_cast<dims.value>(cinstr.get_dest_range()),
				        id_cast<dims.value>(cinstr.get_offset_in_dest()), range_cast<dims.value>(cinstr.get_copy_range()));
			    };
			    switch(cinstr.get_dimensions()) {
			    case 0: dispatch_copy(std::integral_constant<int, 0>()); break;
			    case 1: dispatch_copy(std::integral_constant<int, 1>()); break;
			    case 2: dispatch_copy(std::integral_constant<int, 2>()); break;
			    case 3: dispatch_copy(std::integral_constant<int, 3>()); break;
			    default: abort();
			    }
			    return make_complete_event();
		    } else {
			    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Lime, "I{} copy", cinstr.get_id());
			    return m_backend_queue->memcpy_strided_device(cinstr.get_dimensions(), cinstr.get_source_memory_id(), cinstr.get_dest_memory_id(),
			        source_base_ptr, dest_base_ptr, cinstr.get_element_size(), cinstr.get_source_range(), cinstr.get_offset_in_source(),
			        cinstr.get_dest_range(), cinstr.get_offset_in_dest(), cinstr.get_copy_range());
		    }
	    },
	    [&](const device_kernel_instruction& dkinstr) {
		    CELERITY_DEBUG("[executor] I{}: launch device kernel on D{}, {}{}", dkinstr.get_id(), dkinstr.get_device_id(), dkinstr.get_execution_range(),
		        log_accesses(dkinstr.get_access_allocations()));
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Orange, "I{} device kernel", dkinstr.get_id());

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    auto oob_info =
		        prepare_accessor_boundary_check(dkinstr.get_access_allocations(), dkinstr.get_oob_task_id(), dkinstr.get_oob_task_name(), target::device);
#endif
		    prepare_accessor_hydration(target::device, dkinstr.get_access_allocations() CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, oob_info));

		    std::vector<void*> reduction_ptrs;
		    reduction_ptrs.reserve(dkinstr.get_reduction_allocations().size());
		    for(const auto& ra : dkinstr.get_reduction_allocations()) {
			    reduction_ptrs.push_back(m_allocations.at(ra.allocation_id));
		    }

		    auto evt = m_backend_queue->launch_kernel(dkinstr.get_device_id(), dkinstr.get_launcher(), dkinstr.get_execution_range(), reduction_ptrs);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return make_async_event<boundary_checked_event>(this, std::move(evt), std::move(oob_info));
#else
		    return evt;
#endif
	    },
	    [&](const host_task_instruction& htinstr) {
		    CELERITY_DEBUG(
		        "[executor] I{}: launch host task, {}{}", htinstr.get_id(), htinstr.get_execution_range(), log_accesses(htinstr.get_access_allocations()));
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Orange, "I{} host task", htinstr.get_id());

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    auto oob_info =
		        prepare_accessor_boundary_check(htinstr.get_access_allocations(), htinstr.get_oob_task_id(), htinstr.get_oob_task_name(), target::host_task);
#endif
		    prepare_accessor_hydration(target::host_task, htinstr.get_access_allocations() CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, oob_info));

		    // TODO executor must not have any direct dependency on MPI!
		    MPI_Comm mpi_comm = MPI_COMM_NULL;
		    if(const auto cgid = htinstr.get_collective_group_id(); cgid != non_collective_group_id) {
			    const auto cg = m_collective_groups.at(htinstr.get_collective_group_id());
			    mpi_comm = dynamic_cast<mpi_communicator::collective_group&>(*cg).get_mpi_comm();
		    }

		    const auto& launch = htinstr.get_launcher();
		    auto evt = launch(m_host_queue, htinstr.get_execution_range(), mpi_comm);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return make_async_event<boundary_checked_event>(this, std::move(evt), std::move(oob_info));
#else
		    return evt;
#endif
	    },
	    [&](const send_instruction& sinstr) {
		    CELERITY_DEBUG("[executor] I{}: send M{}.A{}+{}, {}x{} bytes to N{} (tag {})", sinstr.get_id(), host_memory_id /* TODO RDMA */,
		        sinstr.get_source_allocation_id(), sinstr.get_offset_in_source_allocation(), sinstr.get_send_range(), sinstr.get_element_size(),
		        sinstr.get_dest_node_id(), sinstr.get_tag());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Violet, "I{} send", sinstr.get_id());

		    const auto allocation_base = m_allocations.at(sinstr.get_source_allocation_id());
		    const communicator::stride stride{
		        sinstr.get_source_allocation_range(),
		        subrange<3>{sinstr.get_offset_in_source_allocation(), sinstr.get_send_range()},
		        sinstr.get_element_size(),
		    };
		    return m_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_tag(), allocation_base, stride);
	    },
	    [&](const receive_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: receive {} {} into M{}.A{} ({}), x{} bytes", rinstr.get_id(), rinstr.get_transfer_id(),
		        rinstr.get_requested_region(), rinstr.get_dest_memory_id(), rinstr.get_dest_allocation_id(), rinstr.get_allocated_box(),
		        rinstr.get_element_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Violet, "I{} receive", rinstr.get_id());

		    const auto allocation = m_allocations.at(rinstr.get_dest_allocation_id());
		    return m_recv_arbiter.receive(
		        rinstr.get_transfer_id(), rinstr.get_requested_region(), allocation, rinstr.get_allocated_box(), rinstr.get_element_size());
	    },
	    [&](const split_receive_instruction& srinstr) {
		    CELERITY_DEBUG("[executor] I{}: split receive {} {} into M{}.A{} ({}), x{} bytes", srinstr.get_id(), srinstr.get_transfer_id(),
		        srinstr.get_requested_region(), srinstr.get_dest_memory_id(), srinstr.get_dest_allocation_id(), srinstr.get_allocated_box(),
		        srinstr.get_element_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Violet, "I{} split receive", srinstr.get_id());

		    const auto allocation = m_allocations.at(srinstr.get_dest_allocation_id());
		    m_recv_arbiter.begin_split_receive(
		        srinstr.get_transfer_id(), srinstr.get_requested_region(), allocation, srinstr.get_allocated_box(), srinstr.get_element_size());
		    return make_complete_event();
	    },
	    [&](const await_receive_instruction& arinstr) {
		    CELERITY_DEBUG("[executor] I{}: await receive {} {}", arinstr.get_id(), arinstr.get_transfer_id(), arinstr.get_received_region());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Violet, "I{} await receive", arinstr.get_id());

		    return m_recv_arbiter.await_split_receive_subregion(arinstr.get_transfer_id(), arinstr.get_received_region());
	    },
	    [&](const gather_receive_instruction& grinstr) {
		    CELERITY_DEBUG("[executor] I{}: gather receive {} into M{}.A{}, {} bytes per node", grinstr.get_id(), grinstr.get_transfer_id(),
		        grinstr.get_dest_memory_id(), grinstr.get_dest_allocation_id(), grinstr.get_node_chunk_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", Violet, "I{} gather receive", grinstr.get_id());

		    const auto allocation = m_allocations.at(grinstr.get_dest_allocation_id());
		    return m_recv_arbiter.gather_receive(grinstr.get_transfer_id(), allocation, grinstr.get_node_chunk_size());
	    },
	    [&](const fill_identity_instruction& fiinstr) {
		    CELERITY_DEBUG("[executor] I{}: fill identity M{}.A{} x{} for R{}", fiinstr.get_id(), fiinstr.get_memory_id(), fiinstr.get_allocation_id(),
		        fiinstr.get_num_values(), fiinstr.get_reduction_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Blue, "I{} fill identity", fiinstr.get_id());

		    const auto allocation = m_allocations.at(fiinstr.get_allocation_id());
		    const auto& reduction = *m_reductions.at(fiinstr.get_reduction_id());
		    reduction.fill_identity(allocation, fiinstr.get_num_values());
		    return make_complete_event();
	    },
	    [&](const reduce_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: reduce M{}.A{} x{} into M{}.A{} as R{}", rinstr.get_id(), rinstr.get_memory_id(), rinstr.get_source_allocation_id(),
		        rinstr.get_num_source_values(), rinstr.get_memory_id(), rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Blue, "I{} reduce", rinstr.get_id());

		    const auto gather_allocation = m_allocations.at(rinstr.get_source_allocation_id());
		    const auto dest_allocation = m_allocations.at(rinstr.get_dest_allocation_id());
		    const bool include_dest = false; // TODO
		    const auto& reduction = *m_reductions.at(rinstr.get_reduction_id());
		    reduction.reduce(dest_allocation, gather_allocation, rinstr.get_num_source_values(), include_dest);
		    // TODO GC runtime_reduction at some point
		    return make_complete_event();
	    },
	    [&](const fence_instruction& finstr) {
		    CELERITY_DEBUG("[executor] I{}: fence", finstr.get_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Blue, "fence");

		    finstr.get_promise()->fulfill();
		    return make_complete_event();
	    },
	    [&](const destroy_host_object_instruction& dhoinstr) {
		    assert(m_host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
		    CELERITY_DEBUG("[executor] I{}: destroy H{}", dhoinstr.get_id(), dhoinstr.get_host_object_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Gray, "destroy host object");

		    m_host_object_instances.erase(dhoinstr.get_host_object_id());
		    return make_complete_event();
	    },
	    [&](const horizon_instruction& hinstr) {
		    CELERITY_DEBUG("[executor] I{}: horizon", hinstr.get_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Gray, "horizon");

		    if(m_delegate != nullptr) { m_delegate->horizon_reached(hinstr.get_horizon_task_id()); }
		    return make_complete_event();
	    },
	    [&](const epoch_instruction& einstr) {
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE(Gray, "epoch");

		    switch(einstr.get_epoch_action()) {
		    case epoch_action::none: CELERITY_DEBUG("[executor] I{}: epoch", einstr.get_id()); break;
		    case epoch_action::barrier:
			    CELERITY_DEBUG("[executor] I{}: epoch (barrier)", einstr.get_id());
			    m_communicator->get_collective_root()->barrier();
			    break;
		    case epoch_action::shutdown:
			    CELERITY_DEBUG("[executor] I{}: epoch (shutdown)", einstr.get_id());
			    m_expecting_more_submissions = false;
			    break;
		    }
		    if(m_delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
			    m_delegate->epoch_reached(einstr.get_epoch_task_id());
		    }
		    return make_complete_event();
	    });
	return active_instruction;
}

} // namespace celerity::detail
