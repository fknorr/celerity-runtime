#include "live_executor.h"
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "named_threads.h"
#include "out_of_order_engine.h"
#include "receive_arbiter.h"
#include "system_info.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"

#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include <matchbox.hh>

namespace celerity::detail::live_executor_detail {

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
struct boundary_check_info {
	struct accessor_info {
		detail::buffer_id buffer_id = 0;
		std::string buffer_name;
		box<3> accessible_box;
	};

	oob_bounding_box* illegal_access_bounding_boxes = nullptr;
	std::vector<accessor_info> accessors;

	detail::task_type task_type = detail::task_type::epoch;
	detail::task_id task_id = 0;
	std::string task_name;
};
#endif

#if CELERITY_ENABLE_TRACY
struct tracy_async_zone {
	size_t submission_idx = 0;
	const instruction* instr = nullptr;
	std::optional<std::chrono::steady_clock::time_point> approx_begin;
};

struct tracy_async_lane {
	out_of_order_engine::target target = out_of_order_engine::target::immediate;
	std::optional<device_id> device;
	size_t local_lane_id = 0;
	std::string fiber_name;
	size_t next_submission_idx = 0;
	std::optional<TracyCZoneCtx> active_zone;
	std::queue<tracy_async_zone> queued_zones;
};

struct tracy_async_cursor {
	size_t global_lane_id = 0;
	size_t lane_submission_idx = 0;
};
#endif

struct async_instruction_state {
	const instruction* instr = nullptr;
	async_event event;
	CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(std::unique_ptr<boundary_check_info> oob_info;) // unique_ptr: oob_info is optional and rather large
	CELERITY_DETAIL_IF_TRACY(std::optional<tracy_async_cursor> tracy_cursor;)
};

struct executor_impl {
	live_executor::delegate* delegate;
	double_buffered_queue<submission>* submission_queue;

	communicator* root_communicator;
	bool expecting_more_submissions = true;
	std::unique_ptr<detail::backend> backend;
	std::unordered_map<allocation_id, void*> allocations{{null_allocation_id, nullptr}};
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> host_object_instances;
	std::unordered_map<collective_group_id, std::unique_ptr<communicator>> cloned_communicators;
	std::unordered_map<reduction_id, std::unique_ptr<reducer>> reducers;
	receive_arbiter recv_arbiter{*root_communicator};

	std::vector<async_instruction_state> in_flight_async_instructions;
	out_of_order_engine engine;

	std::optional<std::chrono::steady_clock::time_point> last_progress_timestamp;
	bool made_progress = false;
	bool progress_warning_emitted = false;

#if CELERITY_ENABLE_TRACY
	std::vector<std::unique_ptr<tracy_async_lane>>
	    tracy_async_lanes; // TODO instead of making this a unique_ptr we must leak the fiber name string (see tracy manual)
	std::vector<uint8_t /* bool occupied */> tracy_immediate_async_lanes;
	tracy_async_cursor tracy_get_async_lane_cursor(
	    out_of_order_engine::target target, const std::optional<device_id>& device, const std::optional<size_t>& local_lane_id);
#endif

	executor_impl(const system_info& system, std::unique_ptr<detail::backend> backend, communicator* root_comm,
	    double_buffered_queue<submission>& submission_queue, live_executor::delegate* dlg);

	void run();
	void poll_in_flight_async_instructions();
	void poll_submission_queue();
	void try_issue_one_instruction();
	void retire_async_instruction(async_instruction_state& async);
	void check_progress();

	void issue(const clone_collective_group_instruction& ccginstr);
	void issue(const split_receive_instruction& srinstr);
	void issue(const fill_identity_instruction& fiinstr);
	void issue(const reduce_instruction& rinstr);
	void issue(const fence_instruction& finstr);
	void issue(const destroy_host_object_instruction& dhoinstr);
	void issue(const horizon_instruction& hinstr);
	void issue(const epoch_instruction& einstr);

	template <typename Instr>
	auto dispatch_issue(const Instr& instr, const out_of_order_engine::assignment& assignment) //
	    -> decltype(issue(instr));

	void issue_async(const alloc_instruction& ainstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const free_instruction& finstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const copy_instruction& cinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const device_kernel_instruction& dkinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const host_task_instruction& htinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const send_instruction& sinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const receive_instruction& rinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const await_receive_instruction& arinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const gather_receive_instruction& grinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);

	template <typename Instr>
	auto dispatch_issue(const Instr& instr, const out_of_order_engine::assignment& assignment)
	    -> decltype(issue_async(instr, assignment, std::declval<async_instruction_state&>()));

	void collect(const instruction_garbage& garbage);
	void prepare_accessor_hydration(instruction_id iid, target target,
	    const buffer_access_allocation_map& amap CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(
	        , task_type tt, task_id tid, const std::string& task_name, std::unique_ptr<boundary_check_info>& out_oob_info));
};

#if CELERITY_ENABLE_TRACY
tracy_async_cursor executor_impl::tracy_get_async_lane_cursor(
    const out_of_order_engine::target target, const std::optional<device_id>& device, const std::optional<size_t>& local_lane_id) //
{
	auto real_local_lane_id = local_lane_id.value_or(0);
	if(target == out_of_order_engine::target::immediate) {
		assert(!local_lane_id.has_value());
		real_local_lane_id = static_cast<size_t>(
		    std::find(tracy_immediate_async_lanes.begin(), tracy_immediate_async_lanes.end(), 0 /* not occupied */) - tracy_immediate_async_lanes.begin());
		if(real_local_lane_id < tracy_immediate_async_lanes.size()) {
			tracy_immediate_async_lanes.at(real_local_lane_id) = 1 /* occupied */;
		} else {
			tracy_immediate_async_lanes.push_back(0 /* occupied */);
		}
	}
	auto it = std::find_if(tracy_async_lanes.begin(), tracy_async_lanes.end(), [&](const std::unique_ptr<tracy_async_lane>& lane) {
		return lane->target == target && lane->device == device && lane->local_lane_id == real_local_lane_id;
	});
	if(it == tracy_async_lanes.end()) {
		it = tracy_async_lanes.emplace(tracy_async_lanes.end(), std::make_unique<tracy_async_lane>());
		auto& lane = **it;
		lane.target = target, lane.device = device, lane.local_lane_id = real_local_lane_id;
		switch(target) {
		case out_of_order_engine::target::immediate: lane.fiber_name = fmt::format("Send/Receive Lane #{}", real_local_lane_id); break;
		case out_of_order_engine::target::alloc_queue: lane.fiber_name = "Allocation Queue"; break;
		case out_of_order_engine::target::host_queue: lane.fiber_name = fmt::format("Host Queue #{}", lane.local_lane_id); break;
		case out_of_order_engine::target::device_queue: lane.fiber_name = fmt::format("Device {} Queue #{}", lane.device.value(), lane.local_lane_id); break;
		default: utils::unreachable();
		}
	}
	auto& lane = **it;
	const auto global_lane_id = static_cast<size_t>(it - tracy_async_lanes.begin());
	return tracy_async_cursor{global_lane_id, lane.next_submission_idx++};
}

TracyCZoneCtx tracy_begin_zone(const instruction& instr, const bool eager) {
	TracyCZoneCtx ctx;
	std::string_view tag;
	std::string text;

#define CELERITY_DETAIL_BEGIN_CTX(TAG, COLOR)                                                                                                                  \
	TracyCZoneNC(scoped_ctx, "executor::" TAG, tracy::Color::COLOR, true);                                                                                     \
	ctx = scoped_ctx, tag = TAG

	matchbox::match(
	    instr,
	    [&](const clone_collective_group_instruction& ccginstr) {
		    CELERITY_DETAIL_BEGIN_CTX("clone_collective_group", Brown);
		    text = fmt::format("CG{} -> CG{}", ccginstr.get_original_collective_group_id(), ccginstr.get_new_collective_group_id());
	    },
	    [&](const alloc_instruction& ainstr) {
		    CELERITY_DETAIL_BEGIN_CTX("alloc", Turquoise);
		    text = fmt::format("alloc {}, {} % {} bytes", ainstr.get_allocation_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
	    },
	    [&](const free_instruction& finstr) {
		    CELERITY_DETAIL_BEGIN_CTX("free", Turquoise);
		    text = fmt::format("free {}", finstr.get_allocation_id());
	    },
	    [&](const copy_instruction& cinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("copy", Lime);
		    text = fmt::format("copy {} -> {}, {} x{} bytes\n{} bytes total", cinstr.get_source_allocation(), cinstr.get_dest_allocation(),
		        cinstr.get_copy_region(), cinstr.get_element_size(), cinstr.get_copy_region().get_area() * cinstr.get_element_size());
	    },
	    [&](const device_kernel_instruction& dkinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("device_kernel", Orange);
		    text = fmt::format("on D{}", dkinstr.get_device_id());
	    },
	    [&](const host_task_instruction& htinstr) { //
		    CELERITY_DETAIL_BEGIN_CTX("host_task", Orange);
	    },
	    [&](const send_instruction& sinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("send", Violet);
		    text = fmt::format("send {}+{}, {}x{} bytes to N{}\n{} bytes total", sinstr.get_source_allocation_id(), sinstr.get_offset_in_source_allocation(),
		        sinstr.get_send_range(), sinstr.get_element_size(), sinstr.get_dest_node_id(), sinstr.get_send_range() * sinstr.get_element_size());
	    },
	    [&](const receive_instruction& rinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("receive", DarkViolet);
		    text = fmt::format("receive {} {} into {} ({}), x{} bytes", rinstr.get_transfer_id(), rinstr.get_requested_region(),
		        rinstr.get_dest_allocation_id(), rinstr.get_allocated_box(), rinstr.get_element_size());
	    },
	    [&](const split_receive_instruction& srinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("split_receive", DarkViolet);
		    text = fmt::format("split receive {} {} into {} ({}), x{} bytes", srinstr.get_transfer_id(), srinstr.get_requested_region(),
		        srinstr.get_dest_allocation_id(), srinstr.get_allocated_box(), srinstr.get_element_size());
	    },
	    [&](const await_receive_instruction& arinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("await_receive", DarkViolet);
		    text = fmt::format("await receive {} {}", arinstr.get_transfer_id(), arinstr.get_received_region());
	    },
	    [&](const gather_receive_instruction& grinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("gather_receive", DarkViolet);
		    text = fmt::format(
		        "gather receive {} into {}, {} bytes per node", grinstr.get_transfer_id(), grinstr.get_dest_allocation_id(), grinstr.get_node_chunk_size());
	    },
	    [&](const fill_identity_instruction& fiinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("fill_identity", Blue);
		    text = fmt::format("fill identity {} x{} for R{}", fiinstr.get_allocation_id(), fiinstr.get_num_values(), fiinstr.get_reduction_id());
	    },
	    [&](const reduce_instruction& rinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("reduce", Blue);
		    text = fmt::format("reduce {} x{} into {} as R{}", rinstr.get_source_allocation_id(), rinstr.get_num_source_values(),
		        rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());
	    },
	    [&](const fence_instruction& finstr) { //
		    CELERITY_DETAIL_BEGIN_CTX("fence", Blue);
	    },
	    [&](const destroy_host_object_instruction& dhoinstr) {
		    CELERITY_DETAIL_BEGIN_CTX("destroy_host_object", Gray);
		    text = fmt::format("destroy H{}", dhoinstr.get_host_object_id());
	    },
	    [&](const horizon_instruction& hinstr) { //
		    CELERITY_DETAIL_BEGIN_CTX("horizon", Gray);
	    },
	    [&](const epoch_instruction& einstr) {
		    CELERITY_DETAIL_BEGIN_CTX("epoch", Gray);
		    switch(einstr.get_epoch_action()) {
		    case epoch_action::barrier: text = "barrier"; break;
		    case epoch_action::shutdown: text = "shutdown"; break;
		    default:;
		    }
	    });

#undef CELERITY_DETAIL_BEGIN_CTX

	const auto label = fmt::format("{}I{} {}", eager ? "+" : "", instr.get_id(), tag);
	TracyCZoneName(ctx, label.data(), label.size());

	for(size_t i = 0; i < instr.get_dependencies().size(); ++i) {
		text += i == 0 ? "\ndepends: " : ", ";
		fmt::format_to(std::back_inserter(text), "I{}", instr.get_dependencies()[i]);
	}

	fmt::format_to(std::back_inserter(text), "\npriority: {}", instr.get_priority());
	TracyCZoneText(ctx, text.data(), text.size());

	return ctx;
}

void tracy_end_zone(const TracyCZoneCtx& ctx) { TracyCZoneEnd(ctx); }

void tracy_begin_async_zone(tracy_async_lane& lane, tracy_async_zone& zone, const bool eager) {
	assert(!lane.active_zone.has_value());
	zone.approx_begin = std::chrono::steady_clock::now();
	lane.active_zone = tracy_begin_zone(*zone.instr, eager);
}

void tracy_end_async_zone(tracy_async_lane& lane, const tracy_async_zone& zone) {
	assert(lane.active_zone.has_value());
	assert(zone.approx_begin.has_value());
	const auto bytes_processed = matchbox::match(
	    *zone.instr,                                                                                                    //
	    [](const alloc_instruction& ainstr) { return ainstr.get_size_bytes(); },                                        //
	    [](const copy_instruction& cinstr) { return cinstr.get_copy_region().get_area() * cinstr.get_element_size(); }, //
	    [](const send_instruction& sinstr) { return sinstr.get_send_range().size() * sinstr.get_element_size(); },      //
	    [](const auto& /* other */) { return 0; });

	if(bytes_processed > 0) {
		const auto approx_end = std::chrono::steady_clock::now();
		const auto approx_secs = std::chrono::duration_cast<std::chrono::duration<double>>(approx_end - *zone.approx_begin).count();
		const auto text = fmt::format("throughput: {:.2f} MB/s", static_cast<double>(bytes_processed) / (1024.0 * 1024.0 * approx_secs));
		TracyCZoneText(*lane.active_zone, text.data(), text.length());
	}
	TracyCZoneEnd(*lane.active_zone);
	lane.active_zone = std::nullopt;
}
#endif

executor_impl::executor_impl(const system_info& system, std::unique_ptr<detail::backend> backend, communicator* const root_comm,
    double_buffered_queue<submission>& submission_queue, live_executor::delegate* const dlg)
    : delegate(dlg), submission_queue(&submission_queue), root_communicator(root_comm), backend(std::move(backend)), recv_arbiter(*root_communicator),
      engine(system) {}

void executor_impl::run() {
	closure_hydrator::make_available();

#if CELERITY_ENABLE_TRACY
	// TODO have one of these per device as well?
	TracyPlot("active instrs", static_cast<int64_t>(0));
	TracyPlotConfig("active instrs", tracy::PlotFormatType::Number, true /* setp */, true /* fill*/, 0);
#endif

	for(;;) {
		if(engine.is_idle()) {
			if(!expecting_more_submissions) break;
			submission_queue->wait_while_empty();
			last_progress_timestamp.reset();
		}

		recv_arbiter.poll_communicator();
		poll_in_flight_async_instructions();
		poll_submission_queue();
		try_issue_one_instruction();
		check_progress();
	}

	assert(in_flight_async_instructions.empty());
	assert(std::all_of(allocations.begin(), allocations.end(),
	    [](const std::pair<allocation_id, void*>& p) { return p.first == null_allocation_id || p.first.get_memory_id() == user_memory_id; }));
	assert(host_object_instances.empty());
}

void executor_impl::poll_in_flight_async_instructions() {
	CELERITY_DETAIL_IF_TRACY(const auto active_instrs_before = in_flight_async_instructions.size());

	utils::erase_if(in_flight_async_instructions, [&](async_instruction_state& async) {
		if(!async.event.is_complete()) return false;
		retire_async_instruction(async);
		made_progress = true;
		return true;
	});

#if CELERITY_ENABLE_TRACY
	if(in_flight_async_instructions.size() != active_instrs_before) { TracyPlot("active instrs", static_cast<int64_t>(in_flight_async_instructions.size())); }
#endif
}

void executor_impl::poll_submission_queue() {
	for(auto& submission : submission_queue->pop_all()) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("executor::dequeue", Gray, "dequeue");
		matchbox::match(
		    submission,
		    [&](const instruction_pilot_batch& batch) {
			    for(const auto incoming_instr : batch.instructions) {
				    engine.submit(incoming_instr);
			    }
			    for(const auto& pilot : batch.pilots) {
				    root_communicator->send_outbound_pilot(pilot);
			    }
		    },
		    [&](const user_allocation_announcement& ann) {
			    assert(ann.aid != null_allocation_id);
			    assert(ann.aid.get_memory_id() == user_memory_id);
			    assert(allocations.count(ann.aid) == 0);
			    allocations.emplace(ann.aid, ann.ptr);
		    },
		    [&](host_object_instance_announcement& ann) {
			    assert(host_object_instances.count(ann.hoid) == 0);
			    host_object_instances.emplace(ann.hoid, std::move(ann.instance));
		    },
		    [&](reducer_announcement& ann) {
			    assert(reducers.count(ann.rid) == 0);
			    reducers.emplace(ann.rid, std::move(ann.reduction));
		    });
	}
}

void executor_impl::retire_async_instruction(async_instruction_state& async) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("executor::retire", Brown, "retire");

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	if(async.oob_info != nullptr) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("executor::oob_check", Red, "I{} bounds check", async.instr->get_id());
		const auto& oob_info = *async.oob_info;
		for(size_t i = 0; i < oob_info.accessors.size(); ++i) {
			if(const auto oob_box = oob_info.illegal_access_bounding_boxes[i].into_box(); !oob_box.empty()) {
				const auto& accessor_info = oob_info.accessors[i];
				CELERITY_ERROR("Out-of-bounds access detected in {}: accessor {} attempted to access buffer {} indicies between {} and outside the "
				               "declared range {}.",
				    utils::make_task_debug_label(oob_info.task_type, oob_info.task_id, oob_info.task_name), i,
				    utils::make_buffer_debug_label(accessor_info.buffer_id, accessor_info.buffer_name), oob_box, accessor_info.accessible_box);
			}
		}
		if(oob_info.illegal_access_bounding_boxes != nullptr /* i.e. there was at least one accessor */) {
			backend->debug_free(oob_info.illegal_access_bounding_boxes);
		}
	}
#endif

	if(spdlog::should_log(spdlog::level::trace)) {
		if(const auto native_time = async.event.get_native_execution_time(); native_time.has_value()) {
			auto unit_time = std::chrono::duration_cast<std::chrono::duration<double>>(*native_time).count();
			auto unit = "s";
			if(unit_time < 1.0) { unit_time *= 1000.0, unit = "ms"; }
			if(unit_time < 1.0) { unit_time *= 1000.0, unit = "µs"; }
			if(unit_time < 1.0) { unit_time *= 1000.0, unit = "ns"; }
			CELERITY_TRACE("[executor] retired I{} after {:.2f} {}", async.instr->get_id(), unit_time, unit);
		} else {
			CELERITY_TRACE("[executor] retired I{}", async.instr->get_id());
		}
	}

#if CELERITY_ENABLE_TRACY
	if(async.tracy_cursor.has_value()) {
		auto& lane = *tracy_async_lanes.at(async.tracy_cursor->global_lane_id);
		TracyFiberEnter(lane.fiber_name.c_str());
		while(!lane.queued_zones.empty() && lane.queued_zones.front().submission_idx <= async.tracy_cursor->lane_submission_idx) {
			{
				tracy_end_async_zone(lane, lane.queued_zones.front());
				lane.queued_zones.pop();
			}
			if(!lane.queued_zones.empty()) {
				auto& next_zone = lane.queued_zones.front();
				tracy_begin_async_zone(lane, next_zone, true /* eager */);
			}
		}
		TracyFiberLeave;

		if(lane.target == out_of_order_engine::target::immediate) { tracy_immediate_async_lanes.at(lane.local_lane_id) = 0 /* not occupied */; }
	}
#endif

	if(utils::isa<alloc_instruction>(async.instr)) {
		const auto ainstr = utils::as<alloc_instruction>(async.instr);
		const auto ptr = async.event.get_result();
		assert(ptr != nullptr && "backend allocation returned nullptr");
		const auto aid = ainstr->get_allocation_id();
		CELERITY_TRACE("[executor] {} allocated as {}", aid, ptr);
		assert(allocations.count(aid) == 0);
		allocations.emplace(aid, ptr);
	}

	engine.complete_assigned(async.instr);
}

template <typename Instr>
auto executor_impl::dispatch_issue(const Instr& instr, const out_of_order_engine::assignment& assignment) //
    -> decltype(issue(instr))                                                                             //
{
	assert(assignment.target == out_of_order_engine::target::immediate);
	assert(!assignment.lane.has_value());
#if CELERITY_ENABLE_TRACY
	const auto zone = tracy_begin_zone(instr, false /* eager */);
	TracyPlot("active instrs", static_cast<int64_t>(in_flight_async_instructions.size() + 1));
#endif
	issue(instr);
	engine.complete_assigned(&instr);
#if CELERITY_ENABLE_TRACY
	tracy_end_zone(zone);
	TracyPlot("active instrs", static_cast<int64_t>(in_flight_async_instructions.size()));
#endif
}

template <typename Instr>
auto executor_impl::dispatch_issue(const Instr& instr, const out_of_order_engine::assignment& assignment)
    -> decltype(issue_async(instr, assignment, std::declval<async_instruction_state&>())) //
{
	auto& async = in_flight_async_instructions.emplace_back();
	async.instr = assignment.instruction;
	issue_async(instr, assignment, async);

#if CELERITY_ENABLE_TRACY
	const auto cursor = tracy_get_async_lane_cursor(
	    assignment.target, assignment.target == out_of_order_engine::target::device_queue ? assignment.device : std::nullopt, assignment.lane);
	auto& lane = *tracy_async_lanes.at(cursor.global_lane_id);
	TracyFiberEnter(lane.fiber_name.c_str());
	bool start_immediately = lane.queued_zones.empty();
	lane.queued_zones.push(tracy_async_zone{cursor.lane_submission_idx, &instr, {}});
	if(start_immediately) {
		tracy_begin_async_zone(lane, lane.queued_zones.front(), false /* eager */);
	} else {
		auto mark = fmt::format("I{} queued", instr.get_id());
		TracyMessageC(mark.data(), mark.size(), tracy::Color::DarkGray);
	}
	async.tracy_cursor = cursor;
	TracyFiberLeave;
	TracyPlot("active instrs", static_cast<int64_t>(in_flight_async_instructions.size()));
#endif
}

void executor_impl::try_issue_one_instruction() {
	auto assignment = engine.assign_one();
	if(!assignment.has_value()) return;

	CELERITY_DETAIL_TRACY_ZONE_SCOPED("executor::issue", Blue, "issue");
	matchbox::match(*assignment->instruction, [&](const auto& instr) { dispatch_issue(instr, *assignment); });
	made_progress = true;
}

void executor_impl::check_progress() {
	// TODO consider rate-limiting this (e.g. with an overflow counter) if steady_clock::now() turns out to have measurable latency
	if(made_progress) {
		last_progress_timestamp = std::chrono::steady_clock::now();
		progress_warning_emitted = false;
		made_progress = false;
	} else if(last_progress_timestamp.has_value()) {
		const auto assume_stuck_after = std::chrono::seconds(3);
		const auto elapsed_since_last_progress = std::chrono::steady_clock::now() - *last_progress_timestamp;
		if(elapsed_since_last_progress > assume_stuck_after && !progress_warning_emitted) {
			std::string instr_list;
			for(auto& in_flight : in_flight_async_instructions) {
				if(!instr_list.empty()) instr_list += ", ";
				fmt::format_to(std::back_inserter(instr_list), "I{}", in_flight.instr->get_id());
			}
			CELERITY_WARN("[executor] no progress for {:.3f} seconds, potentially stuck. Active instructions: {}",
			    std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_since_last_progress).count(),
			    in_flight_async_instructions.empty() ? "none" : instr_list);
			progress_warning_emitted = true;
		}
	}
}

void executor_impl::issue(const clone_collective_group_instruction& ccginstr) {
	const auto original_cgid = ccginstr.get_original_collective_group_id();
	assert(original_cgid != non_collective_group_id);
	assert(original_cgid == root_collective_group_id || cloned_communicators.count(original_cgid) != 0);

	const auto new_cgid = ccginstr.get_new_collective_group_id();
	assert(new_cgid != non_collective_group_id && new_cgid != root_collective_group_id);
	assert(cloned_communicators.count(new_cgid) == 0);

	CELERITY_TRACE("[executor] I{}: clone collective group CG{} -> CG{}", ccginstr.get_id(), original_cgid, new_cgid);

	const auto original_communicator = original_cgid == root_collective_group_id ? root_communicator : cloned_communicators.at(original_cgid).get();
	cloned_communicators.emplace(new_cgid, original_communicator->collective_clone());
}


void executor_impl::issue(const split_receive_instruction& srinstr) {
	CELERITY_TRACE("[executor] I{}: split receive {} {} into {} ({}), x{} bytes\n{} bytes total", srinstr.get_id(), srinstr.get_transfer_id(),
	    srinstr.get_requested_region(), srinstr.get_dest_allocation_id(), srinstr.get_allocated_box(), srinstr.get_element_size(),
	    srinstr.get_requested_region().get_area() * srinstr.get_element_size());

	const auto allocation = allocations.at(srinstr.get_dest_allocation_id());
	recv_arbiter.begin_split_receive(
	    srinstr.get_transfer_id(), srinstr.get_requested_region(), allocation, srinstr.get_allocated_box(), srinstr.get_element_size());
}

void executor_impl::issue(const fill_identity_instruction& fiinstr) {
	CELERITY_TRACE(
	    "[executor] I{}: fill identity {} x{} for R{}", fiinstr.get_id(), fiinstr.get_allocation_id(), fiinstr.get_num_values(), fiinstr.get_reduction_id());

	const auto allocation = allocations.at(fiinstr.get_allocation_id());
	const auto& reduction = *reducers.at(fiinstr.get_reduction_id());
	reduction.fill_identity(allocation, fiinstr.get_num_values());
}

void executor_impl::issue(const reduce_instruction& rinstr) {
	CELERITY_TRACE("[executor] I{}: reduce {} x{} into {} as R{}", rinstr.get_id(), rinstr.get_source_allocation_id(), rinstr.get_num_source_values(),
	    rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());

	const auto gather_allocation = allocations.at(rinstr.get_source_allocation_id());
	const auto dest_allocation = allocations.at(rinstr.get_dest_allocation_id());
	const auto& reduction = *reducers.at(rinstr.get_reduction_id());
	reduction.reduce(dest_allocation, gather_allocation, rinstr.get_num_source_values());
}

void executor_impl::issue(const fence_instruction& finstr) { // NOLINT(readability-convert-member-functions-to-static)
	CELERITY_TRACE("[executor] I{}: fence", finstr.get_id());

	finstr.get_promise()->fulfill();
}

void executor_impl::issue(const destroy_host_object_instruction& dhoinstr) {
	assert(host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
	CELERITY_TRACE("[executor] I{}: destroy H{}", dhoinstr.get_id(), dhoinstr.get_host_object_id());

	host_object_instances.erase(dhoinstr.get_host_object_id());
}

void executor_impl::issue(const horizon_instruction& hinstr) {
	CELERITY_TRACE("[executor] I{}: horizon", hinstr.get_id());

	if(delegate != nullptr) { delegate->horizon_reached(hinstr.get_horizon_task_id()); }
	collect(hinstr.get_garbage());
}

void executor_impl::issue(const epoch_instruction& einstr) {
	switch(einstr.get_epoch_action()) {
	case epoch_action::none: //
		CELERITY_TRACE("[executor] I{}: epoch", einstr.get_id());
		break;
	case epoch_action::barrier: //
		CELERITY_TRACE("[executor] I{}: epoch (barrier)", einstr.get_id());
		root_communicator->collective_barrier();
		break;
	case epoch_action::shutdown: //
		CELERITY_TRACE("[executor] I{}: epoch (shutdown)", einstr.get_id());
		expecting_more_submissions = false;
		break;
	}
	if(delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
		delegate->epoch_reached(einstr.get_epoch_task_id());
	}
	collect(einstr.get_garbage());
}

void executor_impl::issue_async(const alloc_instruction& ainstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(ainstr.get_allocation_id().get_memory_id() != user_memory_id);
	assert(assignment.target == out_of_order_engine::target::alloc_queue);
	assert(!assignment.lane.has_value());
	assert(assignment.device.has_value() == (ainstr.get_allocation_id().get_memory_id() > host_memory_id));

	CELERITY_TRACE(
	    "[executor] I{}: alloc {}, {} % {} bytes", ainstr.get_id(), ainstr.get_allocation_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_alloc(*assignment.device, ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
	} else {
		async.event = backend->enqueue_host_alloc(ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
	}
}

void executor_impl::issue_async(const free_instruction& finstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	const auto it = allocations.find(finstr.get_allocation_id());
	assert(it != allocations.end());
	const auto ptr = it->second;
	allocations.erase(it);

	CELERITY_TRACE("[executor] I{}: free {}", finstr.get_id(), finstr.get_allocation_id());

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_free(*assignment.device, ptr);
	} else {
		async.event = backend->enqueue_host_free(ptr);
	}
}

void executor_impl::issue_async(const copy_instruction& cinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::host_queue || assignment.target == out_of_order_engine::target::device_queue);
	assert((assignment.target == out_of_order_engine::target::device_queue) == assignment.device.has_value());
	assert(assignment.lane.has_value());

	CELERITY_TRACE("[executor] I{}: copy {} ({}) -> {} ({}), {} x{} bytes", cinstr.get_id(), cinstr.get_source_allocation(), cinstr.get_source_box(),
	    cinstr.get_dest_allocation(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());

	const auto source_base = static_cast<const std::byte*>(allocations.at(cinstr.get_source_allocation().id)) + cinstr.get_source_allocation().offset_bytes;
	const auto dest_base = static_cast<std::byte*>(allocations.at(cinstr.get_dest_allocation().id)) + cinstr.get_dest_allocation().offset_bytes;

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_copy(*assignment.device, *assignment.lane, source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(),
		    cinstr.get_copy_region(), cinstr.get_element_size());
	} else {
		async.event = backend->enqueue_host_copy(
		    *assignment.lane, source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());
	}
}

std::string print_accesses(const buffer_access_allocation_map& map) {
	std::string acc_log;
	for(size_t i = 0; i < map.size(); ++i) {
		auto& aa = map[i];
		const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
		    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
		fmt::format_to(std::back_inserter(acc_log), "{} {} {}", i == 0 ? ", accessing" : ",", aa.allocation_id, accessed_box_in_allocation);
	}
	return acc_log;
}

void executor_impl::issue_async(const device_kernel_instruction& dkinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::device_queue);
	assert(assignment.device == dkinstr.get_device_id());
	assert(assignment.lane.has_value());

	CELERITY_TRACE("[executor] I{}: launch device kernel on D{}, {}{}", dkinstr.get_id(), dkinstr.get_device_id(), dkinstr.get_execution_range(),
	    print_accesses(dkinstr.get_access_allocations()));

	prepare_accessor_hydration(dkinstr.get_id(), target::device,
	    dkinstr.get_access_allocations() //
	    CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, dkinstr.get_oob_task_type(), dkinstr.get_oob_task_id(), dkinstr.get_oob_task_name(), async.oob_info));

	std::vector<void*> reduction_ptrs;
	reduction_ptrs.reserve(dkinstr.get_reduction_allocations().size());
	for(const auto& ra : dkinstr.get_reduction_allocations()) {
		reduction_ptrs.push_back(allocations.at(ra.allocation_id));
	}

	async.event =
	    backend->enqueue_device_kernel(dkinstr.get_device_id(), *assignment.lane, dkinstr.get_launcher(), dkinstr.get_execution_range(), reduction_ptrs);
}

void executor_impl::issue_async(const host_task_instruction& htinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::host_queue);
	assert(!assignment.device.has_value());
	assert(assignment.lane.has_value());

	CELERITY_TRACE("[executor] I{}: launch host task, {}{}", htinstr.get_id(), htinstr.get_execution_range(), print_accesses(htinstr.get_access_allocations()));

	const auto collective_comm =
	    htinstr.get_collective_group_id() != non_collective_group_id ? cloned_communicators.at(htinstr.get_collective_group_id()).get() : nullptr;

	prepare_accessor_hydration(htinstr.get_id(), target::host_task,
	    htinstr.get_access_allocations() //
	    CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, htinstr.get_oob_task_type(), htinstr.get_oob_task_id(), htinstr.get_oob_task_name(), async.oob_info));
	auto launch_hydrated = closure_hydrator::get_instance().hydrate<target::host_task>(htinstr.get_launcher());

	const auto& execution_range = htinstr.get_execution_range();
	async.event = backend->enqueue_host_function(
	    *assignment.lane, [=, launch_hydrated = std::move(launch_hydrated)] { launch_hydrated(execution_range, collective_comm); });
}

void executor_impl::issue_async(
    const send_instruction& sinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: send {}+{}, {}x{} bytes to N{} (MSG{})", sinstr.get_id(), sinstr.get_source_allocation_id(),
	    sinstr.get_offset_in_source_allocation(), sinstr.get_send_range(), sinstr.get_element_size(), sinstr.get_dest_node_id(), sinstr.get_message_id());

	const auto allocation_base = allocations.at(sinstr.get_source_allocation_id());
	const communicator::stride stride{
	    sinstr.get_source_allocation_range(),
	    subrange<3>{sinstr.get_offset_in_source_allocation(), sinstr.get_send_range()},
	    sinstr.get_element_size(),
	};
	async.event = root_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_message_id(), allocation_base, stride);
}

void executor_impl::issue_async(
    const receive_instruction& rinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: receive {} {} into {} ({}), x{} bytes\n{} bytes total", rinstr.get_id(), rinstr.get_transfer_id(),
	    rinstr.get_requested_region(), rinstr.get_dest_allocation_id(), rinstr.get_allocated_box(), rinstr.get_element_size(),
	    rinstr.get_requested_region().get_area() * rinstr.get_element_size());

	const auto allocation = allocations.at(rinstr.get_dest_allocation_id());
	async.event =
	    recv_arbiter.receive(rinstr.get_transfer_id(), rinstr.get_requested_region(), allocation, rinstr.get_allocated_box(), rinstr.get_element_size());
}

void executor_impl::issue_async(
    const await_receive_instruction& arinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: await receive {} {}", arinstr.get_id(), arinstr.get_transfer_id(), arinstr.get_received_region());

	async.event = recv_arbiter.await_split_receive_subregion(arinstr.get_transfer_id(), arinstr.get_received_region());
}

void executor_impl::issue_async(
    const gather_receive_instruction& grinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: gather receive {} into {}, {} bytes per node", grinstr.get_id(), grinstr.get_transfer_id(),
	    grinstr.get_dest_allocation_id(), grinstr.get_node_chunk_size());

	const auto allocation = allocations.at(grinstr.get_dest_allocation_id());
	async.event = recv_arbiter.gather_receive(grinstr.get_transfer_id(), allocation, grinstr.get_node_chunk_size());
}

void executor_impl::collect(const instruction_garbage& garbage) {
	for(const auto rid : garbage.reductions) {
		assert(reducers.count(rid) != 0);
		reducers.erase(rid);
	}
	for(const auto aid : garbage.user_allocations) {
		assert(aid.get_memory_id() == user_memory_id);
		assert(allocations.count(aid) != 0);
		allocations.erase(aid);
	}
}

void executor_impl::prepare_accessor_hydration([[maybe_unused]] const instruction_id iid, target target,
    const buffer_access_allocation_map& amap CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(
        , const task_type tt, const task_id tid, const std::string& task_name, std::unique_ptr<boundary_check_info>& oob_info)) {
	std::vector<closure_hydrator::accessor_info> accessor_infos(amap.size());

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	if(!amap.empty()) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("executor::oob_init", Red, "I{} bounds check init", iid);
		oob_info = std::make_unique<boundary_check_info>();

		// SYCL host memory is writeable by devices
		oob_info->illegal_access_bounding_boxes =
		    static_cast<oob_bounding_box*>(backend->debug_alloc(amap.size() * sizeof(oob_bounding_box))); // uh oh this needs to be sync?
		std::uninitialized_default_construct_n(oob_info->illegal_access_bounding_boxes, amap.size());

		for(size_t i = 0; i < amap.size(); ++i) {
			oob_info->accessors.push_back({amap[i].oob_buffer_id, amap[i].oob_buffer_name, amap[i].accessed_box_in_buffer});
		}
		oob_info->task_type = tt;
		oob_info->task_id = tid;
		oob_info->task_name = task_name;
	}
#endif

	for(size_t i = 0; i < amap.size(); ++i) {
		const auto ptr = allocations.at(amap[i].allocation_id);
		accessor_infos[i] = closure_hydrator::accessor_info{ptr, amap[i].allocated_box_in_buffer,
		    amap[i].accessed_box_in_buffer CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, &oob_info->illegal_access_bounding_boxes[i])};
	}

	closure_hydrator::get_instance().arm(target, std::move(accessor_infos));
}

} // namespace celerity::detail::live_executor_detail

namespace celerity::detail {

live_executor::live_executor(const system_info& system, std::unique_ptr<backend> backend, std::unique_ptr<communicator> root_comm, delegate* const dlg)
    : m_root_comm(std::move(root_comm)), m_thread(&live_executor::thread_main, this, system, std::move(backend), dlg) //
{
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

live_executor::~live_executor() { wait(); }

void live_executor::announce_user_allocation(const allocation_id aid, void* const ptr) {
	m_submission_queue.push(live_executor_detail::user_allocation_announcement{aid, ptr});
}

void live_executor::announce_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push(live_executor_detail::host_object_instance_announcement{hoid, std::move(instance)});
}

void live_executor::announce_reducer(const reduction_id rid, std::unique_ptr<reducer> reducer) {
	assert(reducer != nullptr);
	m_submission_queue.push(live_executor_detail::reducer_announcement{rid, std::move(reducer)});
}

void live_executor::submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) {
	m_submission_queue.push(live_executor_detail::instruction_pilot_batch{std::move(instructions), std::move(pilots)});
}

void live_executor::wait() {
	if(m_thread.joinable()) { m_thread.join(); }
}

void live_executor::thread_main(const system_info& system, std::unique_ptr<backend> backend, delegate* const dlg) {
	CELERITY_DETAIL_TRACY_SET_THREAD_NAME("cy-executor");
	try {
		live_executor_detail::executor_impl(system, std::move(backend), m_root_comm.get(), m_submission_queue, dlg).run();
	} catch(const std::exception& e) {
		CELERITY_CRITICAL("[executor] {}", e.what());
		std::abort();
	}
}

} // namespace celerity::detail
