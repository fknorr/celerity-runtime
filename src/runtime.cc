#include "runtime.h"

#include <limits>
#include <string>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mpi.h>

#if CELERITY_USE_MIMALLOC
// override default new/delete operators to use the mimalloc memory allocator
#include <mimalloc-new-delete.h>
#endif

#include "affinity.h"
#include "cgf_diagnostics.h"
#include "command_graph.h"
#include "device_selection.h"
#include "distributed_graph_generator.h"
#include "host_object.h"
#include "instruction_executor.h"
#include "instruction_graph_generator.h"
#include "log.h"
#include "mpi_communicator.h"
#include "mpi_support.h"
#include "print_graph.h"
#include "reduction.h"
#include "scheduler.h"
#include "task_manager.h"
#include "tracy.h"
#include "user_bench.h"
#include "version.h"

namespace celerity {
namespace detail {

	std::unique_ptr<runtime> runtime::s_instance = nullptr;

	void runtime::mpi_initialize_once(int* argc, char*** argv) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("mpi::init", LightSkyBlue, "MPI_Init");
		assert(!s_mpi_initialized);
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);
		s_mpi_initialized = true;
	}

	void runtime::mpi_finalize_once() {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("mpi::finalize", LightSkyBlue, "MPI_Finalize");
		assert(s_mpi_initialized && !s_mpi_finalized && (!s_test_mode || !s_instance));
		MPI_Finalize();
		s_mpi_finalized = true;
	}

	void runtime::init(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector) {
		assert(!s_instance);
		// if (!s_test_mode && s_initialized_before) { throw std::runtime_error("Cannot re-initialize the runtime"); }
		s_instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_devices_or_selector));
	}

	runtime& runtime::get_instance() {
		if(s_instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *s_instance;
	}

	static auto get_pid() {
#ifdef _MSC_VER
		return _getpid();
#else
		return getpid();
#endif
	}

	static std::string get_version_string() {
		using namespace celerity::version;
		return fmt::format("{}.{}.{} {}{}", major, minor, patch, git_revision, git_dirty ? "-dirty" : "");
	}

	static const char* get_build_type() {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		return "debug";
#else
		return "release";
#endif
	}

	static const char* get_mimalloc_string() {
#if CELERITY_USE_MIMALLOC
		return "using mimalloc";
#else
		return "using the default allocator";
#endif
	}

	static std::string get_sycl_version() {
#if defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
		return fmt::format("hipSYCL {}.{}.{}", HIPSYCL_VERSION_MAJOR, HIPSYCL_VERSION_MINOR, HIPSYCL_VERSION_PATCH);
#elif CELERITY_DPCPP
		return "DPC++ / Clang " __clang_version__;
#elif CELERITY_SIMSYCL
		return "SimSYCL " SIMSYCL_VERSION;
#else
#error "unknown SYCL implementation"
#endif
	}

	runtime::runtime(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("runtime::startup", Gray, "runtime startup");

		m_application_thread = std::this_thread::get_id();

		if(s_test_mode) {
			assert(s_test_active && "initializing the runtime from a test without a runtime_fixture");
			s_test_runtime_was_instantiated = true;
		} else {
			mpi_initialize_once(argc, argv);
		}

		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		m_num_nodes = world_size;

		m_cfg = std::make_unique<config>(argc, argv);
		if(m_cfg->is_dry_run()) { m_num_nodes = m_cfg->get_dry_run_nodes(); }

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		m_local_nid = world_rank;

		if(!s_test_mode) { // do not touch logger settings in tests, where the full (trace) logs are captured
			spdlog::set_level(m_cfg->get_log_level());
			spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", world_rank, int(ceil(log10(world_size)))));
		}

#ifndef __APPLE__
		if(const uint32_t cores = affinity_cores_available(); cores < min_cores_needed) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} "
			              "logical cores. Performance may be negatively impacted.",
			    cores, min_cores_needed);
		}
#endif

		if(CELERITY_ENABLE_TRACY) { CELERITY_WARN("Instrumentation for profiling with Tracy is enabled. Performance may be negatively impacted."); }

		m_user_bench = std::make_unique<experimental::bench::detail::user_benchmarker>(*m_cfg, static_cast<node_id>(world_rank));

		cgf_diagnostics::make_available();

		m_h_queue = std::make_unique<host_queue>();

		if(m_cfg->should_record()) {
			m_task_recorder = std::make_unique<task_recorder>();
			m_command_recorder = std::make_unique<command_recorder>();
			m_instruction_recorder = std::make_unique<instruction_recorder>();
		}

		task_manager::policy_set task_mngr_policy;
		// Merely _declaring_ an uninitialized read is legitimate as long as the kernel does not actually perform the read at runtime - this might happen in the
		// first iteration of a submit-loop. We could get rid of this case by making access-modes a runtime property of accessors (cf
		// https://github.com/celerity/meta/issues/74).
		task_mngr_policy.uninitialized_read_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_warning : error_policy::ignore;

		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, m_h_queue.get(), m_task_recorder.get(), task_mngr_policy);
		if(m_cfg->get_horizon_step()) m_task_mngr->set_horizon_step(m_cfg->get_horizon_step().value());
		if(m_cfg->get_horizon_max_parallelism()) m_task_mngr->set_horizon_max_parallelism(m_cfg->get_horizon_max_parallelism().value());

		std::vector<sycl::device> devices;
		{
			CELERITY_DETAIL_TRACY_SCOPED_ZONE("runtime::pick_devices", PaleVioletRed, "device selection");
			devices =
			    matchbox::match(user_devices_or_selector, [&](const auto& value) { return pick_devices(*m_cfg, value, sycl::platform::get_platforms()); });
			assert(!devices.empty());
		}

		const auto backend_type = backend::get_effective_type(devices.front());
		assert(std::all_of(devices.begin(), devices.end(), [=](const sycl::device& d) { return backend::get_effective_type(d) == backend_type; }));

		// TODO refactor / rename frontend_system_info / backend_system_info?
		// Or maybe should the backend ingest the config and then provide a list of devices / the correct frontend config?
		std::vector<backend::device_config> backend_devices(devices.size());
		instruction_graph_generator::system_info system_info;
		system_info.devices.resize(devices.size());
		system_info.memories.resize(first_device_memory_id + devices.size());
		system_info.memories[user_memory_id].copy_peers.set(user_memory_id);
		system_info.memories[host_memory_id].copy_peers.set(host_memory_id);
		// user memory is only good for copying from and to host memory
		system_info.memories[user_memory_id].copy_peers.set(host_memory_id);
		system_info.memories[host_memory_id].copy_peers.set(user_memory_id);
		for(device_id did = 0; did < devices.size(); ++did) {
			// TODO query the backend about how memory is attached to devices - we want to support SoCs with shared memory
			const auto native_memory = memory_id(first_device_memory_id + did);
			backend_devices[did].device_id = did;
			backend_devices[did].native_memory = native_memory;
			backend_devices[did].sycl_device = devices[did];
			system_info.devices[did].native_memory = native_memory;
			system_info.memories[native_memory].copy_peers.set(host_memory_id);
			system_info.memories[native_memory].copy_peers.set(native_memory);
			system_info.memories[host_memory_id].copy_peers.set(native_memory);
			CELERITY_DEBUG("Device D{} with native memory M{} is {}", backend_devices[did].device_id, backend_devices[did].native_memory,
			    backend_devices[did].sycl_device.get_info<sycl::info::device::name>());
		}
		for(device_id did_a = 0; did_a < devices.size(); ++did_a) {
			const auto mid_a = system_info.devices[did_a].native_memory;
			for(device_id did_b = did_a + 1; did_b < devices.size(); ++did_b) {
				const auto mid_b = system_info.devices[did_b].native_memory;
				if(backend::enable_copy_between_peer_memories(devices[did_a], devices[did_b])) {
					system_info.memories[mid_a].copy_peers.set(mid_b);
					system_info.memories[mid_b].copy_peers.set(mid_a);
				} else {
					CELERITY_DEBUG("No peer copies possible between D{} and D{}, will stage through host memory", did_a, did_b);
				}
			}
		}
		m_num_local_devices = devices.size();

		m_exec = std::make_unique<instruction_executor>(backend::make_queue(backend_type, backend_devices), std::make_unique<mpi_communicator>(MPI_COMM_WORLD),
		    static_cast<instruction_executor::delegate*>(this));

		scheduler::policy_set schdlr_policy;
		// Any uninitialized read that is observed on CDAG generation was already logged on task generation, unless we have a bug.
		schdlr_policy.command_graph_generator.uninitialized_read_error = error_policy::ignore;
		schdlr_policy.instruction_graph_generator.uninitialized_read_error = error_policy::ignore;
		schdlr_policy.command_graph_generator.overlapping_write_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_error : error_policy::ignore;
		schdlr_policy.instruction_graph_generator.overlapping_write_error =
		    CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_error : error_policy::ignore;
		schdlr_policy.instruction_graph_generator.unsafe_oversubscription_error = error_policy::log_warning;

		m_schdlr = std::make_unique<scheduler>(m_num_nodes, m_local_nid, std::move(system_info), *m_task_mngr, static_cast<abstract_scheduler::delegate*>(this),
		    m_command_recorder.get(), m_instruction_recorder.get(), schdlr_policy);
		m_task_mngr->register_task_callback([this](const task* tsk) { m_schdlr->notify_task_created(tsk); });

		CELERITY_INFO("Celerity runtime version {} running on {}. PID = {}, build type = {}, {}", get_version_string(), get_sycl_version(), get_pid(),
		    get_build_type(), get_mimalloc_string());
	}

	void runtime::require_call_from_application_thread() const {
		if(std::this_thread::get_id() != m_application_thread) {
			throw std::runtime_error("Celerity runtime, distr_queue, handler, buffer and host_object types must only be constructed, used, and destroyed from "
			                         "the application thread. Make sure that you did not accidentally capture one of these types in a host_task.");
		}
	}

	runtime::~runtime() {
		if(!is_unreferenced()) {
			// this call might originate from static destruction - we cannot assume spdlog to still be around
			fputs("[CRITICAL] Detected an attempt to destroy runtime while at least one distr_queue, buffer or host_object was still alive. This likely means "
			      "that one of these objects was leaked, or at least its lifetime extended beyond the scope of main(). This is undefined.\n",
			    stderr);
			abort();
		}

		CELERITY_DETAIL_TRACY_SCOPED_ZONE("runtime::shutdown", Gray, "runtime shutdown");

		require_call_from_application_thread();

		// create a shutdown epoch and pass it to the scheduler via callback
		const auto shutdown_epoch = m_task_mngr->generate_epoch_task(epoch_action::shutdown);

		// Wait for the executor to exit its thread after processing the shutdown-epoch instruction.
		m_exec->wait();

		// The scheduler will have exited by now as well after processing the shutdown epoch. Destroy it *after* the executor because it owns the CDAG and IDAG.
		m_schdlr.reset();

		// with the scheduler gone, nobody will submit instructions and pilots to the runtime anymore and we can get rid of the executor.
		m_exec.reset();

		// when the executor is gone, the host queue is guaranteed to not have any work left to do
		m_h_queue.reset();

		// TODO does this actually do anything? Once the executor has exited we are guaranteed to arrived at this epoch anyway
		m_task_mngr->await_epoch(shutdown_epoch);

		if(spdlog::should_log(log_level::info) && m_cfg->should_print_graphs()) {
			if(m_local_nid == 0) { // It's the same across all nodes
				assert(m_task_recorder.get() != nullptr);
				const auto graph_str = detail::print_task_graph(*m_task_recorder);
				CELERITY_INFO("Task graph:\n\n{}\n", graph_str);
			}
			// must be called on all nodes
			auto cmd_graph = gather_command_graph();
			if(m_local_nid == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Avoid racing on stdout with other nodes (funneled through mpirun)
				CELERITY_INFO("Command graph:\n\n{}\n", cmd_graph);

				// IDAGs become unreadable when all nodes print them at the same time - TODO attempt gathering them as well?
				// we are allowed to deref m_instruction_recorder / m_command_recorder because the scheduler thread has exited at this point
				CELERITY_TRACE(
				    "Instruction graph on node 0:\n\n{}\n", detail::print_instruction_graph(*m_instruction_recorder, *m_command_recorder, *m_task_recorder));
			}
		}

		m_task_mngr.reset();

		// all buffers and host objects should have unregistered themselves by now.
		assert(m_live_buffers.empty());
		assert(m_live_host_objects.empty());
		m_h_queue.reset();

		m_instruction_recorder.reset();
		m_command_recorder.reset();
		m_task_recorder.reset();

		cgf_diagnostics::teardown();

		m_user_bench.reset();

		if(!s_test_mode) { mpi_finalize_once(); }
	}

	void runtime::sync(epoch_action action) {
		require_call_from_application_thread();
		const auto epoch = m_task_mngr->generate_epoch_task(action);
		m_task_mngr->await_epoch(epoch);
	}

	task_manager& runtime::get_task_manager() const {
		require_call_from_application_thread();
		return *m_task_mngr;
	}

	std::string runtime::gather_command_graph() const {
		require_call_from_application_thread();

		assert(m_command_recorder.get() != nullptr);
		const auto graph_str = print_command_graph(m_local_nid, *m_command_recorder);

		// Send local graph to rank 0 on all other nodes
		if(m_local_nid != 0) {
			const uint64_t usize = graph_str.size();
			assert(usize < std::numeric_limits<int32_t>::max());
			const int32_t size = static_cast<int32_t>(usize);
			MPI_Send(&size, 1, MPI_INT32_T, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
			if(size > 0) MPI_Send(graph_str.data(), static_cast<int32_t>(size), MPI_BYTE, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
			return "";
		}
		// On node 0, receive and combine
		std::vector<std::string> graphs;
		graphs.push_back(graph_str);
		for(size_t i = 1; i < m_num_nodes; ++i) {
			int32_t size = 0;
			MPI_Recv(&size, 1, MPI_INT32_T, static_cast<int>(i), mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(size > 0) {
				std::string graph;
				graph.resize(size);
				MPI_Recv(graph.data(), size, MPI_BYTE, static_cast<int>(i), mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				graphs.push_back(std::move(graph));
			}
		}
		return combine_command_graphs(graphs);
	}

	// TODO I don't like this. IMO we should have a dry_run_executor that noop's all instructions except epoch / horizon / fence and just implement the
	// scheduler delegate in the the executor base class.
	void runtime::flush_instructions(std::vector<const instruction*> instrs) {
		// thread-safe
		assert(m_exec != nullptr);
		// TODO avoid this loop because it will acquire and release the executor queue lock every iteration
		for(const auto instr : instrs) {
			if(!is_dry_run() || utils::isa<epoch_instruction>(instr) || utils::isa<horizon_instruction>(instr) || utils::isa<fence_instruction>(instr)) {
				m_exec->submit_instruction(*instr);
			}
		}
	}

	void runtime::flush_outbound_pilots(std::vector<outbound_pilot> pilots) {
		// thread-safe
		assert(m_exec != nullptr);
		if(!is_dry_run()) {
			// TODO avoid this loop because it will acquire and release the executor queue lock every iteration
			for(const auto& pilot : pilots) {
				m_exec->submit_pilot(pilot);
			}
		}
	}

	void runtime::horizon_reached(const task_id horizon_tid) {
		// thread-safe
		m_task_mngr->notify_horizon_reached(horizon_tid);

		// TODO duplicated from task_manager::notify_horizon_reached
		if(m_latest_horizon_reached.has_value()) { m_schdlr->notify_epoch_reached(*m_latest_horizon_reached); }
		m_latest_horizon_reached = horizon_tid;
	}

	void runtime::epoch_reached(const task_id epoch_tid) {
		m_task_mngr->notify_epoch_reached(epoch_tid); // thread-safe

		m_schdlr->notify_epoch_reached(epoch_tid);
		m_latest_horizon_reached = std::nullopt; // Any non-applied horizon is now behind the epoch and will therefore never become an epoch itself
	}

	void runtime::create_queue() {
		require_call_from_application_thread();

		if(m_has_live_queue) { throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)"); }
		m_has_live_queue = true;
	}

	void runtime::destroy_queue() {
		require_call_from_application_thread();

		assert(m_has_live_queue);
		m_has_live_queue = false;
		destroy_instance_if_unreferenced();
	}

	allocation_id runtime::create_user_allocation(void* const ptr) {
		require_call_from_application_thread();
		const auto aid = allocation_id(user_memory_id, m_next_user_allocation_id++);
		m_exec->announce_user_allocation(aid, ptr);
		return aid;
	}

	buffer_id runtime::create_buffer(const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_aid) {
		require_call_from_application_thread();

		const auto bid = m_next_buffer_id++;
		m_live_buffers.emplace(bid);
		m_task_mngr->create_buffer(bid, dims, range, user_aid != null_allocation_id);
		m_schdlr->notify_buffer_created(bid, dims, range, elem_size, elem_align, user_aid);
		return bid;
	}

	void runtime::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) {
		require_call_from_application_thread();

		assert(m_live_buffers.count(bid) != 0);
		m_task_mngr->set_buffer_debug_name(bid, debug_name);
		m_schdlr->set_buffer_debug_name(bid, debug_name);
	}

	void runtime::destroy_buffer(const buffer_id bid) {
		require_call_from_application_thread();

		assert(m_live_buffers.count(bid) != 0);
		m_schdlr->notify_buffer_destroyed(bid);
		m_task_mngr->destroy_buffer(bid);
		m_live_buffers.erase(bid);
		destroy_instance_if_unreferenced();
	}

	host_object_id runtime::create_host_object(std::unique_ptr<host_object_instance> instance) {
		require_call_from_application_thread();

		const auto hoid = m_next_host_object_id++;
		m_live_host_objects.emplace(hoid);
		const bool owns_instance = instance != nullptr;
		if(owns_instance) { m_exec->announce_host_object_instance(hoid, std::move(instance)); }
		m_task_mngr->create_host_object(hoid);
		m_schdlr->notify_host_object_created(hoid, owns_instance);
		return hoid;
	}

	void runtime::destroy_host_object(const host_object_id hoid) {
		require_call_from_application_thread();

		assert(m_live_host_objects.count(hoid) != 0);
		m_schdlr->notify_host_object_destroyed(hoid);
		m_task_mngr->destroy_host_object(hoid);
		m_live_host_objects.erase(hoid);
		destroy_instance_if_unreferenced();
	}


	reduction_id runtime::create_reduction(std::unique_ptr<runtime_reduction> reduction) {
		require_call_from_application_thread();

		const auto rid = m_next_reduction_id++;
		m_exec->announce_reduction(rid, std::move(reduction));
		return rid;
	}

	bool runtime::is_unreferenced() const { return !m_has_live_queue && m_live_buffers.empty() && m_live_host_objects.empty(); }

	void runtime::destroy_instance_if_unreferenced() {
		if(s_instance == nullptr) return;
		if(s_instance->is_unreferenced()) { s_instance.reset(); }
	}

} // namespace detail
} // namespace celerity
