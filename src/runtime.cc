#include "runtime.h"

#include <queue>
#include <string>
#include <unordered_map>

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
#include "buffer.h"
#include "buffer_manager.h"
#include "cgf_diagnostics.h"
#include "command_graph.h"
#include "device_queue.h"
#include "distributed_graph_generator.h"
#include "executor.h"
#include "host_object.h"
#include "instruction_executor.h"
#include "instruction_graph_generator.h"
#include "log.h"
#include "mpi_communicator.h"
#include "mpi_support.h"
#include "named_threads.h"
#include "print_graph.h"
#include "scheduler.h"
#include "task_manager.h"
#include "user_bench.h"
#include "utils.h"
#include "version.h"

namespace celerity {
namespace detail {

	std::unique_ptr<runtime> runtime::s_instance = nullptr;

	void runtime::mpi_initialize_once(int* argc, char*** argv) {
		assert(!s_mpi_initialized);
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);
		s_mpi_initialized = true;
	}

	void runtime::mpi_finalize_once() {
		assert(s_mpi_initialized && !s_mpi_finalized && (!s_test_mode || !s_instance));
		MPI_Finalize();
		s_mpi_finalized = true;
	}

	void runtime::init(int* argc, char** argv[], const device_or_selector& user_device_or_selector) {
		assert(!s_instance);
		// TODO if (!s_test_mode && s_initialized_before) { throw std::runtime_error("Cannot re-initialize the runtime"); }
		s_instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device_or_selector));
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
#else
#error "unknown SYCL implementation"
#endif
	}

	runtime::runtime(int* argc, char** argv[], const device_or_selector& user_device_or_selector) {
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

		spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", world_rank, int(ceil(log10(world_size)))));

#ifndef __APPLE__
		if(const uint32_t cores = affinity_cores_available(); cores < min_cores_needed) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} "
			              "logical cores. Performance may be negatively impacted.",
			    cores, min_cores_needed);
		}
#endif
		m_user_bench = std::make_unique<experimental::bench::detail::user_benchmarker>(*m_cfg, static_cast<node_id>(world_rank));

		cgf_diagnostics::make_available();

		m_h_queue = std::make_unique<host_queue>();

		m_reduction_mngr = std::make_unique<reduction_manager>();
		if(m_cfg->is_recording()) m_task_recorder = std::make_unique<task_recorder>();
		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, m_h_queue.get(), m_task_recorder.get());
		if(m_cfg->get_horizon_step()) m_task_mngr->set_horizon_step(m_cfg->get_horizon_step().value());
		if(m_cfg->get_horizon_max_parallelism()) m_task_mngr->set_horizon_max_parallelism(m_cfg->get_horizon_max_parallelism().value());
		m_cdag = std::make_unique<command_graph>();
		if(m_cfg->is_recording()) m_command_recorder = std::make_unique<command_recorder>();
		auto dggen = std::make_unique<distributed_graph_generator>(m_num_nodes, m_local_nid, *m_cdag, *m_task_mngr, m_command_recorder.get());

		// TODO very simplistic device selection: Select all GPUs, and assume that each has their own distinct memory
		auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
		if(gpus.empty()) utils::panic("No GPUs found!");

		const auto backend_type = backend::get_type(gpus[0]);
		if(!std::all_of(gpus.begin(), gpus.end(), [&](const sycl::device& d) { return backend::get_type(d) == backend_type; })) {
			utils::panic("Found multiple GPUs with different backends!");
		}

		std::vector<backend::device_config> backend_devices(gpus.size());
		for(size_t i = 0; i < gpus.size(); ++i) {
			backend_devices[i].device_id = i;
			backend_devices[i].native_memory = 1 + i;
			backend_devices[i].sycl_device = gpus[i];

			CELERITY_INFO("Using device D{}, memory M{}: {} {}", backend_devices[i].device_id, backend_devices[i].native_memory,
			    gpus[i].get_info<sycl::info::device::vendor>(), backend_devices[i].sycl_device.get_info<sycl::info::device::name>());
		}
		m_exec = std::make_unique<instruction_executor>(
		    backend::make_queue(backend_type, backend_devices), mpi_communicator_factory(MPI_COMM_WORLD), static_cast<instruction_executor::delegate*>(this));

		std::vector<instruction_graph_generator::device_info> device_infos(gpus.size());
		for(size_t i = 0; i < gpus.size(); ++i) {
			device_infos[i].native_memory = memory_id(1 + i);
		}
		if(m_cfg->is_recording()) m_instruction_recorder = std::make_unique<instruction_recorder>();
		auto iggen = std::make_unique<instruction_graph_generator>(*m_task_mngr, std::move(device_infos), m_instruction_recorder.get());

		m_schdlr = std::make_unique<scheduler>(is_dry_run(), std::move(dggen), std::move(iggen), *m_exec);
		m_schdlr->startup(); // TODO move into scheduler ctor
		m_task_mngr->register_task_callback([this](const task* tsk) { m_schdlr->notify_task_created(tsk); });

		CELERITY_INFO("Celerity runtime version {} running on {}. PID = {}, build type = {}, {}", get_version_string(), get_sycl_version(), get_pid(),
		    get_build_type(), get_mimalloc_string());
	}

	runtime::~runtime() {
		// create a shutdown epoch and pass it to the scheduler via callback
		const auto shutdown_epoch = m_task_mngr->generate_epoch_task(epoch_action::shutdown);

		// Shut down the scheduler after enqueueing the shutdown epoch
		m_schdlr->shutdown();

		// executor destructor waits for its thread, which exits as soon as it has processed the shutdown-epoch instruction
		m_exec.reset();

		// when the executor is gone, the host queue is guaranteed to not have any work left to do
		m_h_queue.reset();

		// TODO does this actually do anything? Once the executor has exited we are guaranteed to arrived at this epoch anyway
		m_task_mngr->await_epoch(shutdown_epoch);

		if(spdlog::should_log(log_level::trace) && m_cfg->is_recording()) {
			if(m_local_nid == 0) { // It's the same across all nodes
				assert(m_task_recorder.get() != nullptr);
				const auto graph_str = detail::print_task_graph(*m_task_recorder);
				CELERITY_TRACE("Task graph:\n\n{}\n", graph_str);
			}
			// must be called on all nodes
			auto cmd_graph = gather_command_graph();
			if(m_local_nid == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Avoid racing on stdout with other nodes (funneled through mpirun)
				CELERITY_TRACE("Command graph:\n\n{}\n", cmd_graph);

				// IDAGs become unreadable when all nodes print them at the same time - TODO attempt gathering them as well?
				CELERITY_TRACE(
				    "Instruction graph on node 0:\n\n{}\n", detail::print_instruction_graph(*m_instruction_recorder, *m_command_recorder, *m_task_recorder));
			}
		}

		m_cdag.reset();
		m_task_mngr.reset();

		// all buffers and host objects should have unregistered themselves by now.
		assert(m_live_buffers.empty());
		assert(m_live_host_objects.empty());
		m_reduction_mngr.reset();
		m_h_queue.reset();
		m_command_recorder.reset();
		m_task_recorder.reset();

		cgf_diagnostics::teardown();

		m_user_bench.reset();

		if(!s_test_mode) { mpi_finalize_once(); }
	}

	void runtime::sync(epoch_action action) {
		const auto epoch = m_task_mngr->generate_epoch_task(action);
		m_task_mngr->await_epoch(epoch);
	}

	task_manager& runtime::get_task_manager() const { return *m_task_mngr; }

	reduction_manager& runtime::get_reduction_manager() const { return *m_reduction_mngr; }

	std::string runtime::gather_command_graph() const {
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

	void runtime::horizon_reached(const task_id horizon_tid) { m_task_mngr->notify_horizon_reached(horizon_tid); }

	void runtime::epoch_reached(const task_id epoch_tid) { m_task_mngr->notify_epoch_reached(epoch_tid); }

	void runtime::create_queue() {
		if(m_has_live_queue) { throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)"); }
		m_has_live_queue = true;
	}

	void runtime::destroy_queue() {
		assert(m_has_live_queue);
		m_has_live_queue = false;
		destroy_instance_if_unreferenced();
	}

	buffer_id runtime::create_buffer(const int dims, const range<3>& range, const size_t elem_size, const size_t elem_align, const void* const host_init_ptr) {
		const auto bid = m_next_buffer_id++;
		m_live_buffers.emplace(bid);
		const auto is_host_initialized = host_init_ptr != nullptr;
		if(is_host_initialized) { m_exec->announce_buffer_user_pointer(bid, host_init_ptr); }
		m_task_mngr->create_buffer(bid, dims, range, is_host_initialized);
		m_schdlr->notify_buffer_created(bid, dims, range, elem_size, elem_align, is_host_initialized);
		return bid;
	}

	void runtime::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) {
		assert(m_live_buffers.count(bid) != 0);
		m_task_mngr->set_buffer_debug_name(bid, debug_name);
		m_schdlr->set_buffer_debug_name(bid, debug_name);
	}

	void runtime::destroy_buffer(const buffer_id bid) {
		assert(m_live_buffers.count(bid) != 0);
		m_schdlr->notify_buffer_destroyed(bid);
		m_task_mngr->destroy_buffer(bid);
		m_live_buffers.erase(bid);
		destroy_instance_if_unreferenced();
	}

	host_object_id runtime::create_host_object(std::unique_ptr<host_object_instance> instance) {
		const auto hoid = m_next_host_object_id++;
		m_live_host_objects.emplace(hoid);
		const bool owns_instance = instance != nullptr;
		if(owns_instance) { m_exec->announce_host_object_instance(hoid, std::move(instance)); }
		m_task_mngr->create_host_object(hoid);
		m_schdlr->notify_host_object_created(hoid, owns_instance);
		return hoid;
	}

	void runtime::destroy_host_object(const host_object_id hoid) {
		assert(m_live_host_objects.count(hoid) != 0);
		m_schdlr->notify_host_object_destroyed(hoid);
		m_task_mngr->destroy_host_object(hoid);
		m_live_host_objects.erase(hoid);
		destroy_instance_if_unreferenced();
	}

	void runtime::destroy_instance_if_unreferenced() const {
		if(!m_has_live_queue && m_live_buffers.empty() && m_live_host_objects.empty()) { //
			s_instance.reset();
		}
	}

} // namespace detail
} // namespace celerity
