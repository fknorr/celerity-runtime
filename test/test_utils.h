#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <celerity.h>

#include "command.h"
#include "command_graph.h"
#include "device_queue.h"
#include "distributed_graph_generator.h"
#include "graph_generator.h"
#include "graph_serializer.h"
#include "print_graph.h"
#include "range_mapper.h"
#include "runtime.h"
#include "scheduler.h"
#include "task_manager.h"
#include "transformers/naive_split.h"
#include "types.h"

// To avoid having to come up with tons of unique kernel names, we simply use the CPP counter.
// This is non-standard but widely supported.
#define _UKN_CONCAT2(x, y) x##_##y
#define _UKN_CONCAT(x, y) _UKN_CONCAT2(x, y)
#define UKN(name) _UKN_CONCAT(name, __COUNTER__)

/**
 * REQUIRE_LOOP is a utility macro for performing Catch2 REQUIRE assertions inside of loops.
 * The advantage over using a regular REQUIRE is that the number of reported assertions is much lower,
 * as only the first iteration is actually passed on to Catch2 (useful when showing successful assertions with `-s`).
 * If an expression result is false, it will also be forwarded to Catch2.
 *
 * NOTE: Since the checked expression will be evaluated twice, it must be idempotent!
 */
#define REQUIRE_LOOP(...) CELERITY_DETAIL_REQUIRE_LOOP(__VA_ARGS__)

namespace celerity::test_utils {
class dist_cdag_test_context; // Forward decl b/c we want to be mock_buffer's friend
};

namespace celerity {
namespace detail {

	struct runtime_testspy {
		static scheduler& get_schdlr(runtime& rt) { return *rt.m_schdlr; }
		static executor& get_exec(runtime& rt) { return *rt.m_exec; }
		static size_t get_command_count(runtime& rt) { return rt.m_cdag->command_count(); }
		static command_graph& get_cdag(runtime& rt) { return *rt.m_cdag; }
		static std::string print_graph(runtime& rt) {
			return rt.m_cdag.get()->print_graph(0, std::numeric_limits<size_t>::max(), *rt.m_task_mngr, rt.m_buffer_mngr.get()).value();
		}
	};

	struct task_ring_buffer_testspy {
		static void create_task_slot(task_ring_buffer& trb) { trb.m_number_of_deleted_tasks += 1; }
	};

	struct task_manager_testspy {
		static std::optional<task_id> get_current_horizon(task_manager& tm) { return tm.m_current_horizon; }

		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto task_ptr : tm.m_task_buffer) {
				if(task_ptr->get_type() == task_type::horizon) { horizon_counter++; }
			}
			return horizon_counter;
		}

		static region_map<std::optional<task_id>> get_last_writer(task_manager& tm, const buffer_id bid) { return tm.m_buffers_last_writers.at(bid); }

		static int get_max_pseudo_critical_path_length(task_manager& tm) { return tm.get_max_pseudo_critical_path_length(); }

		static auto get_execution_front(task_manager& tm) { return tm.get_execution_front(); }

		static void create_task_slot(task_manager& tm) { task_ring_buffer_testspy::create_task_slot(tm.m_task_buffer); }
	};

	inline bool has_dependency(const task_manager& tm, task_id dependent, task_id dependency, dependency_kind kind = dependency_kind::true_dep) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency && dep.kind == kind) return true;
		}
		return false;
	}

	inline bool has_any_dependency(const task_manager& tm, task_id dependent, task_id dependency) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency) return true;
		}
		return false;
	}
} // namespace detail

namespace test_utils {
	class require_loop_assertion_registry {
	  public:
		static require_loop_assertion_registry& get_instance() {
			if(instance == nullptr) { instance = std::make_unique<require_loop_assertion_registry>(); }
			return *instance;
		}

		void reset() { m_logged_lines.clear(); }

		bool should_log(std::string line_info) {
			auto [_, is_new] = m_logged_lines.emplace(std::move(line_info));
			return is_new;
		}

	  private:
		inline static std::unique_ptr<require_loop_assertion_registry> instance;
		std::unordered_set<std::string> m_logged_lines{};
	};

#define CELERITY_DETAIL_REQUIRE_LOOP(...)                                                                                                                      \
	if(celerity::test_utils::require_loop_assertion_registry::get_instance().should_log(std::string(__FILE__) + std::to_string(__LINE__))) {                   \
		REQUIRE(__VA_ARGS__);                                                                                                                                  \
	} else if(!(__VA_ARGS__)) {                                                                                                                                \
		REQUIRE(__VA_ARGS__);                                                                                                                                  \
	}

	template <int Dims, typename F>
	void for_each_in_range(sycl::range<Dims> range, sycl::id<Dims> offset, F&& f) {
		const auto range3 = detail::range_cast<3>(range);
		sycl::id<3> index;
		for(index[0] = 0; index[0] < range3[0]; ++index[0]) {
			for(index[1] = 0; index[1] < range3[1]; ++index[1]) {
				for(index[2] = 0; index[2] < range3[2]; ++index[2]) {
					f(offset + detail::id_cast<Dims>(index));
				}
			}
		}
	}

	template <int Dims, typename F>
	void for_each_in_range(sycl::range<Dims> range, F&& f) {
		for_each_in_range(range, {}, f);
	}

	class mock_buffer_factory;
	class mock_host_object_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			(void)detail::add_requirement(cgh, m_id, std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, m_size));
		}

		detail::buffer_id get_id() const { return m_id; }

		range<Dims> get_range() const { return m_size; }

	  private:
		friend class mock_buffer_factory;
		friend class dist_cdag_test_context;

		detail::buffer_id m_id;
		cl::sycl::range<Dims> m_size;

		mock_buffer(detail::buffer_id id, cl::sycl::range<Dims> size) : m_id(id), m_size(size) {}
	};

	class mock_host_object {
	  public:
		void add_side_effect(handler& cgh, const experimental::side_effect_order order) { (void)detail::add_requirement(cgh, m_id, order); }

		detail::host_object_id get_id() const { return m_id; }

	  private:
		friend class mock_host_object_factory;
		friend class dist_cdag_test_context;

		detail::host_object_id m_id;

	  public:
		explicit mock_host_object(detail::host_object_id id) : m_id(id) {}
	};

	class cdag_inspector {
	  public:
		auto get_cb() {
			return [](detail::node_id nid, detail::command_pkg frame) {
				// FIXME: Serializer no longer produces a frame
				// #ifndef NDEBUG
				// 				for(const auto dcid : frame->iter_dependencies()) {
				// 					// Sanity check: All dependencies must have already been flushed
				// 					assert(m_commands.count(dcid) == 1);
				// 				}
				// #endif

				// 				const detail::command_id cid = frame->pkg.cid;
				// 				m_commands[cid] = {nid, frame->pkg, std::vector(frame->iter_dependencies().begin(), frame->iter_dependencies().end())};
				// 				if(const auto tid = frame->pkg.get_tid()) { m_by_task[*tid].insert(cid); }
				// 				m_by_node[nid].insert(cid);
			};
		}

		std::set<detail::command_id> get_commands(
		    std::optional<detail::task_id> tid, std::optional<detail::node_id> nid, std::optional<detail::command_type> cmd) const {
			// Sanity check: Not all commands have an associated task id
			assert(tid == std::nullopt
			       || (cmd == std::nullopt || cmd == detail::command_type::execution || cmd == detail::command_type::horizon
			           || cmd == detail::command_type::epoch));

			std::set<detail::command_id> result;
			std::transform(m_commands.cbegin(), m_commands.cend(), std::inserter(result, result.begin()), [](auto p) { return p.first; });

			if(tid != std::nullopt) {
				auto& task_set = m_by_task.at(*tid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), task_set.cbegin(), task_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(nid != std::nullopt) {
				auto& node_set = m_by_node.at(*nid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), node_set.cbegin(), node_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(cmd != std::nullopt) {
				std::set<detail::command_id> new_result;
				std::copy_if(result.cbegin(), result.cend(), std::inserter(new_result, new_result.begin()),
				    [this, cmd](detail::command_id cid) { return m_commands.at(cid).pkg.get_command_type() == cmd; });
				result = std::move(new_result);
			}

			return result;
		}

		bool has_dependency(detail::command_id dependent, detail::command_id dependency) const {
			const auto& deps = m_commands.at(dependent).dependencies;
			return std::find(deps.cbegin(), deps.cend(), dependency) != deps.cend();
		}

		size_t get_dependency_count(detail::command_id dependent) const { return m_commands.at(dependent).dependencies.size(); }

		std::vector<detail::command_id> get_dependencies(detail::command_id dependent) const { return m_commands.at(dependent).dependencies; }

	  private:
		struct cmd_info {
			detail::node_id nid;
			detail::command_pkg pkg;
			std::vector<detail::command_id> dependencies;
		};

		std::map<detail::command_id, cmd_info> m_commands;
		std::map<detail::task_id, std::set<detail::command_id>> m_by_task;
		std::map<experimental::bench::detail::node_id, std::set<detail::command_id>> m_by_node;
	};

	class cdag_test_context {
	  public:
		cdag_test_context(size_t num_nodes) {
			m_tm = std::make_unique<detail::task_manager>(1 /* num_nodes */, nullptr /* host_queue */);
			m_cdag = std::make_unique<detail::command_graph>();
			m_ggen = std::make_unique<detail::graph_generator>(num_nodes, *m_cdag);
			m_gser = std::make_unique<detail::graph_serializer>(*m_cdag, m_inspector.get_cb());
			this->m_num_nodes = num_nodes;
		}

		detail::task_manager& get_task_manager() { return *m_tm; }
		detail::command_graph& get_command_graph() { return *m_cdag; }
		detail::graph_generator& get_graph_generator() { return *m_ggen; }
		cdag_inspector& get_inspector() { return m_inspector; }
		detail::graph_serializer& get_graph_serializer() { return *m_gser; }

		detail::task_id build_task_horizons() {
			const auto most_recently_generated_task_horizon = detail::task_manager_testspy::get_current_horizon(get_task_manager());
			if(most_recently_generated_task_horizon != m_most_recently_built_task_horizon) {
				m_most_recently_built_task_horizon = most_recently_generated_task_horizon;
				if(m_most_recently_built_task_horizon) {
					// naive_split does not really do anything for horizons, but this mirrors the behavior of scheduler::schedule exactly.
					detail::naive_split_transformer naive_split(m_num_nodes, m_num_nodes);
					get_graph_generator().build_task(*m_tm->get_task(*m_most_recently_built_task_horizon), {&naive_split});
					return *m_most_recently_built_task_horizon;
				}
			}
			return 0;
		}

	  private:
		std::unique_ptr<detail::task_manager> m_tm;
		std::unique_ptr<detail::command_graph> m_cdag;
		std::unique_ptr<detail::graph_generator> m_ggen;
		cdag_inspector m_inspector;
		std::unique_ptr<detail::graph_serializer> m_gser;
		size_t m_num_nodes;
		std::optional<detail::task_id> m_most_recently_built_task_horizon;
	};

	class mock_buffer_factory {
	  public:
		explicit mock_buffer_factory() = default;
		explicit mock_buffer_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::graph_generator& ggen) : m_task_mngr(&tm), m_ggen(&ggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::abstract_scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}
		explicit mock_buffer_factory(cdag_test_context& ctx) : m_task_mngr(&ctx.get_task_manager()), m_ggen(&ctx.get_graph_generator()) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = m_next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			if(m_task_mngr != nullptr) { m_task_mngr->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_schdlr != nullptr) { m_schdlr->notify_buffer_registered(bid, detail::range_cast<3>(size), Dims); }
			if(m_ggen != nullptr) { m_ggen->add_buffer(bid, detail::range_cast<3>(size)); }
			return buf;
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::abstract_scheduler* m_schdlr = nullptr;
		detail::graph_generator* m_ggen = nullptr;
		detail::buffer_id m_next_buffer_id = 0;
	};

	class mock_host_object_factory {
	  public:
		mock_host_object create_host_object() { return mock_host_object{m_next_id++}; }

	  private:
		detail::host_object_id m_next_id = 0;
	};

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(
	    detail::task_manager& tm, CGF cgf, cl::sycl::range<KernelDims> global_size = {1, 1}, cl::sycl::id<KernelDims> global_offset = {}) {
		return tm.submit_command_group([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](cl::sycl::id<KernelDims>) {});
		});
	}

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_nd_range_compute_task(detail::task_manager& tm, CGF cgf, celerity::nd_range<KernelDims> execution_range = {{1, 1}, {1, 1}}) {
		return tm.submit_command_group([&, er = execution_range](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(er, [](nd_item<KernelDims>) {});
		});
	}

	template <typename Spec, typename CGF>
	detail::task_id add_host_task(detail::task_manager& tm, Spec spec, CGF cgf) {
		return tm.submit_command_group([&](handler& cgh) {
			cgf(cgh);
			cgh.host_task(spec, [](auto...) {});
		});
	}

	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, size_t num_chunks, detail::task_id tid) {
		detail::naive_split_transformer transformer{num_chunks, num_nodes};
		ctx.get_graph_generator().build_task(*ctx.get_task_manager().get_task(tid), {&transformer});
		// ctx.get_graph_serializer().flush(tid); // NOCOMMIT
		// if(const auto htid = ctx.build_task_horizons()) { ctx.get_graph_serializer().flush(htid); }
		return tid;
	}

	// Defaults to the same number of chunks as nodes
	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, detail::task_id tid) {
		return build_and_flush(ctx, num_nodes, num_nodes, tid);
	}

	// Defaults to one node and chunk
	inline detail::task_id build_and_flush(cdag_test_context& ctx, detail::task_id tid) { return build_and_flush(ctx, 1, 1, tid); }

	class mock_reduction_factory {
	  public:
		detail::reduction_info create_reduction(const detail::buffer_id bid, const bool include_current_buffer_value) {
			return detail::reduction_info{m_next_id++, bid, include_current_buffer_value};
		}

	  private:
		detail::reduction_id m_next_id = 1;
	};

	template <int Dims>
	void add_reduction(handler& cgh, mock_reduction_factory& mrf, const mock_buffer<Dims>& vars, bool include_current_buffer_value) {
		detail::add_reduction(cgh, mrf.create_reduction(vars.get_id(), include_current_buffer_value));
	}

	// This fixture (or a subclass) must be used by all tests that transitively use MPI.
	class mpi_fixture {
	  public:
		mpi_fixture() { detail::runtime::test_require_mpi(); }

		mpi_fixture(const mpi_fixture&) = delete;
		mpi_fixture& operator=(const mpi_fixture&) = delete;
	};

	// This fixture (or a subclass) must be used by all tests that transitively instantiate the runtime.
	class runtime_fixture : public mpi_fixture {
	  public:
		runtime_fixture() { detail::runtime::test_case_enter(); }

		runtime_fixture(const runtime_fixture&) = delete;
		runtime_fixture& operator=(const runtime_fixture&) = delete;

		~runtime_fixture() {
			if(!detail::runtime::test_runtime_was_instantiated()) { WARN("Test specified a runtime_fixture, but did not end up instantiating the runtime"); }
			detail::runtime::test_case_exit();
		}
	};

	class device_queue_fixture : public mpi_fixture { // mpi_fixture for config
	  public:
		~device_queue_fixture() { get_device_queue().get_sycl_queue().wait_and_throw(); }

		detail::device_queue& get_device_queue() {
			if(!m_dq) {
				m_cfg = std::make_unique<detail::config>(nullptr, nullptr);
				m_dq = std::make_unique<detail::device_queue>(0, 1);
				m_dq->init(*m_cfg, detail::auto_select_device{});
			}
			return *m_dq;
		}

	  private:
		std::unique_ptr<detail::config> m_cfg;
		std::unique_ptr<detail::device_queue> m_dq;
	};

	// Printing of graphs can be enabled using the "--print-graphs" command line flag
	inline bool print_graphs = false;

	inline void maybe_print_graph(celerity::detail::task_manager& tm) {
		if(print_graphs) {
			const auto graph_str = tm.print_graph(std::numeric_limits<size_t>::max());
			assert(graph_str.has_value());
			CELERITY_INFO("Task graph:\n\n{}\n", *graph_str);
		}
	}

	inline void maybe_print_graph(celerity::detail::command_graph& cdag, const celerity::detail::task_manager& tm) {
		if(print_graphs) {
			const auto graph_str = cdag.print_graph(0, std::numeric_limits<size_t>::max(), tm, {});
			assert(graph_str.has_value());
			CELERITY_INFO("Command graph:\n\n{}\n", *graph_str);
		}
	}

	inline void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx) {
		if(print_graphs) {
			maybe_print_graph(ctx.get_task_manager());
			maybe_print_graph(ctx.get_command_graph(), ctx.get_task_manager());
		}
	}

	class set_test_env {
	  public:
#ifdef _WIN32
		set_test_env(const std::string& env, const std::string& val) : m_env_var_name(env) {
			//  We use the ANSI version of Get/Set, because it does not require type conversion of char to wchar_t, and we can safely do this
			//  because we are not mutating the text and therefore can treat them as raw bytes without having to worry about the text encoding.
			const auto name_size = GetEnvironmentVariableA(env.c_str(), nullptr, 0);
			if(name_size > 0) {
				m_original_value.resize(name_size);
				const auto res = GetEnvironmentVariableA(env.c_str(), m_original_value.data(), name_size);
				assert(res != 0 && "Failed to get celerity environment variable");
			}
			const auto res = SetEnvironmentVariableA(env.c_str(), val.c_str());
			assert(res != 0 && "Failed to set celerity environment variable");
		}

		~set_test_env() {
			if(m_original_value.empty()) {
				const auto res = SetEnvironmentVariableA(m_env_var_name.c_str(), NULL);
				assert(res != 0 && "Failed to delete celerity environment variable");
			} else {
				const auto res = SetEnvironmentVariableA(m_env_var_name.c_str(), m_original_value.c_str());
				assert(res != 0 && "Failed to reset celerity environment variable");
			}
		}

#else
		set_test_env(const std::string& env, const std::string& val) {
			const char* has_value = std::getenv(env.c_str());
			if(has_value != nullptr) { m_original_value = has_value; }
			const auto res = setenv(env.c_str(), val.c_str(), 1);
			assert(res == 0 && "Failed to set celerity environment variable");
			m_env_var_name = env;
		}
		~set_test_env() {
			if(m_original_value.empty()) {
				const auto res = unsetenv(m_env_var_name.c_str());
				assert(res == 0 && "Failed to unset celerity environment variable");
			} else {
				const auto res = setenv(m_env_var_name.c_str(), m_original_value.c_str(), 1);
				assert(res == 0 && "Failed to reset celerity environment variable");
			}
		}
#endif
	  private:
		std::string m_env_var_name;
		std::string m_original_value;
	};

	class dist_cdag_test_context;

	class task_builder {
		friend class dist_cdag_test_context;

		using action = std::function<void(handler&)>;

		class step {
		  public:
			step(dist_cdag_test_context& dctx, std::deque<action> actions) : m_dctx(dctx), m_actions(std::move(actions)) {}
			virtual ~step() noexcept(false);

			detail::task_id submit();

			step(const step&) = delete;

		  private:
			dist_cdag_test_context& m_dctx;
			std::deque<action> m_actions;

		  protected:
			template <typename StepT>
			StepT chain(action a) {
				static_assert(std::is_base_of_v<step, StepT>);
				m_actions.push_front(a);
				return StepT{m_dctx, std::move(m_actions)};
			}
		};

		class buffer_access_step : public step {
		  public:
			buffer_access_step(dist_cdag_test_context& dctx, std::deque<action> actions) : step(dctx, std::move(actions)) {}

			buffer_access_step(const buffer_access_step&) = delete;

			template <typename BufferT, typename RangeMapper>
			buffer_access_step read(BufferT& buf, RangeMapper rmfn) {
				return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read>(cgh, rmfn); });
			}

			template <typename BufferT, typename RangeMapper>
			buffer_access_step read_write(BufferT& buf, RangeMapper rmfn) {
				return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read_write>(cgh, rmfn); });
			}

			template <typename BufferT, typename RangeMapper>
			buffer_access_step discard_write(BufferT& buf, RangeMapper rmfn) {
				return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
			}

			// FIXME: Misnomer (not a "buffer access")
			template <typename HostObjT>
			buffer_access_step affect(HostObjT& host_obj) {
				return chain<buffer_access_step>([&host_obj](handler& cgh) { host_obj.add_side_effect(cgh, experimental::side_effect_order::sequential); });
			}

			// FIXME: Misnomer (not a "buffer access")
			template <typename Hint>
			buffer_access_step hint(Hint hint) {
				return chain<buffer_access_step>([&hint](handler& cgh) { experimental::hint(cgh, hint); });
			}
		};

	  public:
		template <typename Name, int Dims>
		buffer_access_step device_compute(const range<Dims>& global_size, const id<Dims>& global_offset) {
			std::deque<action> actions;
			actions.push_front([global_size, global_offset](handler& cgh) { cgh.parallel_for<Name>(global_size, global_offset, [](id<Dims>) {}); });
			return buffer_access_step(m_dctx, std::move(actions));
		}

		template <typename Name, int Dims>
		buffer_access_step device_compute(const nd_range<Dims>& nd_range) {
			std::deque<action> actions;
			actions.push_front([nd_range](handler& cgh) { cgh.parallel_for<Name>(nd_range, [](nd_item<Dims>) {}); });
			return buffer_access_step(m_dctx, std::move(actions));
		}

		template <int Dims>
		buffer_access_step host_task(const range<Dims>& global_size) {
			std::deque<action> actions;
			actions.push_front([global_size](handler& cgh) { cgh.host_task(global_size, [](partition<Dims>) {}); });
			return buffer_access_step(m_dctx, std::move(actions));
		}

	  private:
		dist_cdag_test_context& m_dctx;

		task_builder(dist_cdag_test_context& dctx) : m_dctx(dctx) {}
	};

	class command_query {
		friend class dist_cdag_test_context;

		class query_exception : public std::runtime_error {
			using std::runtime_error::runtime_error;
		};

	  public:
		// TODO Other ideas:
		// - remove(filters...)						=> Remove all commands matching the filters
		// - executes(global_size)					=> Check that commands execute a given global size (exactly? at least? ...)
		// - writes(buffer_id, subrange) 			=> Check that commands write a given buffer subrange (exactly? at least? ...)
		// - find_one(filters...)					=> Throws if result set contains more than 1 (per node?)

		template <typename... Filters>
		command_query find_all(Filters... filters) const {
			static_assert(((std::is_same_v<detail::node_id, Filters> || std::is_same_v<detail::task_id, Filters>
			                  || std::is_same_v<detail::command_type, Filters> || std::is_same_v<detail::command_id, Filters>)&&...),
			    "Unsupported filter");

			const auto node_filter = get_optional<detail::node_id>(filters...);
			const auto task_filter = get_optional<detail::task_id>(filters...);
			const auto type_filter = get_optional<detail::command_type>(filters...);
			// Note that command ids are not unique across nodes!
			const auto id_filter = get_optional<detail::command_id>(filters...);

			std::vector<std::unordered_set<const detail::abstract_command*>> filtered(m_commands_by_node.size());
			for(detail::node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
				if(node_filter.has_value() && *node_filter != nid) continue;
				for(const auto* cmd : m_commands_by_node[nid]) {
					if(task_filter.has_value()) {
						if(!detail::isa<detail::task_command>(cmd)) continue;
						if(static_cast<const detail::task_command*>(cmd)->get_tid() != *task_filter) continue;
					}
					if(type_filter.has_value()) {
						if(get_type(cmd) != *type_filter) continue;
					}
					if(id_filter.has_value()) {
						if(cmd->get_cid() != id_filter) continue;
					}
					filtered[nid].insert(cmd);
				}
			}

			return command_query{std::move(filtered)};
		}

		template <typename... Filters>
		command_query find_predecessors(Filters... filters) const {
			return find_adjacent(true, filters...);
		}

		template <typename... Filters>
		command_query find_successors(Filters... filters) const {
			return find_adjacent(false, filters...);
		}

		size_t count() const {
			return std::accumulate(
			    m_commands_by_node.begin(), m_commands_by_node.end(), size_t(0), [](size_t current, auto& cmds) { return current + cmds.size(); });
		}

		bool empty() const { return count() == 0; }

		command_query subtract(const command_query& other) const {
			assert(m_commands_by_node.size() == other.m_commands_by_node.size());
			std::vector<std::unordered_set<const detail::abstract_command*>> result(m_commands_by_node.size());
			for(detail::node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
				std::copy_if(m_commands_by_node[nid].cbegin(), m_commands_by_node[nid].cend(), std::inserter(result[nid], result[nid].begin()),
				    [&other, nid](const detail::abstract_command* cmd) { return other.m_commands_by_node[nid].count(cmd) == 0; });
			}
			return command_query{std::move(result)};
		}

		// Call the provided function once for each node, with a subquery containing commands only for that node.
		template <typename PerNodeCallback>
		void for_each_node(PerNodeCallback&& cb) const {
			for(detail::node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
				UNSCOPED_INFO(fmt::format("On node {}", nid));
				cb(find_all(nid));
			}
		}

		// Call the provided function once for each command, with a subquery only containing that command.
		template <typename PerCmdCallback>
		void for_each_command(PerCmdCallback&& cb) const {
			for(detail::node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
				for(auto* cmd : m_commands_by_node[nid]) {
					UNSCOPED_INFO(fmt::format("Command {} on node {}", nid, cmd->get_cid()));
					// We also need to filter by node here, as command ids are not globally unique!
					cb(find_all(nid, cmd->get_cid()));
				}
			}
		}

		// TODO: Use plural 'have_type'? Have both but singular throws if count > 1?
		bool has_type(const detail::command_type expected) const {
			return for_all_commands([expected](const detail::node_id nid, const detail::abstract_command* cmd) {
				const auto received = get_type(cmd);
				if(received != expected) {
					UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have type '{}' but found type '{}'", cmd->get_nid(), nid,
					    get_type_name(expected), get_type_name(received)));
					return false;
				}
				return true;
			});
		}

		bool has_successor(const command_query& successors, const std::optional<detail::dependency_kind>& kind = std::nullopt) const {
			return for_all_commands([&successors, &kind](const detail::node_id nid, const detail::abstract_command* cmd) {
				for(const auto* expected : successors.m_commands_by_node[nid]) {
					bool found = false;
					for(const auto received : cmd->get_dependents()) {
						if(received.node == expected) {
							found = true;
							if(kind.has_value() && received.kind != *kind) {
								UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {} with kind {}, but found kind {}", cmd->get_cid(),
								    nid, expected->get_cid(), *kind, received.kind));
								return false;
							}
						}
					}
					if(!found) {
						UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {}", cmd->get_cid(), nid, expected->get_cid()));
						return false;
					}
				}
				return true;
			});
		}

		std::vector<const detail::abstract_command*> get_raw(const detail::node_id nid) const {
			std::vector<const detail::abstract_command*> result;
			std::copy(m_commands_by_node.at(nid).cbegin(), m_commands_by_node.at(nid).cend(), std::back_inserter(result));
			return result;
		}

	  private:
		std::vector<std::unordered_set<const detail::abstract_command*>> m_commands_by_node;

		// Constructor for initial top-level query (containing all commands)
		command_query(const std::vector<std::unique_ptr<detail::command_graph>>& cdags) {
			for(auto& cdag : cdags) {
				m_commands_by_node.push_back({cdag->all_commands().begin(), cdag->all_commands().end()});
			}
		}

		// Constructor for narrowed-down queries
		command_query(std::vector<std::unordered_set<const detail::abstract_command*>> commands_by_node) : m_commands_by_node(std::move(commands_by_node)) {}

		template <typename Callback>
		bool for_all_commands(Callback&& cb) const {
			bool cont = true;
			for(detail::node_id nid = 0; cont && nid < m_commands_by_node.size(); ++nid) {
				for(const auto* cmd : m_commands_by_node[nid]) {
					if constexpr(std::is_invocable_r_v<bool, Callback, detail::node_id, decltype(cmd)>) {
						cont &= cb(nid, cmd);
					} else {
						cb(nid, cmd);
					}
					if(!cont) break;
				}
			}
			return cont;
		}

		template <typename... Filters>
		command_query find_adjacent(const bool find_predecessors, Filters... filters) const {
			const auto kind_filter = get_optional<detail::dependency_kind>(filters...);

			std::vector<std::unordered_set<const detail::abstract_command*>> adjacent(m_commands_by_node.size());
			for_all_commands([&adjacent, find_predecessors, kind_filter](const detail::node_id nid, const detail::abstract_command* cmd) {
				const auto iterable = find_predecessors ? cmd->get_dependencies() : cmd->get_dependents();
				for(auto it = iterable.begin(); it != iterable.end(); ++it) {
					if(kind_filter.has_value() && it->kind != *kind_filter) continue;
					adjacent[nid].insert(it->node);
				}
			});

			const auto query = command_query{std::move(adjacent)};
			// Filter resulting set of commands, but remove dependency_kind filter (if present)
			// TODO: Refactor into generic utility
			const auto filters_tuple = std::tuple{filters...};
			constexpr auto idx = get_index_of<detail::dependency_kind>(filters_tuple);
			if constexpr(idx != -1) {
				const auto filters_without_kind = tuple_splice<size_t(idx), 1>(filters_tuple);
				return std::apply([&query](auto... fs) { return query.find_all(fs...); }, filters_without_kind);
			} else {
				return query.find_all(filters...);
			}
		}

		template <typename T, typename... Ts>
		static constexpr std::optional<T> get_optional(const std::tuple<Ts...>& tuple) {
			if constexpr((std::is_same_v<T, Ts> || ...)) { return std::get<T>(tuple); }
			return std::nullopt;
		}

		template <typename T, typename... Ts>
		static constexpr std::optional<T> get_optional(Ts... ts) {
			return get_optional<T>(std::tuple(ts...));
		}

		// TODO: Move to utils header?

		template <typename T, size_t I = 0, typename Tuple>
		static constexpr int64_t get_index_of(const Tuple& t) {
			if constexpr(I >= std::tuple_size_v<Tuple>) {
				return -1;
			} else if(std::is_same_v<T, std::tuple_element_t<I, Tuple>>) {
				return I;
			} else {
				return get_index_of<T, I + 1>(t);
			}
		}

		template <size_t Offset, size_t Count, size_t... Prefix, size_t... Suffix, typename Tuple>
		static constexpr auto tuple_splice_impl(std::index_sequence<Prefix...>, std::index_sequence<Suffix...>, const Tuple& t) {
			return std::tuple_cat(std::tuple{std::get<Prefix>(t)...}, std::tuple{std::get<Offset + Count + Suffix>(t)...});
		}

		template <size_t Offset, size_t Count, typename Tuple>
		static constexpr auto tuple_splice(const Tuple& t) {
			constexpr size_t N = std::tuple_size_v<Tuple>;
			static_assert(Offset + Count <= N);
			return tuple_splice_impl<Offset, Count>(std::make_index_sequence<Offset>{}, std::make_index_sequence<N - Count - Offset>{}, t);
		}

		static detail::command_type get_type(const detail::abstract_command* cmd) {
			if(detail::isa<detail::epoch_command>(cmd)) return detail::command_type::epoch;
			if(detail::isa<detail::horizon_command>(cmd)) return detail::command_type::horizon;
			if(detail::isa<detail::execution_command>(cmd)) return detail::command_type::execution;
			if(detail::isa<detail::data_request_command>(cmd)) return detail::command_type::data_request;
			if(detail::isa<detail::push_command>(cmd)) return detail::command_type::push;
			if(detail::isa<detail::await_push_command>(cmd)) return detail::command_type::await_push;
			if(detail::isa<detail::reduction_command>(cmd)) return detail::command_type::reduction;
			throw query_exception("Unknown command type");
		}

		static std::string get_type_name(const detail::command_type type) {
			switch(type) {
			case detail::command_type::epoch: return "epoch";
			case detail::command_type::horizon: return "horizon";
			case detail::command_type::execution: return "execution";
			case detail::command_type::data_request: return "data_request";
			case detail::command_type::push: return "push";
			case detail::command_type::await_push: return "await_push";
			case detail::command_type::reduction: return "reduction";
			default: return "<unknown>";
			}
		}
	};

	class dist_cdag_test_context {
		friend class task_builder;

	  public:
		dist_cdag_test_context(size_t num_nodes, size_t devices_per_node = 1) : m_num_nodes(num_nodes) {
			m_rm = std::make_unique<detail::reduction_manager>();
			m_tm = std::make_unique<detail::task_manager>(num_nodes, nullptr /* host_queue */);
			// m_gser = std::make_unique<graph_serializer>(*m_cdag, m_inspector.get_cb());
			for(detail::node_id nid = 0; nid < num_nodes; ++nid) {
				m_cdags.emplace_back(std::make_unique<detail::command_graph>());
				m_dggens.emplace_back(std::make_unique<detail::distributed_graph_generator>(num_nodes, devices_per_node, nid, *m_cdags[nid], *m_tm));
			}
		}

		~dist_cdag_test_context() { maybe_print_graphs(); }

		template <int Dims>
		test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = m_next_buffer_id++;
			const auto buf = test_utils::mock_buffer<Dims>(bid, size);
			m_tm->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized);
			for(auto& dggen : m_dggens) {
				dggen->add_buffer(bid, detail::range_cast<3>(size), Dims);
			}
			return buf;
		}

		test_utils::mock_host_object create_host_object() { return test_utils::mock_host_object{m_next_host_object_id++}; }

		template <typename Name = detail::unnamed_kernel, int Dims>
		auto device_compute(const range<Dims>& global_size, const id<Dims>& global_offset = {}) {
			return task_builder(*this).device_compute<Name>(global_size, global_offset);
		}

		template <typename Name = detail::unnamed_kernel, int Dims>
		auto device_compute(const nd_range<Dims>& nd_range) {
			return task_builder(*this).device_compute<Name>(nd_range);
		}

		template <int Dims>
		auto host_task(const range<Dims>& global_size) {
			return task_builder(*this).host_task(global_size);
		}

		command_query query() { return command_query(m_cdags); }

		void set_horizon_step(const int step) { m_tm->set_horizon_step(step); }

		detail::distributed_graph_generator& get_graph_generator(detail::node_id nid) { return *m_dggens.at(nid); }

		detail::task_manager& get_task_manager() { return *m_tm; }

	  private:
		size_t m_num_nodes;
		detail::buffer_id m_next_buffer_id = 0;
		detail::host_object_id m_next_host_object_id = 0;
		std::optional<detail::task_id> m_most_recently_built_horizon;
		std::unique_ptr<detail::reduction_manager> m_rm;
		std::unique_ptr<detail::task_manager> m_tm;
		std::vector<std::unique_ptr<detail::command_graph>> m_cdags;
		std::vector<std::unique_ptr<detail::distributed_graph_generator>> m_dggens;

		void build_task(const detail::task_id tid) {
			for(auto& dggen : m_dggens) {
				dggen->build_task(*m_tm->get_task(tid));
			}
		}

		void maybe_build_horizon() {
			const auto current_horizon = detail::task_manager_testspy::get_current_horizon(*m_tm);
			if(m_most_recently_built_horizon != current_horizon) {
				assert(current_horizon.has_value());
				build_task(*current_horizon);
			}
			m_most_recently_built_horizon = current_horizon;
		}

		void maybe_print_graphs() {
			if(test_utils::print_graphs) {
				test_utils::maybe_print_graph(*m_tm);

				std::vector<std::string> graphs;
				for(detail::node_id nid = 0; nid < m_num_nodes; ++nid) {
					const auto& cdag = m_cdags[nid];
					const auto graph = cdag->print_graph(nid, std::numeric_limits<size_t>::max(), *m_tm, nullptr);
					assert(graph.has_value());
					graphs.push_back(*graph);
				}
				CELERITY_INFO("Command graph:\n\n{}\n", detail::combine_command_graphs(graphs));
			}
		}
	};

	inline task_builder::step::~step() noexcept(false) {
		if(!m_actions.empty()) { throw std::runtime_error("Found incomplete task build. Did you forget to call submit()?"); }
	}

	inline detail::task_id task_builder::step::submit() {
		assert(!m_actions.empty());
		const auto tid = m_dctx.get_task_manager().submit_command_group([this](handler& cgh) {
			while(!m_actions.empty()) {
				auto a = m_actions.front();
				a(cgh);
				m_actions.pop_front();
			}
		});
		m_dctx.build_task(tid);
		m_dctx.maybe_build_horizon();
		m_actions.clear();
		return tid;
	}


} // namespace test_utils
} // namespace celerity


namespace Catch {

template <int Dims>
struct StringMaker<cl::sycl::id<Dims>> {
	static std::string convert(const cl::sycl::id<Dims>& value) {
		switch(Dims) {
		case 1: return fmt::format("{{{}}}", value[0]);
		case 2: return fmt::format("{{{}, {}}}", value[0], value[1]);
		case 3: return fmt::format("{{{}, {}, {}}}", value[0], value[1], value[2]);
		default: return {};
		}
	}
};

template <int Dims>
struct StringMaker<cl::sycl::range<Dims>> {
	static std::string convert(const cl::sycl::range<Dims>& value) {
		switch(Dims) {
		case 1: return fmt::format("{{{}}}", value[0]);
		case 2: return fmt::format("{{{}, {}}}", value[0], value[1]);
		case 3: return fmt::format("{{{}, {}, {}}}", value[0], value[1], value[2]);
		default: return {};
		}
	}
};

template <>
struct StringMaker<sycl::device> {
	static std::string convert(const sycl::device& d) {
		return fmt::format("sycl::device(vendor_id={}, name=\"{}\")", d.get_info<sycl::info::device::vendor_id>(), d.get_info<sycl::info::device::name>());
	}
};

template <>
struct StringMaker<sycl::platform> {
	static std::string convert(const sycl::platform& d) {
		return fmt::format("sycl::platform(vendor=\"{}\", name=\"{}\")", d.get_info<sycl::info::platform::vendor>(), d.get_info<sycl::info::platform::name>());
	}
};

} // namespace Catch