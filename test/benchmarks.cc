#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "intrusive_graph.h"
#include "task_manager.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

struct bench_graph_node : intrusive_graph_node<bench_graph_node> {};

// try to cover the dependency counts we'll see in practice
TEMPLATE_TEST_CASE_SIG("benchmark intrusive graph dependency handling with N nodes", "[benchmark][intrusive_graph_node]", ((int N), N), 1, 10, 100) {
	// note that bench_graph_nodes are created/destroyed *within* the BENCHMARK
	// in the first two cases while the latter 2 cases only operate on already
	// existing nodes -- this is intentional; both cases are relevant in practise

	BENCHMARK("creating nodes") {
		bench_graph_node nodes[N];
		return nodes[N - 1].get_pseudo_critical_path_length(); // trick the compiler
	};

	BENCHMARK("creating and adding dependencies") {
		bench_graph_node n0;
		bench_graph_node nodes[N];
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		}
		return n0.get_dependencies();
	};

	bench_graph_node n0;
	bench_graph_node nodes[N];
	BENCHMARK("adding and removing dependencies") {
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		}
		for(int i = 0; i < N; ++i) {
			n0.remove_dependency(&nodes[i]);
		}
		return n0.get_dependencies();
	};

	for(int i = 0; i < N; ++i) {
		n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP, dependency_origin::dataflow});
	}
	BENCHMARK("checking for dependencies") {
		int d = 0;
		for(int i = 0; i < N; ++i) {
			d += n0.has_dependency(&nodes[i]) ? 1 : 0;
		}
		return d;
	};
}


struct task_manager_benchmark_context {
	task_manager tm{1, nullptr, nullptr};
	test_utils::mock_buffer_factory mbf{&tm};

	~task_manager_benchmark_context() { tm.generate_epoch_task(celerity::detail::epoch_action::shutdown); }

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

struct graph_generator_benchmark_context {
	const size_t num_nodes;
	command_graph cdag;
	graph_serializer gsrlzr{cdag, [](node_id, command_pkg, const std::vector<command_id>&) {}};
	reduction_manager rm;
	task_manager tm{num_nodes, nullptr, &rm};
	graph_generator ggen{num_nodes, tm, rm, cdag};
	test_utils::mock_buffer_factory mbf{&tm, &ggen};

	explicit graph_generator_benchmark_context(size_t num_nodes) : num_nodes{num_nodes} {
		tm.register_task_callback([this](const task_id tid) {
			naive_split_transformer transformer{this->num_nodes, this->num_nodes};
			ggen.build_task(tid, {&transformer});
			gsrlzr.flush(tid);
		});
	}

	~graph_generator_benchmark_context() { tm.generate_epoch_task(celerity::detail::epoch_action::shutdown); }

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		// note: This ignores communication overhead with the scheduler thread
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

struct scheduler_benchmark_context {
	// note: This will include thread creation / destruction overhead in the benchmark times.
	const size_t num_nodes;
	command_graph cdag;
	graph_serializer gsrlzr{cdag, [](node_id, command_pkg, const std::vector<command_id>&) {}};
	reduction_manager rm;
	task_manager tm{num_nodes, nullptr, &rm};
	graph_generator ggen{num_nodes, tm, rm, cdag};
	scheduler schdlr{ggen, gsrlzr, num_nodes};
	test_utils::mock_buffer_factory mbf{&tm, &ggen};

	explicit scheduler_benchmark_context(size_t num_nodes) : num_nodes{num_nodes} { //
		schdlr.startup();
	}

	~scheduler_benchmark_context() {
		// scheduler operates in a FIFO manner, so awaiting shutdown will await processing of all pending tasks first
		const auto tid = tm.generate_epoch_task(celerity::detail::epoch_action::shutdown);
		schdlr.notify_task_created(tid);
		schdlr.shutdown();
	}

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		const auto tid = tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
		schdlr.notify_task_created(tid);
	}
};

template <typename BenchmarkContext>
[[gnu::noinline]] void generate_soup_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 1000;
	const range<1> buffer_range{2048};

	for(int t = 0; t < num_tasks; ++t) {
		ctx.create_task(buffer_range, [](handler& cgh) {});
	}
}

template <typename BenchmarkContext>
[[gnu::noinline]] void generate_chain_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 200;
	const range<1> buffer_range{2048};

	auto buffer = ctx.mbf.create_buffer(buffer_range);
	for(int t = 0; t < num_tasks; ++t) {
		ctx.create_task(buffer_range, [&](handler& cgh) { buffer.template get_access<access_mode::read_write>(cgh, celerity::access::one_to_one()); });
	}
}

enum class TreeTopology { Map, Reduce };

template <TreeTopology Topology, typename BenchmarkContext>
[[gnu::noinline]] void generate_tree_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 100;
	const range<1> buffer_range{2048};

	auto buffer = ctx.mbf.create_buffer(buffer_range);
	auto buffer2 = ctx.mbf.create_buffer(buffer_range);

	int numEpochs = std::log2(num_tasks);
	int curEpochTasks = Topology == TreeTopology::Map ? 1 : 1 << numEpochs;
	int sentinelEpoch = Topology == TreeTopology::Map ? numEpochs - 1 : 0;
	int sentinelEpochMax = num_tasks - (curEpochTasks - 1); // how many tasks to generate at the last/first epoch to reach exactly numTasks

	for(int e = 0; e < numEpochs; ++e) {
		int taskCount = curEpochTasks;
		if(e == sentinelEpoch) taskCount = sentinelEpochMax;

		// build tasks for this epoch
		for(int t = 0; t < taskCount; ++t) {
			ctx.create_task(range<1>{1}, [&](celerity::handler& cgh) {
				// mappers constructed to build a binary (potentially inverted) tree
				auto read_mapper = [=](const celerity::chunk<1>& chunk) {
					return Topology == TreeTopology::Map ? celerity::subrange<1>(t / 2, 1) : celerity::subrange<1>(t * 2, 2);
				};
				auto write_mapper = [=](const celerity::chunk<1>& chunk) { return celerity::subrange<1>(t, 1); };
				buffer.template get_access<access_mode::write>(cgh, write_mapper);
				buffer2.template get_access<access_mode::read>(cgh, read_mapper);
			});
		}

		// get ready for the next epoch
		if(Topology == TreeTopology::Map) {
			curEpochTasks *= 2;
		} else {
			curEpochTasks /= 2;
		}
		std::swap(buffer, buffer2);
	}
}

// graphs identical to the wave_sim example
template <typename BenchmarkContext>
[[gnu::noinline]] void generate_wave_sim_graph(BenchmarkContext&& ctx) {
	constexpr int N = 512;
	constexpr float T = 20;
	constexpr float dt = 0.25f;

	const auto fill = [&](test_utils::mock_buffer<2> u) {
		ctx.create_task(u.get_range(), [&](celerity::handler& cgh) { u.get_access<access_mode::discard_write>(cgh, celerity::access::one_to_one{}); });
	};

	const auto step = [&](test_utils::mock_buffer<2> up, test_utils::mock_buffer<2> u) {
		ctx.create_task(up.get_range(), [&](celerity::handler& cgh) {
			up.get_access<access_mode::read_write>(cgh, celerity::access::one_to_one{});
			u.get_access<access_mode::read>(cgh, celerity::access::neighborhood{1, 1});
		});
	};

	auto up = ctx.mbf.create_buffer(celerity::range<2>(N, N));
	auto u = ctx.mbf.create_buffer(celerity::range<2>(N, N));

	fill(u);
	fill(up);
	step(up, u);

	auto t = 0.0;
	size_t i = 0;
	while(t < T) {
		step(up, u);
		std::swap(u, up);
		t += dt;
	}
}

TEST_CASE("generating large task graphs", "[benchmark][task-graph]") {
	BENCHMARK("soup topology") { generate_soup_graph(task_manager_benchmark_context{}); };
	BENCHMARK("chain topology") { generate_chain_graph(task_manager_benchmark_context{}); };
	BENCHMARK("map topology") { generate_tree_graph<TreeTopology::Map>(task_manager_benchmark_context{}); };
	BENCHMARK("reduce topology") { generate_tree_graph<TreeTopology::Reduce>(task_manager_benchmark_context{}); };
	BENCHMARK("wave_sim topology") { generate_wave_sim_graph(task_manager_benchmark_context{}); };
}

TEMPLATE_TEST_CASE_SIG("generating large command graphs for N nodes", "[benchmark][command-graph]", ((size_t NumNodes), NumNodes), 1, 2, 4) {
	BENCHMARK("soup topology") { generate_soup_graph(graph_generator_benchmark_context{NumNodes}); };
	BENCHMARK("chain topology") { generate_chain_graph(graph_generator_benchmark_context{NumNodes}); };
	BENCHMARK("map topology") { generate_tree_graph<TreeTopology::Map>(graph_generator_benchmark_context{NumNodes}); };
	BENCHMARK("reduce topology") { generate_tree_graph<TreeTopology::Reduce>(graph_generator_benchmark_context{NumNodes}); };
	BENCHMARK("wave_sim topology") { generate_wave_sim_graph(graph_generator_benchmark_context{NumNodes}); };
}

TEMPLATE_TEST_CASE_SIG("processing large graphs with a scheduler thread for N nodes", "[benchmark][scheduler]", ((size_t NumNodes), NumNodes), 1, 2, 4) {
	BENCHMARK("soup topology") { generate_soup_graph(scheduler_benchmark_context{NumNodes}); };
	BENCHMARK("chain topology") { generate_chain_graph(scheduler_benchmark_context{NumNodes}); };
	BENCHMARK("map topology") { generate_tree_graph<TreeTopology::Map>(scheduler_benchmark_context{NumNodes}); };
	BENCHMARK("reduce topology") { generate_tree_graph<TreeTopology::Reduce>(scheduler_benchmark_context{NumNodes}); };
	BENCHMARK("wave_sim topology") { generate_wave_sim_graph(scheduler_benchmark_context{NumNodes}); };
}
