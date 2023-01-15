#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "command_graph.h"
#include "instruction_graph_generator.h"
#include "print_graph.h"
#include "task_ring_buffer.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;
namespace acc = celerity::access;


// According to Wikipedia https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
std::vector<abstract_command*> topsort(const command_graph& cdag) {
	const auto commands = cdag.all_commands();
	std::unordered_set<abstract_command*> temporary_marked;
	std::unordered_set<abstract_command*> permanent_marked;
	std::unordered_set<abstract_command*> unmarked(commands.begin(), commands.end());
	std::vector<abstract_command*> sorted(unmarked.size());
	auto sorted_front = sorted.rbegin();

	const auto visit = [&](abstract_command* const cmd, auto& visit /* to allow recursion in lambda */) {
		if(permanent_marked.count(cmd) != 0) return;
		assert(temporary_marked.count(cmd) == 0 || !"cyclic command graph");
		unmarked.erase(cmd);
		temporary_marked.insert(cmd);
		for(const auto& dep : cmd->get_dependents()) {
			visit(dep.node, visit);
		}
		temporary_marked.erase(cmd);
		permanent_marked.insert(cmd);
		*sorted_front++ = cmd;
	};

	while(!unmarked.empty()) {
		visit(*unmarked.begin(), visit);
	}
	return sorted;
}


TEST_CASE("instruction graph") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {256};
	auto buf = dctx.create_buffer(test_range);
	auto ho = dctx.create_host_object();

	dctx.device_compute<class UKN(producer)>(test_range).discard_write(buf, acc::one_to_one()).submit();
	dctx.device_compute<class UKN(consumer)>(test_range).read(buf, acc::all()).submit();
	dctx.host_task(range<1>(2)).affect(ho).submit();
	dctx.host_task(range<1>(2)).affect(ho).submit();

	const size_t num_devices = 2;
	instruction_graph_generator iggen(dctx.get_task_manager(), num_devices);
	iggen.register_buffer(buf.get_id(), range<3>(256, 1, 1)); // TODO have an idag_test_context to do this
	iggen.register_host_object(ho.get_id());
	for(const auto cmd : topsort(dctx.get_graph_generator(1).NOCOMMIT_get_cdag())) {
		iggen.compile(*cmd);
	}

	fmt::print("{}\n", print_instruction_graph(iggen.get_graph()));
}
