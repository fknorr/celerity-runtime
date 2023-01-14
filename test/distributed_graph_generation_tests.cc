#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <celerity.h>

#include "cool_region_map.h"
#include "distributed_graph_generator.h"
#include "print_graph.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;


TEST_CASE("distributed push-model hello world", "[NOCOMMIT][dist-ggen]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};

	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	// FIXME: We can't use this for writing as we cannot invert it. Need higher-level mechanism.
	const auto swap_rm = [test_range](chunk<1> chnk) { return subrange<1>{{test_range[0] - chnk.range[0] - chnk.offset[0]}, chnk.range}; };

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, swap_rm).submit();
	const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_d = dctx.device_compute<class UKN(task_d)>(test_range).read(buf0, swap_rm).submit();

	const auto cmds_b = dctx.query().find_all(tid_b);
	CHECK(cmds_b.count() == 2);
	CHECK(cmds_b.has_type(command_type::execution));

	const auto pushes_c = dctx.query().find_all(tid_a).find_successors(dependency_kind::true_dep);
	CHECK(pushes_c.count() == 2);
	CHECK(pushes_c.has_type(command_type::push));

	const auto pushes_d = dctx.query().find_all(tid_c).find_successors(dependency_kind::true_dep);
	CHECK(pushes_d.count() == 2);
	CHECK(pushes_d.has_type(command_type::push));

	const auto cmds_d = dctx.query().find_all(tid_d);
	const auto await_pushes_d = cmds_d.find_predecessors();
	CHECK(await_pushes_d.count() == 2);
	CHECK(await_pushes_d.has_type(command_type::await_push));
	CHECK(cmds_b.has_successor(await_pushes_d, dependency_kind::anti_dep));
}

TEST_CASE("don't treat replicated data as owned", "[regression][dist-ggen]") {
	dist_cdag_test_context dctx(2);

	const range<2> test_range = {128, 128};

	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::slice<2>{0}).discard_write(buf1, acc::one_to_one{}).submit();

	const auto pushes = dctx.query().find_all(command_type::push);
	CHECK(pushes.count() == 2);
	// Regression: Node 0 assumed that it owned the data it got pushed by node 1 for its chunk, so it generated a push for node 1's chunk.
	pushes.for_each_node([](const auto& q) { CHECK(q.count() == 1); });
}

// NOCOMMIT TODO: Test that same transfer isn't being generated twice!!

TEST_CASE("a single await push command can await multiple pushes", "[dist-ggen]") {
	dist_cdag_test_context dctx(3);

	const range<1> test_range = {128};

	auto buf0 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::all{}).submit();
	dctx.query().find_all(command_type::await_push).for_each_node([](const auto& q) { CHECK(q.count() == 1); });
	dctx.query().find_all(command_type::push).for_each_node([](const auto& q) { CHECK(q.count() == 2); });
}

TEST_CASE("data owners generate separate push commands for each last writer command", "[dist-ggen]") {
	// TODO: Add this test to document the current behavior. OR: Make it a !shouldfail and check for a single command?
}

// TODO: Move?
namespace celerity::detail {

// FIXME: Duplicated from graph_compaction_tests
struct region_map_testspy {
	template <typename T>
	static size_t get_num_regions(const region_map<T>& map) {
		return map.m_region_values.size();
	}
	template <typename T>
	static size_t get_num_regions(const my_cool_region_map_wrapper<T>& map) {
		switch(map.dims) {
		case 1: return std::get<1>(map.region_map).get_num_regions();
		case 2: return std::get<2>(map.region_map).get_num_regions();
		case 3: return std::get<3>(map.region_map).get_num_regions();
		};
		return -1;
	}
};

struct distributed_graph_generator_testspy {
	static size_t get_last_writer_num_regions(const distributed_graph_generator& dggen, const buffer_id bid) {
		return region_map_testspy::get_num_regions(dggen.m_buffer_states.at(bid).local_last_writer);
	}

	static size_t get_command_buffer_reads_size(const distributed_graph_generator& dggen) { return dggen.m_command_buffer_reads.size(); }
};
} // namespace celerity::detail

// This started out as a port of "horizons prevent number of regions from growing indefinitely", but was then changed (and simplified) considerably
TEST_CASE("horizons prevent tracking data structures from growing indefinitely", "[horizon][command-graph]") {
	constexpr int num_timesteps = 100;

	dist_cdag_test_context dctx(1);
	const size_t buffer_width = 300;
	auto buf_a = dctx.create_buffer(range<2>(num_timesteps, buffer_width));

	const int horizon_step_size = GENERATE(values({1, 2, 3}));
	CAPTURE(horizon_step_size);

	dctx.set_horizon_step(horizon_step_size);

	for(int t = 0; t < num_timesteps; ++t) {
		CAPTURE(t);
		const auto read_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(t, buffer_width);
			ret.offset = id<2>(0, 0);
			return ret;
		};
		const auto write_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(1, buffer_width);
			ret.offset = id<2>(t, 0);
			return ret;
		};
		dctx.device_compute<class UKN(timestep)>(range<1>(buffer_width)).read(buf_a, read_accessor).discard_write(buf_a, write_accessor).submit();

		auto& ggen = dctx.get_graph_generator(0);

		// Assert once we've reached steady state as to not overcomplicate things
		if(t > 2 * horizon_step_size) {
			const auto num_regions = distributed_graph_generator_testspy::get_last_writer_num_regions(ggen, buf_a.get_id());
			const size_t cmds_before_applied_horizon = 1;
			const size_t cmds_after_applied_horizon = horizon_step_size + ((t + 1) % horizon_step_size);
			REQUIRE_LOOP(num_regions == cmds_before_applied_horizon + cmds_after_applied_horizon);

			// Pruning happens with a one step delay after a horizon has been applied
			const size_t expected_reads = horizon_step_size + (t % horizon_step_size) + 1;
			const size_t reads_per_timestep = t > 0 ? 1 : 0;
			REQUIRE_LOOP(distributed_graph_generator_testspy::get_command_buffer_reads_size(ggen) == expected_reads);
		}

		REQUIRE_LOOP(dctx.query().find_all(command_type::horizon).count() <= 3);
	}
}

TEST_CASE("the same buffer range is not pushed twice") {
	dist_cdag_test_context dctx(2);
	auto buf1 = dctx.create_buffer(range<1>(128));

	dctx.device_compute<class UKN(task_a)>(buf1.get_range()).discard_write(buf1, acc::one_to_one{}).submit();
	dctx.device_compute<class UKN(task_b)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {32}})).submit();

	const auto pushes_b = dctx.query().find_all(command_type::push);
	CHECK(pushes_b.count() == 1);

	SECTION("when requesting the exact same range") {
		dctx.device_compute<class UKN(task_c)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {32}})).submit();
		const auto pushes_c = dctx.query().find_all(command_type::push).subtract(pushes_b);
		CHECK(pushes_c.empty());
	}

	SECTION("when requesting a partially overlapping range") {
		dctx.device_compute<class UKN(task_c)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {64}})).submit();
		const auto pushes_c = dctx.query().find_all(command_type::push).subtract(pushes_b);
		REQUIRE(pushes_c.count() == 1);
		const auto push_cmd = dynamic_cast<const push_command*>(pushes_c.get_raw(0)[0]);
		CHECK(subrange_cast<1>(push_cmd->get_range()) == subrange<1>({32}, {32}));
	}
}

// TODO: I've removed an assertion in this same commit that caused problems when determining anti-dependencies for read-write accesses.
// The difference to master-worker is that we now update the local_last_writer directly instead of using an update list when generating an await push.
// This means that the write access finds the await push command as the last writer when determining anti-dependencies, which we then simply skip.
// This should be fine as it already has a true dependency on it anyway (and the await push already has all transitive anti-dependencies it needs).
// The order of processing accesses (producer/consumer) shouldn't matter either as we do defer the update of the last writer for the actual
// execution command until all modes have been processed (i.e., we won't forget to generate the await push).
// Go through this again and see if everything works as expected (in particular with multiple chunks).
TEST_CASE("read_write access works", "[smoke-test]") {
	dist_cdag_test_context dctx(2);

	auto buf = dctx.create_buffer(range<1>(128));

	dctx.device_compute(buf.get_range()).discard_write(buf, acc::one_to_one{}).submit();
	dctx.device_compute(buf.get_range()).read_write(buf, acc::fixed<1>({{0, 64}})).submit();
}

// NOCOMMIT TODO: Test intra-task anti-dependencies

TEST_CASE("side effect dependencies") {
	dist_cdag_test_context dctx(1);
	auto hobj = dctx.create_host_object();
	const auto tid_a = dctx.host_task(range<1>(1)).affect(hobj).submit();
	const auto tid_b = dctx.host_task(range<1>(1)).affect(hobj).submit();
	CHECK(dctx.query().find_all(tid_a).has_successor(dctx.query().find_all(tid_b)));
	// NOCOMMIT TODO: Test horizon / epoch subsumption as well
}

TEST_CASE("reading from host-initialized or uninitialized buffers doesn't generate faulty await push commands") {
	const int num_nodes = GENERATE(1, 2); // (We used to generate an await push even for a single node!)
	dist_cdag_test_context dctx(num_nodes);

	const auto test_range = range<1>(128);
	auto host_init_buf = dctx.create_buffer(test_range, true);
	auto uninit_buf = dctx.create_buffer(test_range, false);
	const auto tid_a = dctx.device_compute(test_range).read(host_init_buf, acc::one_to_one{}).read(uninit_buf, acc::one_to_one{}).submit();
	CHECK(dctx.query().find_all(command_type::await_push).empty());
	CHECK(dctx.query().find_all(command_type::push).empty());
}

// Regression test
TEST_CASE("overlapping read/write access to the same buffer doesn't generate intra-task dependencies between chunks on the same worker") {
	dist_cdag_test_context dctx(1, 2);

	const auto test_range = range<1>(128);
	auto buf = dctx.create_buffer(test_range, true);
	dctx.device_compute(test_range).read(buf, acc::neighborhood<1>{1}).discard_write(buf, acc::one_to_one{}).submit();
	CHECK(dctx.query().find_all(command_type::execution).count() == 2);
	dctx.query().find_all(command_type::execution).for_each_command([](const auto& q) {
		// Both commands should only depend on initial epoch, not each other
		CHECK(q.find_predecessors().count() == 1);
	});
}

TEST_CASE("local chunks can create multiple await push commands for a single push") {
	dist_cdag_test_context dctx(2, 2);
	const auto test_range = range<1>(128);
	auto buf = dctx.create_buffer(test_range);
	const auto transpose = [](celerity::chunk<1> chnk) { return celerity::subrange<1>(chnk.global_size[0] - chnk.offset[0] - chnk.range[0], chnk.range); };

	// Since we currently create a separate push command for each last writer command, this just happens to work out.
	SECTION("this works by accident") {
		dctx.device_compute(test_range).discard_write(buf, acc::one_to_one{}).submit();
		dctx.device_compute(test_range).read(buf, transpose).submit();
		CHECK(dctx.query().find_all(command_type::push).count() == dctx.query().find_all(command_type::await_push).count());
	}

	SECTION("this is what we actually wanted") {
		// Prevent additional local chunks from being created by using nd_range
		dctx.device_compute(nd_range<1>(test_range, {64})).discard_write(buf, acc::one_to_one{}).submit();
		dctx.device_compute(test_range).read(buf, transpose).submit();

		// NOCOMMIT TODO: If would be sweet if we could somehow get the union region across all await pushes and check it against the corresponding push
		CHECK(dctx.query().find_all(node_id(0), command_type::push).count() == 1);
		CHECK(dctx.query().find_all(node_id(1), command_type::await_push).count() == 2);
		CHECK(dctx.query().find_all(node_id(1), command_type::push).count() == 1);
		CHECK(dctx.query().find_all(node_id(0), command_type::await_push).count() == 2);
	}
}

// Regression test
TEST_CASE("kernel offsets work correctly") {
	dist_cdag_test_context dctx(1, 1);
	const auto test_range = range<1>(128);
	auto buf = dctx.create_buffer(test_range);

	dctx.device_compute(test_range - range<1>(2), id<1>(1)).discard_write(buf, acc::one_to_one{}).submit();
	dctx.device_compute(range<1>(1)).read(buf, acc::all{}).submit();
	CHECK(dctx.query().find_all(command_type::await_push).count() == 0);
}

TEST_CASE("per-device 2d oversubscribed chunks cover the same region as the corresponding original chunks would have") {
	dist_cdag_test_context dctx(1, 4);
	const auto test_range = range<2>(128, 128);

	const auto tid_a = dctx.device_compute(test_range).hint(experimental::hints::tiled_split{}).submit();
	const auto tid_b = dctx.device_compute(test_range).hint(experimental::hints::tiled_split{}).hint(experimental::hints::oversubscribe{4}).submit();

	GridRegion<3> full_chunks_by_device[4];
	CHECK(dctx.query().find_all(tid_a).count() == 4);
	for(const auto* acmd : dctx.query().find_all(tid_a).get_raw(0)) {
		REQUIRE_LOOP(isa<execution_command>(acmd));
		auto& ecmd = *static_cast<const execution_command*>(acmd);
		auto& chnk = full_chunks_by_device[ecmd.get_device_id()];
		REQUIRE_LOOP(chnk.empty());
		chnk = subrange_to_grid_box(ecmd.get_execution_range());
	}

	CHECK(dctx.query().find_all(tid_b).count() == 16);
	for(const auto* acmd : dctx.query().find_all(tid_b).get_raw(0)) {
		REQUIRE_LOOP(isa<execution_command>(acmd));
		auto& ecmd = *static_cast<const execution_command*>(acmd);
		auto& chnk = full_chunks_by_device[ecmd.get_device_id()];
		REQUIRE_LOOP(!chnk.empty());
		REQUIRE_LOOP(!GridRegion<3>::intersect(chnk, subrange_to_grid_box(ecmd.get_execution_range())).empty());
		chnk = GridRegion<3>::difference(chnk, subrange_to_grid_box(ecmd.get_execution_range()));
	}

	for(size_t i = 0; i < 4; ++i) {
		REQUIRE_LOOP(full_chunks_by_device[i].empty());
	}
}