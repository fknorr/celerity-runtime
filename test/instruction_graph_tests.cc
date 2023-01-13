#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "command_graph.h"
#include "instruction_graph_generator.h"
#include "print_graph.h"
#include "task_ring_buffer.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("instruction graph") {
	const task_id tid_epoch(0), tid_producer(1), tid_consumer(2);

	task_ring_buffer trb;
	const task_ring_buffer::wait_callback wcb = [](const task_id) { std::abort(); };
	trb.put(trb.reserve_task_entry(wcb), task::make_epoch(tid_epoch, epoch_action::none));

	buffer_access_map am_producer;
	am_producer.add_access(
	    0, std::make_unique<range_mapper<1, celerity::access::one_to_one<0>>>(celerity::access::one_to_one(), access_mode::discard_write, range<1>(256)));
	trb.put(trb.reserve_task_entry(wcb),
	    task::make_device_compute(tid_producer, task_geometry{1, {256, 1, 1}, {}, {32, 1, 1}}, nullptr, std::move(am_producer), {}, "producer"));

	buffer_access_map am_consumer;
	am_consumer.add_access(0, std::make_unique<range_mapper<1, celerity::access::all<0, 0>>>(celerity::access::all(), access_mode::read, range<1>(256)));
	trb.put(trb.reserve_task_entry(wcb),
	    task::make_device_compute(tid_consumer, task_geometry{1, {256, 1, 1}, {}, {32, 1, 1}}, nullptr, std::move(am_consumer), {}, "consumer"));

	command_graph cdag;
	const auto* init = cdag.create<epoch_command>(node_id(0), tid_epoch, epoch_action::none);
	const auto* producer = cdag.create<execution_command>(node_id(0), tid_producer, subrange<3>({}, {128, 1, 1}));
	const auto* await = cdag.create<await_push_command>(node_id(0), buffer_id(0), transfer_id(0), subrange_to_grid_box(subrange<3>({128, 0, 0}, {256, 1, 1})));
	const auto* consumer = cdag.create<execution_command>(node_id(0), tid_consumer, subrange<3>({}, {256, 1, 1}));

	const size_t num_devices = 2;
	instruction_graph_generator iggen(trb, num_devices);
	iggen.register_buffer(0, range<3>(256, 1, 1));
	iggen.compile(*producer);
	iggen.compile(*await);
	iggen.compile(*consumer);

	fmt::print("{}\n", print_instruction_graph(iggen.get_graph()));
}
