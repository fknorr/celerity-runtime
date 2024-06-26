#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "backend/sycl_backend.h"
#include "nd_memory.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;


void* backend_alloc(backend& backend, const std::optional<device_id>& device, const size_t size, const size_t alignment) {
	return test_utils::await(device.has_value() ? backend.enqueue_device_alloc(*device, size, alignment) : backend.enqueue_host_alloc(size, alignment));
}

void backend_free(backend& backend, const std::optional<device_id>& device, void* const ptr) {
	test_utils::await(device.has_value() ? backend.enqueue_device_free(*device, ptr) : backend.enqueue_host_free(ptr));
}

void backend_copy(backend& backend, const std::optional<device_id>& source_device, const std::optional<device_id>& dest_device, const void* const source_base,
    void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) {
	if(source_device.has_value() || dest_device.has_value()) {
		auto device = source_device.has_value() ? *source_device : *dest_device;
		test_utils::await(backend.enqueue_device_copy(device, 0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
	} else {
		test_utils::await(backend.enqueue_host_copy(0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
	}
}

std::vector<sycl::device> select_devices_for_backend(sycl_backend_type type) {
	// device discovery - we need at least one to run anything and two to run device-to-peer tests
	const auto all_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	std::vector<sycl::device> backend_devices;
	std::copy_if(all_devices.begin(), all_devices.end(), std::back_inserter(backend_devices),
	    [=](const sycl::device& device) { return utils::contains(sycl_backend_enumerator{}.compatible_backends(device), type); });
	if(backend_devices.empty()) { SKIP(fmt::format("No devices available for {} backend", type)); }
	return backend_devices;
}

TEST_CASE_METHOD(test_utils::backend_fixture, "backend allocations are pattern-filled in debug builds", "[backend]") {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
	const auto backend_type = GENERATE(test_utils::from_vector(sycl_backend_enumerator{}.available_backends()));
	const auto sycl_devices = select_devices_for_backend(backend_type);
	CAPTURE(backend_type, sycl_devices);

	if(sycl_devices.empty()) { SKIP("No devices available for backend"); }
	auto backend = make_sycl_backend(backend_type, sycl_devices, false /* enable_profiling */);
	sycl::queue sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});

	constexpr size_t size = 1024;
	const std::vector<uint8_t> expected(size, sycl_backend_detail::uninitialized_memory_pattern);

	for(const auto did : std::initializer_list<std::optional<device_id>>{std::nullopt, device_id(0)}) {
		CAPTURE(did);
		const auto ptr = backend_alloc(*backend, did, 1024, 1);
		std::vector<uint8_t> contents(size);
		sycl_queue.memcpy(contents.data(), ptr, size).wait();
		CHECK(contents == expected);
		backend_free(*backend, did, ptr);
	}
#else
	SKIP("Not in a debug build");
#endif
}

struct copy_test_layout {
	box<3> source_box;
	box<3> dest_box;
	box<3> copy_box;
};

constexpr range<3> copy_test_max_range{6, 6, 6};

std::vector<copy_test_layout> generate_copy_test_layouts() {
	const std::vector<int> no_shifts = {0};
	const std::vector<int> all_shifts = {-2, 0, 2};

	std::vector<copy_test_layout> layouts;
	for(int dims = 0; dims < 3; ++dims) {
		id<3> copy_min{3, 4, 5};
		id<3> copy_max{7, 8, 9};
		for(int d = dims; d < 3; ++d) {
			copy_min[d] = 0;
			copy_max[d] = 1;
		}

		// A negative shift means the source/dest box exceeds the copy box on the left side, a positive shift means it exceeds it on the right side.
		for(const int source_shift_x : dims > 0 ? all_shifts : no_shifts) {
			for(const int dest_shift_x : dims > 0 ? all_shifts : no_shifts) {
				for(const int source_shift_y : dims > 1 ? all_shifts : no_shifts) {
					for(const int dest_shift_y : dims > 1 ? all_shifts : no_shifts) {
						for(const int source_shift_z : dims > 2 ? all_shifts : no_shifts) {
							for(const int dest_shift_z : dims > 2 ? all_shifts : no_shifts) {
								id<3> source_min = copy_min;
								id<3> source_max = copy_max;
								id<3> dest_min = copy_min;
								id<3> dest_max = copy_max;
								const int source_shift[] = {source_shift_x, source_shift_y, source_shift_z};
								const int dest_shift[] = {dest_shift_x, dest_shift_y, dest_shift_z};
								for(int d = 0; d < dims; ++d) {
									if(source_shift[d] > 0) { source_min[d] -= static_cast<size_t>(source_shift[d]); }
									if(source_shift[d] < 0) { source_max[d] += static_cast<size_t>(-source_shift[d]); }
									if(dest_shift[d] > 0) { dest_min[d] -= static_cast<size_t>(dest_shift[d]); }
									if(dest_shift[d] < 0) { dest_max[d] += static_cast<size_t>(-dest_shift[d]); }
								}
								layouts.push_back({
								    box<3>{source_min, source_max},
								    box<3>{dest_min, dest_max},
								    box<3>{copy_min, copy_max},
								});
							}
						}
					}
				}
			}
		}
	}

	return layouts;
}

TEST_CASE_METHOD(test_utils::backend_fixture, "backend::enqueue_copy works correctly on all source- and destination layouts", "[backend]") {
	const auto backend_type = GENERATE(test_utils::from_vector(sycl_backend_enumerator{}.available_backends()));
	auto sycl_devices = select_devices_for_backend(backend_type);
	CAPTURE(backend_type, sycl_devices);

	if(sycl_devices.empty()) { SKIP("No devices available for backend"); }
	auto backend = make_sycl_backend(backend_type, sycl_devices, false /* enable_profiling */);

	// device_to_itself is used for buffer resizes, and device_to_peer for coherence (if the backend supports it)
	const auto direction = GENERATE(values<std::string>({"host to device", "device to host", "device to peer", "device to itself"}));
	CAPTURE(direction);

	std::optional<device_id> source_did; // host memory if nullopt
	std::optional<device_id> dest_did;   // host memory if nullopt
	if(direction == "host to device") {
		dest_did = device_id(0);
	} else if(direction == "device to host") {
		source_did = device_id(0);
	} else if(direction == "device to itself") {
		source_did = device_id(0);
		dest_did = device_id(0);
	} else if(direction == "device to peer") {
		const auto& system = backend->get_system_info();
		if(system.devices.size() < 2) { SKIP("Not enough devices available to test peer-to-peer copy"); }
		if(system.devices[0].native_memory < first_device_memory_id || system.devices[1].native_memory < first_device_memory_id
		    || system.devices[0].native_memory == system.devices[1].native_memory) {
			SKIP("Available devices do not report disjoint, dedicated memories");
		}
		if(!system.memories[system.devices[0].native_memory].copy_peers.test(system.devices[1].native_memory)) {
			SKIP("Available devices do not support peer-to-peer copy");
		}
		source_did = device_id(0);
		dest_did = device_id(1);
	} else {
		FAIL("Unknown test type");
	}
	CAPTURE(source_did, dest_did);

	// use a helper SYCL queues to init allocations and copy between user and source/dest memories
	sycl::queue source_sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});
	sycl::queue dest_sycl_queue(sycl_devices[direction == "device to peer" ? 1 : 0], sycl::property::queue::in_order{});

	const auto source_base = backend_alloc(*backend, source_did, copy_test_max_range.size() * sizeof(int), alignof(int));
	const auto dest_base = backend_alloc(*backend, dest_did, copy_test_max_range.size() * sizeof(int), alignof(int));

	// generate the source pattern in user memory
	std::vector<int> source_template(copy_test_max_range.size());
	std::iota(source_template.begin(), source_template.end(), 1);

	// use a loop instead of GENERATE() to avoid re-instantiating the backend and re-allocating device memory on each iteration (very expensive!)
	for(const auto& [source_box, dest_box, copy_box] : generate_copy_test_layouts()) {
		CAPTURE(source_box, dest_box, copy_box);
		REQUIRE(all_true(source_box.get_range() <= copy_test_max_range));
		REQUIRE(all_true(dest_box.get_range() <= copy_test_max_range));

		// reference is nd_copy_host (tested in nd_memory_tests)
		std::vector<int> expected_dest(dest_box.get_area());
		nd_copy_host(source_template.data(), expected_dest.data(), box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

		source_sycl_queue.memcpy(source_base, source_template.data(), source_box.get_area() * sizeof(int)).wait();
		dest_sycl_queue.memset(dest_base, 0, dest_box.get_area() * sizeof(int)).wait();

		backend_copy(
		    *backend, source_did, dest_did, source_base, dest_base, box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

		std::vector<int> actual_dest(dest_box.get_area());
		dest_sycl_queue.memcpy(actual_dest.data(), dest_base, actual_dest.size() * sizeof(int)).wait();

		REQUIRE(actual_dest == expected_dest);
	}

	backend_free(*backend, source_did, source_base);
	backend_free(*backend, dest_did, dest_base);
}
