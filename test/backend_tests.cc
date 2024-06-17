#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "backend/sycl_backend.h"
#include "nd_memory.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;


void* await(async_event&& evt) {
	while(!evt.is_complete()) {}
	return evt.take_result();
}

void* backend_alloc(backend& backend, const std::optional<device_id>& device, const size_t size, const size_t alignment) {
	return await(device.has_value() ? backend.enqueue_device_alloc(*device, size, alignment) : backend.enqueue_host_alloc(size, alignment));
}

void backend_free(backend& backend, const std::optional<device_id>& device, void* const ptr) {
	await(device.has_value() ? backend.enqueue_device_free(*device, ptr) : backend.enqueue_host_free(ptr));
}

void backend_copy(backend& backend, const std::optional<device_id>& source_device, const std::optional<device_id>& dest_device, const void* const source_base,
    void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) {
	if(source_device.has_value() || dest_device.has_value()) {
		auto device = source_device.has_value() ? *source_device : *dest_device;
		await(backend.enqueue_device_copy(device, 0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
	} else {
		await(backend.enqueue_host_copy(0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
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
// TODO
#else
	SKIP("Not in a debug build");
#endif
}

TEMPLATE_TEST_CASE_METHOD_SIG(test_utils::backend_fixture_dims, "backend::enqueue_copy works correctly on all source- and destination layouts ", "[backend]",
    ((int Dims), Dims), 0, 1, 2, 3) //
{
	const auto backend_type = GENERATE(test_utils::from_vector(sycl_backend_enumerator{}.available_backends()));
	auto sycl_devices = select_devices_for_backend(backend_type);
	CAPTURE(backend_type, sycl_devices);

	if(sycl_devices.empty()) { SKIP("No devices available for backend"); }

	auto backend = make_sycl_backend(backend_type, sycl_devices);

	// device_to_itself is used for buffer resizes, and device_to_peer for coherence (if the backend supports it)
	const auto direction = GENERATE(values<std::string_view>({"host to device", "device to host", "device to peer", "device to itself"}));
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

	// generate random boundaries before and after copy range in every dimension
	int source_shift[3];
	int dest_shift[3];
	if constexpr(Dims > 0) { source_shift[0] = GENERATE(values({-2, 0, 2})), dest_shift[0] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 1) { source_shift[1] = GENERATE(values({-2, 0, 2})), dest_shift[1] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 2) { source_shift[2] = GENERATE(values({-2, 0, 2})), dest_shift[2] = GENERATE(values({-2, 0, 2})); }

	const auto copy_box = box<Dims>(test_utils::truncate_id<Dims>({2, 2, 2}), test_utils::truncate_range<Dims>({5, 6, 7}));
	CAPTURE(copy_box);

	box<Dims> source_box;
	box<Dims> dest_box;
	{
		id<Dims> source_min;
		id<Dims> source_max;
		id<Dims> dest_min;
		id<Dims> dest_max;
		for(int i = 0; i < Dims; ++i) {
			source_min[i] = std::max(0, source_shift[i]);
			source_max[i] = copy_box.get_max()[i] + std::max(0, -source_shift[i]);
			dest_min[i] = std::max(0, dest_shift[i]);
			dest_max[i] = copy_box.get_max()[i] + std::max(0, -dest_shift[i]);
		}
		source_box = box<Dims>(source_min, source_max);
		dest_box = box<Dims>(dest_min, dest_max);
	}
	CAPTURE(source_box, dest_box);

	// generate the source pattern in user memory
	std::vector<int> source_template(source_box.get_area());
	std::iota(source_template.begin(), source_template.end(), 1);

	// reference is nd_copy_host (tested in nd_memory_tests)
	std::vector<int> expected_dest(dest_box.get_area());
	copy_region_host(source_template.data(), expected_dest.data(), box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

	// use a helper SYCL queues to init allocations and copy between user and source/dest memories
	sycl::queue source_sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});
	sycl::queue dest_sycl_queue(sycl_devices[direction == "device to peer" ? 1 : 0], sycl::property::queue::in_order{});

	const auto source_base = backend_alloc(*backend, source_did, source_box.get_area() * sizeof(int), alignof(int));
	source_sycl_queue.memcpy(source_base, source_template.data(), source_template.size() * sizeof(int)).wait();

	const auto dest_base = backend_alloc(*backend, dest_did, dest_box.get_area() * sizeof(int), alignof(int));
	dest_sycl_queue.memset(dest_base, 0, dest_box.get_area() * sizeof(int)).wait();

	backend_copy(*backend, source_did, dest_did, source_base, dest_base, box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

	std::vector<int> actual_dest(dest_box.get_area());
	if(dest_did.has_value()) {
		dest_sycl_queue.memcpy(actual_dest.data(), dest_base, actual_dest.size() * sizeof(int)).wait();
	} else {
		// use explicit memcpy here because of https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1474
		memcpy(actual_dest.data(), dest_base, actual_dest.size() * sizeof(int));
	}

	CHECK(actual_dest == expected_dest);

	backend_free(*backend, source_did, source_base);
	backend_free(*backend, dest_did, dest_base);
}
