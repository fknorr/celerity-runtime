#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "backend/sycl_backend.h"
#include "nd_memory.h"

#include "copy_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

// backend_*() functions here dispatch _host / _device member functions based on whether a device id is provided or not

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

/// For extracting hydration results
template <target Target>
struct mock_accessor {
	hydration_id hid;
	std::optional<closure_hydrator::accessor_info> info;

	explicit mock_accessor(hydration_id hid) : hid(hid) {}
	mock_accessor(const mock_accessor& other) : hid(other.hid) { copy_and_hydrate(other); }
	mock_accessor(mock_accessor&&) = delete;
	mock_accessor& operator=(const mock_accessor& other) { hid = other.hid, copy_and_hydrate(other); }
	mock_accessor& operator=(mock_accessor&&) = delete;
	~mock_accessor() = default;

	void copy_and_hydrate(const mock_accessor& other) {
		if(!info.has_value() && detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating()) {
			info = detail::closure_hydrator::get_instance().get_accessor_info<Target>(hid);
		}
	}
};

std::vector<sycl::device> select_devices_for_backend(sycl_backend_type type) {
	// device discovery - we need at least one to run anything and two to run device-to-peer tests
	const auto all_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	std::vector<sycl::device> backend_devices;
	std::copy_if(all_devices.begin(), all_devices.end(), std::back_inserter(backend_devices),
	    [=](const sycl::device& device) { return utils::contains(sycl_backend_enumerator{}.compatible_backends(device), type); });
	return backend_devices;
}

std::tuple<sycl_backend_type, std::unique_ptr<backend>, std::vector<sycl::device>> generate_backends_with_devices() {
	const auto backend_type = GENERATE(test_utils::from_vector(sycl_backend_enumerator{}.available_backends()));
	auto sycl_devices = select_devices_for_backend(backend_type);
	CAPTURE(backend_type, sycl_devices);

	if(sycl_devices.empty()) { SKIP("No devices available for backend"); }
	auto backend = make_sycl_backend(backend_type, sycl_devices, false /* enable_profiling */);
	return {backend_type, std::move(backend), std::move(sycl_devices)};
}

TEST_CASE_METHOD(test_utils::backend_fixture, "debug allocations are host- and device-accessible", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const auto debug_ptr = static_cast<int*>(backend->debug_alloc(sizeof(int)));
	*debug_ptr = 1;
	sycl::queue(sycl_devices[0], sycl::property::queue::in_order{}).single_task([=]() { *debug_ptr += 1; }).wait();
	CHECK(*debug_ptr == 2);
	backend->debug_free(debug_ptr);
}

TEST_CASE_METHOD(test_utils::backend_fixture, "backend allocations are properly aligned", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	constexpr size_t size = 1024;
	constexpr size_t sycl_max_alignment = 64; // See SYCL spec 4.14.2.6

	const auto host_ptr = backend_alloc(*backend, std::nullopt, size, sycl_max_alignment);
	CHECK(reinterpret_cast<uintptr_t>(host_ptr) % sycl_max_alignment == 0);
	backend_free(*backend, std::nullopt, host_ptr);

	for(device_id did = 0; did < sycl_devices.size(); ++did) {
		CAPTURE(did);
		const auto device_ptr = backend_alloc(*backend, did, size, sycl_max_alignment);
		CHECK(reinterpret_cast<uintptr_t>(device_ptr) % sycl_max_alignment == 0);
		backend_free(*backend, did, device_ptr);
	}
}

TEST_CASE_METHOD(test_utils::backend_fixture, "backend allocations are pattern-filled in debug builds", "[backend]") {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

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

TEST_CASE_METHOD(test_utils::backend_fixture, "host task lambdas are hydrated and invoked with the correct parameters", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const mock_accessor<target::host_task> acc1(hydration_id(1));
	const mock_accessor<target::host_task> acc2(hydration_id(2));
	const std::vector<closure_hydrator::accessor_info> accessor_infos{
	    {reinterpret_cast<void*>(0x1337), box<3>{{1, 2, 3}, {4, 5, 6}},
	        box<3>{{0, 1, 2}, {7, 8, 9}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x69420))},
	    {reinterpret_cast<void*>(0x7331), box<3>{{3, 2, 1}, {6, 5, 4}},
	        box<3>{{2, 1, 0}, {9, 8, 7}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x1230))}};

	constexpr size_t lane = 0;
	const box<3> execution_range({1, 2, 3}, {4, 5, 6});
	const auto collective_comm = reinterpret_cast<const communicator*>(0x42000);

	int value = 1;

	// no accessors
	test_utils::await(backend->enqueue_host_task(
	    lane,
	    [&](const box<3>& b, const communicator* c) {
		    CHECK(b == execution_range);
		    CHECK(c == collective_comm);
		    value += 1;
	    },
	    {}, execution_range, collective_comm));

	// yes accessors
	test_utils::await(backend->enqueue_host_task(
	    lane,
	    [&, acc1, acc2](const box<3>& b, const communicator* c) {
		    CHECK(acc1.info == accessor_infos[0]);
		    CHECK(acc2.info == accessor_infos[1]);
		    CHECK(b == execution_range);
		    CHECK(c == collective_comm);
		    value += 1;
	    },
	    accessor_infos, execution_range, collective_comm));

	CHECK(value == 3);
}

TEST_CASE_METHOD(test_utils::backend_fixture, "host tasks in a single lane execute in-order", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	constexpr size_t lane = 0;
	const auto first = backend->enqueue_host_task(
	    lane, [](const box<3>&, const communicator*) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }, {}, box_cast<3>(box<0>()), nullptr);
	const auto second = backend->enqueue_host_task(
	    lane, [](const box<3>&, const communicator* /* collective_comm */) {}, {}, box_cast<3>(box<0>()), nullptr);

	for(;;) {
		if(second.is_complete()) {
			CHECK(first.is_complete());
			break;
		}
	}
}

TEST_CASE_METHOD(test_utils::backend_fixture, "device kernel command groups are hydrated and invoked with the correct parameters", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const mock_accessor<target::device> acc1(hydration_id(1));
	const mock_accessor<target::device> acc2(hydration_id(2));
	const std::vector<closure_hydrator::accessor_info> accessor_infos{
	    {reinterpret_cast<void*>(0x1337), box<3>{{1, 2, 3}, {4, 5, 6}},
	        box<3>{{0, 1, 2}, {7, 8, 9}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x69420))},
	    {reinterpret_cast<void*>(0x7331), box<3>{{3, 2, 1}, {6, 5, 4}},
	        box<3>{{2, 1, 0}, {9, 8, 7}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x1230))}};

	constexpr size_t lane = 0;
	const box<3> execution_range({1, 2, 3}, {4, 5, 6});
	const std::vector<void*> reduction_ptrs{nullptr, reinterpret_cast<void*>(1337)};

	const auto value_ptr = static_cast<int*>(backend->debug_alloc(sizeof(int)));

	for(device_id did = 0; did < sycl_devices.size(); ++did) {
		*value_ptr = 1;

		// no accessors
		test_utils::await(backend->enqueue_device_kernel(
		    did, lane,
		    [&](sycl::handler& cgh, const box<3>& b, const std::vector<void*>& r) {
			    CHECK(b == execution_range);
			    CHECK(r == reduction_ptrs);
			    cgh.single_task([=] { *value_ptr += 1; });
		    },
		    {}, execution_range, reduction_ptrs));

		// yes accessors
		test_utils::await(backend->enqueue_device_kernel(
		    did, lane,
		    [&, acc1, acc2](sycl::handler& cgh, const box<3>& b, const std::vector<void*>& r) {
			    CHECK(acc1.info == accessor_infos[0]);
			    CHECK(acc2.info == accessor_infos[1]);
			    CHECK(b == execution_range);
			    CHECK(r == reduction_ptrs);
			    cgh.single_task([=] { *value_ptr += 1; });
		    },
		    accessor_infos, execution_range, reduction_ptrs));

		CHECK(*value_ptr == 3);
	}

	backend->debug_free(value_ptr);
}

TEST_CASE_METHOD(test_utils::backend_fixture, "device kernels in a single lane execute in-order", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const auto dummy = static_cast<volatile int*>(backend->debug_alloc(sizeof(int)));
	*dummy = 0;

	constexpr size_t lane = 0;

	const auto first = backend->enqueue_device_kernel(device_id(0), lane,
	    [=](sycl::handler& cgh, const box<3>&, const std::vector<void*>&) {
		    cgh.single_task([=] {
			    while(++*dummy < 100'000) {} // busy "wait" - takes ~10ms on hipSYCL debug build with RTX 3090
		    });
	    },
	    {}, box_cast<3>(box<0>()), {});

	const auto second = backend->enqueue_device_kernel(
	    device_id(0), lane, [=](sycl::handler& cgh, const box<3>&, const std::vector<void*>&) { cgh.single_task([=] {}); }, {}, box_cast<3>(box<0>()), {});

	for(;;) {
		if(second.is_complete()) {
			CHECK(first.is_complete());
			break;
		}
	}

	backend->debug_free(const_cast<int*>(dummy));
}

TEST_CASE_METHOD(test_utils::backend_fixture, "backend copies work correctly on all source- and destination layouts", "[backend]") {
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

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

	// use a helper SYCL queue to init allocations and copy between user and source/dest memories
	sycl::queue source_sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});
	sycl::queue dest_sycl_queue(sycl_devices[direction == "device to peer" ? 1 : 0], sycl::property::queue::in_order{});

	const auto source_base = backend_alloc(*backend, source_did, test_utils::copy_test_max_range.size() * sizeof(int), alignof(int));
	const auto dest_base = backend_alloc(*backend, dest_did, test_utils::copy_test_max_range.size() * sizeof(int), alignof(int));

	// generate the source pattern in user memory
	std::vector<int> source_template(test_utils::copy_test_max_range.size());
	std::iota(source_template.begin(), source_template.end(), 1);

	// use a loop instead of GENERATE() to avoid re-instantiating the backend and re-allocating device memory on each iteration (very expensive!)
	for(const auto& [source_box, dest_box, copy_box] : test_utils::generate_copy_test_layouts()) {
		CAPTURE(source_box, dest_box, copy_box);
		REQUIRE(all_true(source_box.get_range() <= test_utils::copy_test_max_range));
		REQUIRE(all_true(dest_box.get_range() <= test_utils::copy_test_max_range));

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
