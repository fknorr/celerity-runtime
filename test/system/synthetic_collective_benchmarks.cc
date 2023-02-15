#include "../test_utils.h"

#include <algorithm>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <mpi.h>

#include <celerity.h>

#include "../log_test_utils.h"

using namespace celerity;
using namespace celerity::test_utils;

size_t operator""_i(unsigned long long v) { return v; }
size_t operator""_Ki(unsigned long long v) { return v << 10; }
size_t operator""_Mi(unsigned long long v) { return v << 20; }
size_t operator""_Gi(unsigned long long v) { return v << 30; }

template <typename F>
std::chrono::microseconds run_distributed_benchmark(size_t n_warmup, size_t n_passes, F&& f) {
	celerity::distr_queue q;
	q.slow_full_sync();

	std::chrono::microseconds time{};
	for(size_t i = 0; i < n_warmup + n_passes; ++i) {
		const auto start = std::chrono::steady_clock::now();
		f(q);
		q.slow_full_sync();
		const auto end = std::chrono::steady_clock::now();

		if(i > n_warmup) { time += std::chrono::duration_cast<std::chrono::microseconds>(end - start); }
	}
	return time / n_passes;
}

bool is_master_node() {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank == 0;
}

// TODO this turns out to be really slow on Marconi with collectives enabled for 2 and 4 nodes.
//	- run again with device broadcast enabled
//  - is there a way to get_buffer_data() from multiple devices without the host-buffer hop?
TEST_CASE_METHOD(runtime_fixture, "Allgather") {
	const size_t n_warmup = 2;
	const size_t n_passes = 20;
	const size_t n_iters = 10;
	const size_t range = GENERATE(values({4_Ki, 256_Ki, 16_Mi, 256_Mi}));

	celerity::buffer<int> buf(range);

	const auto time = run_distributed_benchmark(n_warmup, n_passes, [&](celerity::distr_queue& q) mutable {
		q.submit([=](celerity::handler& cgh) {
			accessor write_acc(buf, cgh, access::one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(init)>(celerity::range<1>(range), [=](celerity::item<1> it) { //
				(void)write_acc;
			});
		});

		for(size_t i = 0; i < n_iters; ++i) {
			q.submit([=](celerity::handler& cgh) {
				accessor read_acc(buf, cgh, access::all(), read_only);
				accessor write_acc(buf, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(allgather)>(celerity::range<1>(range), [=](celerity::item<1>) {
					(void)read_acc;
					(void)write_acc;
				});
			});
		}
	});

	if(is_master_node()) { fmt::print("Allgather;{};{}\n", range, time.count()); }
}

TEST_CASE_METHOD(runtime_fixture, "Gather-Scatter") {
	const size_t n_warmup = 2;
	const size_t n_passes = 20;
	const size_t n_iters = 10;
	const size_t range = GENERATE(values({4_Ki, 256_Ki, 16_Mi, 256_Mi}));

	celerity::buffer<int> buf(range);

	const auto time = run_distributed_benchmark(n_warmup, n_passes, [&](celerity::distr_queue& q) mutable {
		q.submit([=](celerity::handler& cgh) {
			accessor write_acc(buf, cgh, access::one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(init)>(celerity::range<1>(range), [=](celerity::item<1> it) { (void)write_acc; });
		});

		for(size_t i = 0; i < n_iters; ++i) {
			q.submit([=](celerity::handler& cgh) {
				accessor acc(buf, cgh, access::all(), read_write);
				cgh.parallel_for<class UKN(gather)>(celerity::range<1>(1), [=](celerity::item<1>) { (void)acc; });
			});

			q.submit([=](celerity::handler& cgh) {
				accessor read_acc(buf, cgh, access::fixed<1>({0, 1}), read_only);
				accessor write_acc(buf, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(scatter)>(celerity::range<1>(range), [=](celerity::item<1>) { (void)read_acc, (void) write_acc; });
			});
		}
	});

	if(is_master_node()) { fmt::print("Gather-Bcast;{};{}\n", range, time.count()); }
}

TEST_CASE_METHOD(runtime_fixture, "Alltoall") {
	const size_t n_warmup = 2;
	const size_t n_passes = 20;
	const size_t n_iters = 10;
	const size_t size = GENERATE(values({64_i, 512_i, 4_Ki, 16_Ki}));
	const range<2> range(size, size);

	celerity::buffer<int, 2> buf_a(range);
	celerity::buffer<int, 2> buf_b(range);

	const auto time = run_distributed_benchmark(n_warmup, n_passes, [&](celerity::distr_queue& q) mutable {
		q.submit([=](celerity::handler& cgh) {
			accessor write_acc(buf_a, cgh, access::one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(init)>(range, [=](celerity::item<2> it) { //
				(void)write_acc;
			});
		});

		for(size_t i = 0; i < n_iters; ++i) {
			q.submit([=](celerity::handler& cgh) {
				accessor read_acc(buf_a, cgh, experimental::access::transposed<1, 0>(), read_only);
				accessor write_acc(buf_b, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(gather)>(range, [=](celerity::item<2>) {
					(void)read_acc;
					(void)write_acc;
				});
			});

			std::swap(buf_a, buf_b);
		}
	});

	if(is_master_node()) { fmt::print("Alltoall;{};{}\n", range.size(), time.count()); }
}
