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

unsigned long long operator""_Ki(unsigned long long v) { return v << 10; }
unsigned long long operator""_Mi(unsigned long long v) { return v << 20; }
unsigned long long operator""_Gi(unsigned long long v) { return v << 30; }

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

TEST_CASE_METHOD(runtime_fixture, "Allgather") {
	const size_t n_warmup = 2;
	const size_t n_passes = 20;
	const size_t n_iters = 100;
	const size_t range = GENERATE(values({2_Ki, 16_Ki, 256_Ki, 2_Mi, 16_Mi /*, 256_Mi, 2_Gi */}));

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

TEST_CASE_METHOD(runtime_fixture, "Gather-Bcast") {
	const size_t n_warmup = 2;
	const size_t n_passes = 20;
	const size_t n_iters = 100;
	const size_t range = GENERATE(values({2_Ki, 16_Ki, 256_Ki, 2_Mi, 16_Mi /*, 256_Mi, 2_Gi */}));

	celerity::buffer<int> buf_1(range);
	celerity::buffer<int> buf_2(1);

	const auto time = run_distributed_benchmark(n_warmup, n_passes, [&](celerity::distr_queue& q) mutable {
		q.submit([=](celerity::handler& cgh) {
			accessor read_acc(buf_1, cgh, access::all(), read_only);
			accessor write_acc(buf_2, cgh, access::one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(allgather)>(celerity::range<1>(1), [=](celerity::item<1>) {
				(void)read_acc;
				(void)write_acc;
			});
		});

		for(size_t i = 0; i < n_iters; ++i) {
			q.submit([=](celerity::handler& cgh) {
				accessor read_acc(buf_2, cgh, access::all(), read_only);
				accessor write_acc(buf_1, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(allgather)>(celerity::range<1>(range), [=](celerity::item<1>) {
					(void)read_acc;
					(void)write_acc;
				});
			});
		}
	});

	if(is_master_node()) { fmt::print("Gather-Bcast;{};{}\n", range, time.count()); }
}
