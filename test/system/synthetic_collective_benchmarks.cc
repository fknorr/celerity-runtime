#include <algorithm>

#include <celerity.h>
#include <mpi.h>


// To avoid having to come up with tons of unique kernel names, we simply use the CPP counter.
// This is non-standard but widely supported.
// TODO HACK copy-pasted from test_utils so we don't have to draw in Catch2 here
#define _UKN_CONCAT2(x, y) x##_##y
#define _UKN_CONCAT(x, y) _UKN_CONCAT2(x, y)
#define UKN(name) _UKN_CONCAT(name, __COUNTER__)


namespace celerity::detail {
extern bool NOMERGE_skip_device_kernel_execution;
}


using namespace celerity;

class benchmark_runner {
  public:
	explicit benchmark_runner(size_t total_num_devices, size_t n_warmup, size_t n_passes, const std::optional<std::string>& csv_file_name)
	    : m_total_num_devices(total_num_devices), m_n_warmup(n_warmup), m_n_passes(n_passes) {
		if(csv_file_name.has_value()) {
			m_csv = fopen(csv_file_name->c_str(), "wb");
			if(!m_csv) {
				perror("fopen");
				abort();
			}
			fprintf(m_csv, "devices;benchmark;range;system;configuration;nanoseconds\n");
			fflush(m_csv);
		}
	}

	benchmark_runner(benchmark_runner&&) = delete;
	benchmark_runner& operator=(benchmark_runner&&) = delete;

	~benchmark_runner() {
		if(m_csv) { fclose(m_csv); }
	}

	template <typename F>
	void run(const char* name, size_t range, F&& f) {
		for(const bool collectives : {false, true}) {
			celerity::detail::task_manager::NOMERGE_generate_collectives = collectives;
			const auto configuration = collectives ? "collectives" : "p2p";

			for(size_t i = 0; i < m_n_warmup + m_n_passes; ++i) {
				m_q.slow_full_sync();
				const auto start = std::chrono::steady_clock::now();
				f(m_q);
				m_q.slow_full_sync();
				const auto end = std::chrono::steady_clock::now();

				if(i >= m_n_warmup && m_csv) {
					const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
					fmt::print(m_csv, "{};{};{};Celerity;{};{}\n", m_total_num_devices, name, range, configuration, ns.count());
					fflush(m_csv);
				}
			}
		}
	}

  private:
	celerity::distr_queue m_q;
	size_t m_total_num_devices;
	size_t m_n_warmup;
	size_t m_n_passes;
	FILE* m_csv = nullptr;
};


size_t operator""_i(unsigned long long v) { return v; }
size_t operator""_Ki(unsigned long long v) { return v << 10; }
size_t operator""_Mi(unsigned long long v) { return v << 20; }
size_t operator""_Gi(unsigned long long v) { return v << 30; }


void allgather_benchmark(benchmark_runner& runner) {
	const size_t n_iters = 10;

	for(const auto range : {4_Ki, 256_Ki, 16_Mi, 256_Mi}) {
		celerity::buffer<int> buf_a(range);
		celerity::buffer<int> buf_b(range);

		runner.run("Allgather", range, [&](celerity::distr_queue& q) {
			q.submit([=](celerity::handler& cgh) {
				accessor write_acc(buf_a, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(init)>(celerity::range<1>(range), [=](celerity::item<1> it) { (void)write_acc; });
			});

			for(size_t i = 0; i < n_iters; ++i) {
				q.submit([=](celerity::handler& cgh) {
					accessor read_acc(buf_a, cgh, access::all(), read_only);
					accessor write_acc(buf_b, cgh, access::one_to_one(), write_only, no_init);
					cgh.parallel_for<class UKN(allgather)>(celerity::range<1>(range), [=](celerity::item<1>) { (void)read_acc, (void)write_acc; });
				});
				std::swap(buf_a, buf_b);
			}
		});
	}
}


void gather_broadcast_benchmark(benchmark_runner& runner) {
	const size_t n_iters = 10;
	for(const auto range : {4_Ki, 256_Ki, 16_Mi, 256_Mi}) {
		celerity::buffer<int> buf_a(range);
		celerity::buffer<int> buf_b(range);

		runner.run("Gather-Bcast", range, [&](celerity::distr_queue& q) {
			q.submit([=](celerity::handler& cgh) {
				accessor write_acc(buf_a, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(init)>(celerity::range<1>(range), [=](celerity::item<1> it) { (void)write_acc; });
			});

			for(size_t i = 0; i < n_iters; ++i) {
				q.submit([=](celerity::handler& cgh) {
					accessor acc(buf_a, cgh, access::all(), read_write);
					cgh.parallel_for<class UKN(gather)>(celerity::range<1>(1), [=](celerity::item<1>) { (void)acc; });
				});

				q.submit([=](celerity::handler& cgh) {
					accessor read_acc(buf_a, cgh, access::all(), read_only);
					accessor write_acc(buf_b, cgh, access::one_to_one(), write_only, no_init);
					cgh.parallel_for<class UKN(scatter)>(celerity::range<1>(range), [=](celerity::item<1>) { (void)read_acc, (void)write_acc; });
				});
				std::swap(buf_a, buf_b);
			}
		});
	}
}


void gather_scatter_benchmark(benchmark_runner& runner) {
	const size_t n_iters = 10;
	for(const auto range : {4_Ki, 256_Ki, 16_Mi, 256_Mi}) {
		celerity::buffer<int> buf(range);

		runner.run("Gather-Scatter", range, [&](celerity::distr_queue& q) {
			q.submit([=](celerity::handler& cgh) {
				accessor acc(buf, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(init)>(celerity::range<1>(range), [=](celerity::item<1> it) { (void)acc; });
			});

			for(size_t i = 0; i < n_iters; ++i) {
				q.submit([=](celerity::handler& cgh) {
					accessor acc(buf, cgh, access::all(), read_write);
					cgh.parallel_for<class UKN(gather)>(celerity::range<1>(1), [=](celerity::item<1>) { (void)acc; });
				});

				q.submit([=](celerity::handler& cgh) {
					accessor acc(buf, cgh, access::one_to_one(), read_write);
					cgh.parallel_for<class UKN(scatter)>(celerity::range<1>(range), [=](celerity::item<1>) { (void)acc; });
				});
			}
		});
	}
}


void alltoall_benchmark(benchmark_runner& runner) {
	const size_t n_iters = 10;

	for(auto size : {64_i, 512_i, 4_Ki, 16_Ki}) {
		const range<2> range(size, size);

		celerity::buffer<int, 2> buf_a(range);
		celerity::buffer<int, 2> buf_b(range);

		runner.run("Alltoall", range.size(), [&](celerity::distr_queue& q) {
			q.submit([=](celerity::handler& cgh) {
				accessor write_acc(buf_a, cgh, access::one_to_one(), write_only, no_init);
				cgh.parallel_for<class UKN(init)>(range, [=](celerity::item<2> it) { (void)write_acc; });
			});

			for(size_t i = 0; i < n_iters; ++i) {
				q.submit([=](celerity::handler& cgh) {
					accessor read_acc(buf_a, cgh, experimental::access::transposed<1, 0>(), read_only);
					accessor write_acc(buf_b, cgh, access::one_to_one(), write_only, no_init);
					cgh.parallel_for<class UKN(gather)>(range, [=](celerity::item<2>) { (void)read_acc, (void)write_acc; });
				});
				std::swap(buf_a, buf_b);
			}
		});
	}
}


int main(int argc, char** argv) {
	celerity::runtime::init(&argc, &argv);

	int comm_size, comm_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	const size_t total_num_devices = comm_size * celerity::detail::runtime::get_instance().get_local_devices().num_compute_devices();

	std::optional<std::string> csv_name;
	if(comm_rank == 0) {
		if(const char* job_id = getenv("SLURM_JOB_ID")) {
			csv_name = fmt::format("celerity-scb-{}-{}.csv", total_num_devices, job_id);
		} else {
			csv_name = fmt::format("celerity-scb-{}.csv", total_num_devices);
		}
		CELERITY_INFO("Writing to {}", *csv_name);
	}

	const size_t n_warmup = 2;
	const size_t n_passes = 10;

	celerity::detail::NOMERGE_skip_device_kernel_execution = true;
	benchmark_runner runner(total_num_devices, n_warmup, n_passes, csv_name);

	allgather_benchmark(runner);
	gather_scatter_benchmark(runner);
	gather_broadcast_benchmark(runner);
	alltoall_benchmark(runner);
}
