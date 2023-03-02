#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <mpi.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

size_t operator""_i(unsigned long long v) { return v; }
size_t operator""_Ki(unsigned long long v) { return v << 10; }
size_t operator""_Mi(unsigned long long v) { return v << 20; }
size_t operator""_Gi(unsigned long long v) { return v << 30; }

template <typename... Printf>
void log(const char* fmt, Printf... p) {
	fprintf(stderr, fmt, p...);
	fputc('\n', stderr);
}

template <typename... Printf>
[[noreturn]] void panic(Printf... p) {
	log(p...);
	abort();
}

template <typename... Printf>
[[noreturn]] void panic(const char* cuda_func, cudaError_t cuda_error) {
	panic("%s: %s", cuda_func, cudaGetErrorString(cuda_error));
}

#define CUDA_CHECK(f, ...)                                                                                                                                     \
	if(const cudaError_t cuda_error = (f)(__VA_ARGS__); cuda_error != cudaSuccess) { panic(STRINGIFY(f), cuda_error); }

constexpr int max_num_devices = 4;
constexpr int max_comm_size = 64;

using comm_func = void (*)(int comm_rank, int comm_size, int* buffer, size_t rank_range);
using alltoall_func = void (*)(int comm_rank, int comm_size, int*& buffer, int*& aux, size_t rank_range);

struct collective_host_allgather {
	void operator()(int, int, int* buffer, size_t rank_range) const {
		ZoneScopedN("collective_host_allgather");
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, buffer, rank_range, MPI_INT, MPI_COMM_WORLD);
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_allgather {
	void operator()(int comm_rank, int comm_size, int* buffer, size_t rank_range) const {
		ZoneScopedN("p2p_host_allgather");

		MPI_Request sends[max_comm_size];
		MPI_Request recvs[max_comm_size];
		for(int other_rank = 0; other_rank < comm_size; ++other_rank) {
			if(other_rank != comm_rank) {
				MPI_Isend(buffer + comm_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &sends[other_rank]);
				MPI_Irecv(buffer + other_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &recvs[other_rank]);
			} else {
				sends[other_rank] = MPI_REQUEST_NULL;
				recvs[other_rank] = MPI_REQUEST_NULL;
			}
		}
		MPI_Waitall(comm_size, recvs, MPI_STATUSES_IGNORE);
		MPI_Waitall(comm_size, sends, MPI_STATUSES_IGNORE);
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostAllgather>
struct device_allgather {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_allgather");

		const size_t rank_range = buffer_range / comm_size;
		const size_t rank_offset = rank_range * comm_rank;
		{
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer + rank_offset, device_buffer + rank_offset, rank_range * sizeof(int), cudaMemcpyDefault);
		}

		HostAllgather{}(comm_rank, comm_size, host_buffer, rank_range);

		const size_t range_before = rank_offset;
		const size_t offset_after = rank_offset + rank_range;
		const size_t range_after = buffer_range - offset_after;
		if(range_before > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer, host_buffer, range_before * sizeof(int), cudaMemcpyDefault);
		}
		if(range_after > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + offset_after, host_buffer + offset_after, range_after * sizeof(int), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostAllgather::configuration;
};

template <typename DeviceAllgather>
struct allgather_pass {
	void operator()(size_t range, int comm_rank, int comm_size, int* host_buffer, int*, int* device_buffer, int n_iter) {
		ZoneScopedN("allgather_pass");

		for(int j = 0; j < n_iter; ++j) {
			ZoneScopedN("iter");
			// imagine a no-op kernel here
			DeviceAllgather{}(comm_rank, comm_size, device_buffer, host_buffer, range);
		}
	}

	static constexpr const char* benchmark = "Allgather";
	static constexpr const char* configuration = DeviceAllgather::configuration;
};

struct collective_host_gather {
	void operator()(int comm_rank, int, int* buffer, size_t rank_range) const {
		ZoneScopedN("collective_host_gather");

		if(comm_rank == 0) {
			MPI_Gather(MPI_IN_PLACE, 0, MPI_INT, buffer, rank_range, MPI_INT, 0, MPI_COMM_WORLD);
		} else {
			MPI_Gather(buffer + comm_rank * rank_range, rank_range, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_gather {
	void operator()(int comm_rank, int comm_size, int* buffer, size_t rank_range) const {
		ZoneScopedN("p2p_host_gather");

		if(comm_rank == 0) {
			MPI_Request recvs[max_comm_size];
			recvs[0] = MPI_REQUEST_NULL;
			for(int other_rank = 1; other_rank < comm_size; ++other_rank) {
				MPI_Irecv(buffer + other_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &recvs[other_rank]);
			}
			MPI_Waitall(comm_size, recvs, MPI_STATUSES_IGNORE);
		} else {
			MPI_Send(buffer + comm_rank * rank_range, rank_range, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostGather>
struct device_gather {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_gather");

		const size_t rank_range = buffer_range / comm_size;
		const size_t rank_offset = rank_range * comm_rank;
		if(comm_rank > 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer + rank_offset, device_buffer + rank_offset, rank_range * sizeof(int), cudaMemcpyDefault);
		}

		HostGather{}(comm_rank, comm_size, host_buffer, rank_range);

		if(comm_rank == 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + rank_range, host_buffer + rank_range, rank_range * (comm_size - 1) * sizeof(int), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostGather::configuration;
};

struct collective_host_scatter {
	void operator()(int comm_rank, int, int* buffer, size_t rank_range) const {
		ZoneScopedN("collective_host_scatter");

		if(comm_rank == 0) {
			// TODO Spectrum MPI does not seem to recognize recvBuffer == MPI_IN_PLACE
			MPI_Scatter(buffer, rank_range, MPI_INT, buffer, rank_range, MPI_INT, 0, MPI_COMM_WORLD);
		} else {
			MPI_Scatter(nullptr, 0, MPI_INT, buffer + comm_rank * rank_range, rank_range, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_scatter {
	void operator()(int comm_rank, int comm_size, int* buffer, size_t rank_range) const {
		ZoneScopedN("p2p_host_scatter");

		if(comm_rank == 0) {
			MPI_Request sends[max_comm_size];
			sends[0] = MPI_REQUEST_NULL;
			for(int other_rank = 1; other_rank < comm_size; ++other_rank) {
				MPI_Isend(buffer + other_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &sends[other_rank]);
			}
			MPI_Waitall(comm_size, sends, MPI_STATUSES_IGNORE);
		} else {
			MPI_Recv(buffer + comm_rank * rank_range, rank_range, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostScatter>
struct device_scatter {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_scatter");

		const size_t rank_range = buffer_range / comm_size;
		const size_t rank_offset = rank_range * comm_rank;
		if(comm_rank == 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer + rank_range, device_buffer + rank_range, rank_range * (comm_size - 1) * sizeof(int), cudaMemcpyDefault);
		}

		HostScatter{}(comm_rank, comm_size, host_buffer, rank_range);

		if(comm_rank > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + rank_offset, host_buffer + rank_offset, rank_range * sizeof(int), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostScatter::configuration;
};

template <typename DeviceGather, typename DeviceScatter>
struct gather_scatter_pass {
	void operator()(size_t range, int comm_rank, int comm_size, int* host_buffer, int*, int* device_buffer, int n_iter) const {
		ZoneScopedN("gather_scatter_pass");

		for(int j = 0; j < n_iter; ++j) {
			ZoneScopedN("iter");
			// imagine a kernel here
			DeviceGather{}(comm_rank, comm_size, device_buffer, host_buffer, range);
			// imagine a kernel here
			DeviceScatter{}(comm_rank, comm_size, device_buffer, host_buffer, range);
		}
	}

	static constexpr const char* benchmark = "Gather-Scatter";
	static constexpr const char* configuration = DeviceGather::configuration;
};

struct collective_host_bcast {
	void operator()(int, int, int* buffer, size_t buffer_range) const {
		ZoneScopedN("collective_host_bcast");
		MPI_Bcast(buffer, buffer_range, MPI_INT, 0, MPI_COMM_WORLD);
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_bcast {
	void operator()(int comm_rank, int comm_size, int* buffer, size_t buffer_range) const {
		ZoneScopedN("p2p_host_bcast");

		if(comm_rank == 0) {
			MPI_Request sends[max_comm_size];
			sends[0] = MPI_REQUEST_NULL;
			for(int other_rank = 1; other_rank < comm_size; ++other_rank) {
				MPI_Isend(buffer, buffer_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &sends[other_rank]);
			}
			MPI_Waitall(comm_size, sends, MPI_STATUSES_IGNORE);
		} else {
			MPI_Recv(buffer, buffer_range, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostBcast>
struct device_bcast {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_bcast");
		if(comm_rank == 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer, device_buffer, buffer_range * sizeof(int), cudaMemcpyDefault);
		}

		HostBcast{}(comm_rank, comm_size, host_buffer, buffer_range);

		if(comm_rank > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer, host_buffer, buffer_range * sizeof(int), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostBcast::configuration;
};

template <typename DeviceGather, typename DeviceBcast>
struct gather_bcast_pass {
	void operator()(size_t range, int comm_rank, int comm_size, int* host_alloc, int*, int* device_alloc, int n_iter) const {
		ZoneScopedN("gather_bcast_pass");

		for(int j = 0; j < n_iter; ++j) {
			ZoneScopedN("iter");
			// imagine a kernel here
			DeviceGather{}(comm_rank, comm_size, device_alloc, host_alloc, range);
			// imagine a kernel here
			DeviceBcast{}(comm_rank, comm_size, device_alloc, host_alloc, range);
		}
	}

	static constexpr const char* benchmark = "Gather-Bcast";
	static constexpr const char* configuration = DeviceGather::configuration;
};

struct collective_host_alltoall {
	void operator()(int, int, int* buffer, int*, size_t rank_range) const {
		ZoneScopedN("collective_host_alltoall");
		MPI_Alltoall(MPI_IN_PLACE, 0, MPI_INT, buffer, rank_range, MPI_INT, MPI_COMM_WORLD);
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_alltoall {
	void operator()(int comm_rank, int comm_size, int*& buffer, int*& aux_buffer, size_t rank_range) const {
		ZoneScopedN("p2p_host_alltoall");

		MPI_Request sends[max_comm_size];
		MPI_Request recvs[max_comm_size];
		for(int other_rank = 0; other_rank < comm_size; ++other_rank) {
			if(other_rank != comm_rank) {
				MPI_Isend(buffer + other_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &sends[other_rank]);
				MPI_Irecv(aux_buffer + other_rank * rank_range, rank_range, MPI_INT, other_rank, 0, MPI_COMM_WORLD, &recvs[other_rank]);
			} else {
				sends[other_rank] = MPI_REQUEST_NULL;
				recvs[other_rank] = MPI_REQUEST_NULL;
			}
		}
		MPI_Waitall(comm_size, recvs, MPI_STATUSES_IGNORE);
		MPI_Waitall(comm_size, sends, MPI_STATUSES_IGNORE);
		std::swap(aux_buffer, buffer);
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostAlltoall>
struct device_alltoall {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int*& host_buffer, int*& aux_buffer, size_t buffer_range) const {
		ZoneScopedN("device_alltoall");

		const size_t rank_range = buffer_range / comm_size;
		const size_t rank_offset = rank_range * comm_rank;
		const size_t range_before = rank_offset;
		const size_t offset_after = rank_offset + rank_range;
		const size_t range_after = buffer_range - offset_after;

		if(range_before > 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer, device_buffer, range_before * sizeof(int), cudaMemcpyDefault);
		}
		if(range_after > 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer + offset_after, device_buffer + offset_after, range_after * sizeof(int), cudaMemcpyDefault);
		}

		HostAlltoall{}(comm_rank, comm_size, host_buffer, aux_buffer, rank_range);

		if(range_before > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer, host_buffer, range_before * sizeof(int), cudaMemcpyDefault);
		}
		if(range_after > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + offset_after, host_buffer + offset_after, range_after * sizeof(int), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostAlltoall::configuration;
};

template <typename DeviceAlltoall>
struct alltoall_pass {
	void operator()(size_t range, int comm_rank, int comm_size, int* host_alloc, int* aux_alloc, int* device_alloc, int n_iter) const {
		ZoneScopedN("alltoall_pass");

		for(int j = 0; j < n_iter; ++j) {
			ZoneScopedN("iter");
			// imagine a no-op kernel here
			DeviceAlltoall{}(comm_rank, comm_size, device_alloc, host_alloc, aux_alloc, range);
		}
	}

	static constexpr const char* benchmark = "Alltoall";
	static constexpr const char* configuration = DeviceAlltoall::configuration;
};

struct p2p_host_boundex {
	void operator()(int comm_rank, int comm_size, int* buffer, size_t rank_range, size_t boundary_range) const {
		ZoneScopedN("p2p_host_boundex");

		MPI_Request reqs[4];
		if(comm_rank > 0) {
			MPI_Isend(buffer + comm_rank * rank_range, boundary_range, MPI_INT, comm_rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(buffer + comm_rank * rank_range - boundary_range, boundary_range, MPI_INT, comm_rank - 1, 0, MPI_COMM_WORLD, &reqs[1]);
		} else {
			reqs[0] = reqs[1] = MPI_REQUEST_NULL;
		}
		if(comm_rank + 1 < comm_size) {
			MPI_Isend(buffer + (comm_rank + 1) * rank_range - boundary_range, boundary_range, MPI_INT, comm_rank + 1, 0, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(buffer + (comm_rank + 1) * rank_range, boundary_range, MPI_INT, comm_rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);
		} else {
			reqs[2] = reqs[3] = MPI_REQUEST_NULL;
		}
		MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
	}

	static constexpr const char* configuration = "p2p";
};

template <typename HostBoundex>
struct device_boundex {
	void operator()(int comm_rank, int comm_size, int* device_buffer, int* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_boundex");

		const size_t rank_range = buffer_range / comm_size;
		const size_t boundary_range = static_cast<size_t>(sqrt(buffer_range));
		assert(boundary_range * boundary_range == buffer_range);

		if(comm_rank > 0) {
			ZoneScopedN("d2h");
			CUDA_CHECK(
			    cudaMemcpy, host_buffer + comm_rank * rank_range, device_buffer + comm_rank * rank_range, boundary_range * sizeof(int), cudaMemcpyDefault);
		}
		if(comm_rank + 1 < comm_size) {
			ZoneScopedN("d2h");
			CUDA_CHECK(cudaMemcpy, host_buffer + (comm_rank + 1) * rank_range - boundary_range, device_buffer + (comm_rank + 1) * rank_range - boundary_range,
			    boundary_range * sizeof(int), cudaMemcpyDefault);
		}

		HostBoundex{}(comm_rank, comm_size, host_buffer, rank_range, boundary_range);

		if(comm_rank > 0) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + comm_rank * rank_range - boundary_range, host_buffer + comm_rank * rank_range - boundary_range,
			    boundary_range * sizeof(int), cudaMemcpyDefault);
		}
		if(comm_rank + 1 < comm_size) {
			ZoneScopedN("h2d");
			CUDA_CHECK(cudaMemcpy, device_buffer + (comm_rank + 1) * rank_range, host_buffer + (comm_rank + 1) * rank_range, boundary_range * sizeof(int),
			    cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostBoundex::configuration;
};

template <typename DeviceBoundex>
struct stencil_pass {
	void operator()(size_t range, int comm_rank, int comm_size, int* host_alloc, int*, int* device_alloc, int n_iter) const {
		ZoneScopedN("stencil_pass");

		for(int j = 0; j < n_iter; ++j) {
			ZoneScopedN("iter");
			// imagine a no-op kernel here
			DeviceBoundex{}(comm_rank, comm_size, device_alloc, host_alloc, range);
		}
	}

	static constexpr const char* benchmark = "Stencil";
	static constexpr const char* configuration = DeviceBoundex::configuration;
};

template <typename Pass>
void benchmark(FILE* csv, size_t range, int comm_rank, int comm_size) {
	int* host_alloc;
	CUDA_CHECK(cudaHostAlloc, (void**)&host_alloc, range * sizeof(int), cudaHostAllocDefault | cudaHostAllocPortable);
	int* host_alloc_aux;
	CUDA_CHECK(cudaHostAlloc, (void**)&host_alloc_aux, range * sizeof(int), cudaHostAllocDefault | cudaHostAllocPortable);

	int* device_alloc;
	CUDA_CHECK(cudaMalloc, (void**)&device_alloc, range * sizeof(int));

	const int n_warmup_iters = 2;
	const int n_measured_iters = 20;

	Pass{}(range, comm_rank, comm_size, host_alloc, host_alloc_aux, device_alloc, n_warmup_iters);

	MPI_Barrier(MPI_COMM_WORLD);
	const auto start = std::chrono::steady_clock::now();

	Pass{}(range, comm_rank, comm_size, host_alloc, host_alloc_aux, device_alloc, n_measured_iters);

	MPI_Barrier(MPI_COMM_WORLD);
	const auto end = std::chrono::steady_clock::now();

	if(comm_rank == 0) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		fprintf(csv, "%d;%s;%zu;MPI+CUDA;%s;%llu\n", comm_size, Pass::benchmark, range, Pass::configuration, (unsigned long long)ns.count());
		fflush(csv);
	}

	CUDA_CHECK(cudaFree, device_alloc);
	CUDA_CHECK(cudaFreeHost, host_alloc_aux);
	CUDA_CHECK(cudaFreeHost, host_alloc);
}


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

#if TRACY_ENABLE
	tracy::StartupProfiler();
#endif

	int comm_size, comm_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

	if(comm_rank == 0) log("have %d rank(s)", comm_size);

	int num_devices;
	CUDA_CHECK(cudaGetDeviceCount, &num_devices);
	num_devices = std::min(num_devices, max_num_devices);
	int device = 0;
	if(num_devices < 1) {
		panic("no devices found.");
	} else if(num_devices > 1) {
		device = comm_rank % num_devices;
		log("found %d device(s), picking device %d", num_devices, device);
	}
	CUDA_CHECK(cudaSetDevice, device);

	FILE* csv = nullptr;
	if(comm_rank == 0) {
		char csv_name[100];
		const char* job_id = getenv("SLURM_JOB_ID");
		if(job_id) {
			snprintf(csv_name, sizeof csv_name, "scb-%d-%s.csv", comm_size, job_id);
		} else {
			snprintf(csv_name, sizeof csv_name, "scb-%d.csv", comm_size);
		}
		csv = fopen(csv_name, "wb");
		if(!csv) panic("error opening output file: %s: %s", csv_name, strerror(errno));
		fprintf(csv, "devices;benchmark;range;system;configuration;nanoseconds\n");
		fflush(csv);
	}

	for(auto range : {4_Ki, 256_Ki, 1_Mi, 16_Mi, 256_Mi}) {
		benchmark<allgather_pass<device_allgather<collective_host_allgather>>>(csv, range, comm_rank, comm_size);
		benchmark<allgather_pass<device_allgather<p2p_host_allgather>>>(csv, range, comm_rank, comm_size);
		benchmark<gather_scatter_pass<device_gather<collective_host_gather>, device_scatter<collective_host_scatter>>>(csv, range, comm_rank, comm_size);
		benchmark<gather_scatter_pass<device_gather<p2p_host_gather>, device_scatter<p2p_host_scatter>>>(csv, range, comm_rank, comm_size);
		benchmark<gather_bcast_pass<device_gather<collective_host_gather>, device_bcast<collective_host_bcast>>>(csv, range, comm_rank, comm_size);
		benchmark<gather_bcast_pass<device_gather<p2p_host_gather>, device_bcast<p2p_host_bcast>>>(csv, range, comm_rank, comm_size);
		benchmark<alltoall_pass<device_alltoall<collective_host_alltoall>>>(csv, range, comm_rank, comm_size);
		benchmark<alltoall_pass<device_alltoall<p2p_host_alltoall>>>(csv, range, comm_rank, comm_size);
		benchmark<stencil_pass<device_boundex<p2p_host_boundex>>>(csv, range, comm_rank, comm_size);
	}

	if(comm_rank == 0) fclose(csv);

#if TRACY_ENABLE
	tracy::ShutdownProfiler();
#endif

	MPI_Finalize();
}
