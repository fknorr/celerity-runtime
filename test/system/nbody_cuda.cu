#include <cerrno>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <vector>
using namespace std::chrono_literals;

#include "utils/wyhash.h"
#include <mpi.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#ifndef USE_FLOAT
#define USE_FLOAT 1
#endif

#if USE_FLOAT == 1
#define VAL_TYPE float
#define MPI_VAL_TYPE MPI_FLOAT
#else
#define VAL_TYPE double
#define MPI_VAL_TYPE MPI_DOUBLE
#endif

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr VAL_TYPE SOFTENING = 1e-9f;
constexpr VAL_TYPE DT = 0.01f;

void randomizeBodies(VAL_TYPE* data, int n) {
	unsigned int seed = 42;
	for(int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand_r(&seed) / (VAL_TYPE)RAND_MAX) - 1.0f;
	}
}

__device__ static inline double r_sqrt(double v) { return rsqrt(v); }
__device__ static inline float r_sqrt(float v) { return rsqrtf(v); }

__device__ static inline void inplace_warp_sum(VAL_TYPE& val) {
	for(int offset = 16; offset > 0; offset /= 2) {
		val += __shfl_down_sync(0xffff'ffff, val, offset);
	}
}

__global__ void bodyForce(
    VAL_TYPE* pos_x, VAL_TYPE* pos_y, VAL_TYPE* pos_z, VAL_TYPE* vel_x, VAL_TYPE* vel_y, VAL_TYPE* vel_z, int n_bodies, int rank_offset, int rank_range) {
	__shared__ VAL_TYPE stage[3][BLOCK_SIZE];

	const int input_tid = threadIdx.x;
	const int output_item = rank_offset + blockDim.x * blockIdx.x + threadIdx.x / WARP_SIZE;
	const int output_tid = threadIdx.x % WARP_SIZE;

	VAL_TYPE out_pos_x;
	VAL_TYPE out_pos_y;
	VAL_TYPE out_pos_z;
	VAL_TYPE force_x = 0.0f;
	VAL_TYPE force_y = 0.0f;
	VAL_TYPE force_z = 0.0f;

	if(output_item < n_bodies) {
		out_pos_x = pos_x[output_item];
		out_pos_y = pos_y[output_item];
		out_pos_z = pos_z[output_item];
	}

	for(int input_offset = 0; input_offset < n_bodies; input_offset += BLOCK_SIZE) {
		const int input_item = input_tid + input_offset;

		if(input_item < n_bodies) {
			stage[0][input_tid] = pos_x[input_item];
			stage[1][input_tid] = pos_y[input_item];
			stage[2][input_tid] = pos_z[input_item];
		}
		__syncthreads();

		if(input_item < n_bodies) {
			for(int stage_item = output_tid; stage_item < BLOCK_SIZE; stage_item += WARP_SIZE) {
				const VAL_TYPE dx = stage[0][stage_item] - out_pos_x;
				const VAL_TYPE dy = stage[1][stage_item] - out_pos_y;
				const VAL_TYPE dz = stage[2][stage_item] - out_pos_z;
				const VAL_TYPE distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
				const VAL_TYPE invDist = r_sqrt(distSqr);
				const VAL_TYPE invDist3 = -invDist * invDist * invDist;

				force_x += dx * invDist3;
				force_y += dy * invDist3;
				force_z += dz * invDist3;
			}
		}
		__syncthreads();
	}

	inplace_warp_sum(force_x);
	inplace_warp_sum(force_y);
	inplace_warp_sum(force_z);

	if(output_item < n_bodies && output_tid == 0) {
		vel_x[output_item] += DT * force_x;
		vel_y[output_item] += DT * force_y;
		vel_z[output_item] += DT * force_z;
	}
}

__global__ void bodyPos(VAL_TYPE* pos_x, VAL_TYPE* pos_y, VAL_TYPE* pos_z, VAL_TYPE* vel_x, VAL_TYPE* vel_y, VAL_TYPE* vel_z, int rank_offset, int rank_range) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < rank_range) {
		pos_x[rank_offset + i] += vel_x[rank_offset + i] * DT;
		pos_y[rank_offset + i] += vel_y[rank_offset + i] * DT;
		pos_z[rank_offset + i] += vel_z[rank_offset + i] * DT;
	}
}

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

template <typename... Printf>
void log(const char* fmt, Printf... p) {
	int comm_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	fprintf(stderr, "[%d] ", comm_rank);
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

struct collective_host_allgather {
	void operator()(int, int, VAL_TYPE* buffer, size_t rank_range) const {
		ZoneScopedN("Allgather");
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_VAL_TYPE, buffer, rank_range, MPI_VAL_TYPE, MPI_COMM_WORLD);
	}

	static constexpr const char* configuration = "collective";
};

struct p2p_host_allgather {
	void operator()(int comm_rank, int comm_size, VAL_TYPE* buffer, size_t rank_range) const {
		ZoneScopedN("Send/Recv");

		MPI_Request sends[max_comm_size];
		MPI_Request recvs[max_comm_size];
		for(int other_rank = 0; other_rank < comm_size; ++other_rank) {
			if(other_rank != comm_rank) {
				MPI_Isend(buffer + comm_rank * rank_range, rank_range, MPI_VAL_TYPE, other_rank, 0, MPI_COMM_WORLD, &sends[other_rank]);
				MPI_Irecv(buffer + other_rank * rank_range, rank_range, MPI_VAL_TYPE, other_rank, 0, MPI_COMM_WORLD, &recvs[other_rank]);
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
	void operator()(int comm_rank, int comm_size, VAL_TYPE* device_buffer, VAL_TYPE* host_buffer, size_t buffer_range) const {
		ZoneScopedN("device_allgather");

		const size_t rank_range = buffer_range / comm_size;
		const size_t rank_offset = rank_range * comm_rank;
		CUDA_CHECK(cudaMemcpy, host_buffer + rank_offset, device_buffer + rank_offset, rank_range * sizeof(VAL_TYPE), cudaMemcpyDefault);

		HostAllgather{}(comm_rank, comm_size, host_buffer, rank_range);

		const size_t range_before = rank_offset;
		const size_t offset_after = rank_offset + rank_range;
		const size_t range_after = buffer_range - offset_after;
		if(range_before > 0) { CUDA_CHECK(cudaMemcpy, device_buffer, host_buffer, range_before * sizeof(VAL_TYPE), cudaMemcpyDefault); }
		if(range_after > 0) {
			CUDA_CHECK(cudaMemcpy, device_buffer + offset_after, host_buffer + offset_after, range_after * sizeof(VAL_TYPE), cudaMemcpyDefault);
		}
	}

	static constexpr const char* configuration = HostAllgather::configuration;
};


template <typename DeviceAllgather>
struct nbody_pass {
	void operator()(int comm_rank, int comm_size, VAL_TYPE* d_buf, VAL_TYPE* h_buf, int n, int nIters) const {
		ZoneScopedN("pass");

		const size_t rank_range = n / comm_size;
		const size_t rank_offset = rank_range * comm_rank;

		const auto h_pos_x = h_buf + 0 * n;
		const auto h_pos_y = h_buf + 1 * n;
		const auto h_pos_z = h_buf + 2 * n;

		const auto d_pos_x = d_buf + 0 * n;
		const auto d_pos_y = d_buf + 1 * n;
		const auto d_pos_z = d_buf + 2 * n;
		const auto d_vel_x = d_buf + 3 * n;
		const auto d_vel_y = d_buf + 4 * n;
		const auto d_vel_z = d_buf + 5 * n;

		const auto body_force_blocks = (rank_range * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
		const auto body_pos_blocks = (rank_range + BLOCK_SIZE - 1) / BLOCK_SIZE;
		for(int iter = 0; iter < nIters; iter++) {
			ZoneScopedN("iter");

			bodyForce<<<body_force_blocks, BLOCK_SIZE>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, n, rank_offset, rank_range);
			bodyPos<<<body_pos_blocks, BLOCK_SIZE>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, rank_offset, rank_range);

			DeviceAllgather{}(comm_rank, comm_size, d_pos_x, h_pos_x, n);
			DeviceAllgather{}(comm_rank, comm_size, d_pos_y, h_pos_y, n);
			DeviceAllgather{}(comm_rank, comm_size, d_pos_z, h_pos_z, n);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	static constexpr const char* benchmark = "NBody";
	static constexpr const char* configuration = DeviceAllgather::configuration;
};

template <typename DeviceAllgather>
void benchmark(FILE* csv, int comm_rank, int comm_size, int n, int nIters) {
	using pass = nbody_pass<DeviceAllgather>;

	const size_t buf_count = 2 * 3 * n;

	VAL_TYPE* h_buf;
	CUDA_CHECK(cudaHostAlloc, &h_buf, buf_count * sizeof(VAL_TYPE), cudaHostAllocDefault | cudaHostAllocPortable);

	randomizeBodies(h_buf, buf_count); // init pos / vel data

	VAL_TYPE* d_buf;
	CUDA_CHECK(cudaMalloc, &d_buf, buf_count * sizeof(VAL_TYPE));

	const int n_warmup = 2;
	const int n_passes = 10;

	for(int i = 0; i < n_warmup + n_passes; ++i) {
		CUDA_CHECK(cudaMemcpy, d_buf, h_buf, buf_count * sizeof(VAL_TYPE), cudaMemcpyHostToDevice);

		MPI_Barrier(MPI_COMM_WORLD);
		const auto start = std::chrono::steady_clock::now();
		pass{}(comm_rank, comm_size, d_buf, h_buf, n, nIters);
		const auto end = std::chrono::steady_clock::now();

		if(comm_rank == 0 && i >= n_warmup) {
			const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
			fprintf(csv, "%d;%s;%d;CUDA+MPI;%s;%s;%d;%llu\n", comm_size, pass::benchmark, n, pass::configuration, USE_FLOAT ? "single" : "double", nIters,
			    (unsigned long long)ns.count());
			fflush(csv);
		}
	}

	const auto h_vel_x = h_buf + 3 * n;
	const auto h_vel_y = h_buf + 4 * n;
	const auto h_vel_z = h_buf + 5 * n;

	const auto d_vel_x = d_buf + 3 * n;
	const auto d_vel_y = d_buf + 4 * n;
	const auto d_vel_z = d_buf + 5 * n;

	DeviceAllgather{}(comm_rank, comm_size, d_vel_x, h_vel_x, n);
	DeviceAllgather{}(comm_rank, comm_size, d_vel_y, h_vel_y, n);
	DeviceAllgather{}(comm_rank, comm_size, d_vel_z, h_vel_z, n);

	if(comm_rank == 0) {
		CUDA_CHECK(cudaMemcpy, h_buf, d_buf, 2 * n * sizeof(VAL_TYPE), cudaMemcpyDeviceToHost);

		const auto h_pos_x = h_buf + 0 * n;
		const auto h_pos_y = h_buf + 1 * n;
		const auto h_pos_z = h_buf + 2 * n;

		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += h_pos_x[i] + h_pos_y[i] + h_pos_z[i];
		}
		const auto hash = wyhash(h_buf, buf_count * sizeof(VAL_TYPE), 0, _wyp);
		printf("%12s result hash: %16lX, position avg: %15.8f\n", pass::configuration, hash, sum / n);
	}

	CUDA_CHECK(cudaFree, d_buf);
	CUDA_CHECK(cudaFreeHost, h_buf);
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
	} else {
		device = comm_rank % num_devices;
		log("found %d device(s), picking device %d", num_devices, device);
	}
	CUDA_CHECK(cudaSetDevice, device);

	FILE* csv = nullptr;
	if(comm_rank == 0) {
		char csv_name[100];
		const char* job_id = getenv("SLURM_JOB_ID");
		if(job_id) {
			snprintf(csv_name, sizeof csv_name, "nbody-%d-%s.csv", comm_size, job_id);
		} else {
			snprintf(csv_name, sizeof csv_name, "nbody-%d.csv", comm_size);
		}
		csv = fopen(csv_name, "wb");
		if(!csv) panic("error opening output file: %s: %s", csv_name, strerror(errno));
		fprintf(csv, "devices;benchmark;range;system;configuration;precision;iterations;nanoseconds\n");
		fflush(csv);
	}

	const int n = (argc > 1) ? atoi(argv[1]) : 262144;
	const int nIters = (argc > 2) ? atoi(argv[2]) : 10; // simulation iterations

	log("Block size %d, data type: %s.\n", BLOCK_SIZE, USE_FLOAT == 1 ? "float" : "double");

	benchmark<device_allgather<p2p_host_allgather>>(csv, comm_rank, comm_size, n, nIters);
	benchmark<device_allgather<collective_host_allgather>>(csv, comm_rank, comm_size, n, nIters);

	if(comm_rank == 0) fclose(csv);

#if TRACY_ENABLE
	tracy::ShutdownProfiler();
#endif

	MPI_Finalize();
}
