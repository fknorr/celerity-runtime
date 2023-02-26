#include <celerity.h>

#include <chrono>
#include <stdio.h>
#include <vector>
using namespace std::chrono_literals;

#include "utils/wyhash.h"

#ifndef USE_TILED
#define USE_TILED 1
#endif
#ifndef USE_FLOAT
#define USE_FLOAT 1
#endif

#if USE_FLOAT == 1
#define VAL_TYPE float
#define VAL4_TYPE sycl::float4
#else
#define VAL_TYPE double
#define VAL4_TYPE sycl::double4
#endif

constexpr int BLOCK_SIZE = 256;
constexpr VAL_TYPE SOFTENING = 1e-9f;
constexpr VAL_TYPE DT = 0.01f;

typedef struct {
	VAL4_TYPE *pos, *vel;
} BodySystem;

void randomizeBodies(VAL_TYPE* data, int n) {
	unsigned int seed = 42;
	for(int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand_r(&seed) / (VAL_TYPE)RAND_MAX) - 1.0f;
	}
}

// there is a significant speed difference between the native cuda rsqrt
// and the current implementation in HipSYCL. Allow optional access to native impl.
#ifndef DIRECTLY_USE_CUDA_RSQRT
#define DIRECTLY_USE_CUDA_RSQRT 1
#endif

#if DIRECTLY_USE_CUDA_RSQRT == 1
#include <__clang_cuda_math.h>
#ifdef USE_FLOAT
#define r_sqrt(__v) rsqrtf(__v)
#else
#define r_sqrt(__v) rsqrt(__v)
#endif
#else // DIRECTLY_USE_CUDA_RSQRT == 1
#define r_sqrt(__v) sycl::rsqrt(__v)
#endif // DIRECTLY_USE_CUDA_RSQRT == 1

void print_system(const BodySystem& p, int nBodies) {
	for(int i = 0; i < nBodies; ++i) {
		printf("%8.4f %8.4f %8.4f / %8.4f %8.4f %8.4f\n", p.pos[i].x(), p.pos[i].y(), p.pos[i].z(), p.vel[i].x(), p.vel[i].y(), p.vel[i].z());
	}
}

#if USE_TILED == 1
void bodyForce(celerity::distr_queue& queue, celerity::buffer<VAL4_TYPE, 1>& posBuff, celerity::buffer<VAL4_TYPE, 1>& velBuff) {
	int nBodies = posBuff.get_range()[0];
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor readPos{posBuff, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor rwVel{velBuff, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::local_accessor<VAL4_TYPE> localMem(BLOCK_SIZE, cgh);
		cgh.parallel_for<class BodyForce>(celerity::nd_range<1>(nBodies, BLOCK_SIZE), [=](celerity::nd_item<1> item) {
			const size_t i = item.get_global_id(0);
			const size_t threadIdx = item.get_local_id(0);
			const size_t gridDimx = item.get_group_range(0);
			if(i < nBodies) {
				VAL_TYPE Fx = 0.0f;
				VAL_TYPE Fy = 0.0f;
				VAL_TYPE Fz = 0.0f;

				for(size_t tile = 0; tile < gridDimx; tile++) {
					localMem[threadIdx] = readPos[tile * BLOCK_SIZE + threadIdx];
					celerity::group_barrier(item.get_group());

					for(size_t j = 0; j < BLOCK_SIZE; j++) {
						const VAL_TYPE dx = localMem[j].x() - readPos[i].x();
						const VAL_TYPE dy = localMem[j].y() - readPos[i].y();
						const VAL_TYPE dz = localMem[j].z() - readPos[i].z();
						const VAL_TYPE distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
						const VAL_TYPE invDist = r_sqrt(distSqr);
						const VAL_TYPE invDist3 = invDist * invDist * invDist;

						Fx += dx * invDist3;
						Fy += dy * invDist3;
						Fz += dz * invDist3;
					}
					celerity::group_barrier(item.get_group());
				}

				rwVel[i].x() += DT * Fx;
				rwVel[i].y() += DT * Fy;
				rwVel[i].z() += DT * Fz;
			}
		});
	});
}
#else  // USE_TILED == 1
void bodyForce(celerity::distr_queue& queue, celerity::buffer<VAL4_TYPE, 1>& posBuff, celerity::buffer<VAL4_TYPE, 1>& velBuff) {
	int nBodies = posBuff.get_range()[0];
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor readPos{posBuff, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor rwVel{velBuff, cgh, celerity::access::one_to_one{}, celerity::read_write};
		cgh.parallel_for<class BodyForce>(celerity::nd_range<1>(nBodies, BLOCK_SIZE), [=](celerity::nd_item<1> item) {
			const size_t i = item.get_global_id(0);
			if(i < nBodies) {
				VAL_TYPE Fx = 0.0f;
				VAL_TYPE Fy = 0.0f;
				VAL_TYPE Fz = 0.0f;

				for(size_t j = 0; j < nBodies; j++) {
					const VAL_TYPE dx = readPos[j].x() - readPos[i].x();
					const VAL_TYPE dy = readPos[j].y() - readPos[i].y();
					const VAL_TYPE dz = readPos[j].z() - readPos[i].z();
					const VAL_TYPE distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
					const VAL_TYPE invDist = r_sqrt(distSqr);
					const VAL_TYPE invDist3 = invDist * invDist * invDist;

					Fx += dx * invDist3;
					Fy += dy * invDist3;
					Fz += dz * invDist3;
				}

				rwVel[i].x() += DT * Fx;
				rwVel[i].y() += DT * Fy;
				rwVel[i].z() += DT * Fz;
			}
		});
	});
}
#endif // USE_TILED == 1

int main(const int argc, const char** argv) {
	const int nBodies = (argc > 1) ? atoi(argv[1]) : 262144;
	const int nIters = (argc > 2) ? atoi(argv[2]) : 10; // simulation iterations
	const bool debug = false && nBodies <= 32;

	printf("Tiling %s, block size %d, data type: %s.\n", USE_TILED == 1 ? "enabled" : "disabled", BLOCK_SIZE, USE_FLOAT == 1 ? "float" : "double");

	celerity::distr_queue queue;

	const int total_num_devices =
	    celerity::detail::runtime::get_instance().get_local_devices().num_compute_devices() * celerity::detail::runtime::get_instance().get_num_nodes();

	celerity::experimental::host_object<FILE*> csv_obj;
	queue.submit([=](celerity::handler& cgh) {
		celerity::experimental::side_effect csv(csv_obj, cgh);
		cgh.host_task(celerity::on_master_node, [=] {
			char csv_name[100];
			const char* job_id = getenv("SLURM_JOB_ID");
			if(job_id) {
				snprintf(csv_name, sizeof csv_name, "celerity-nbody-%d-%s.csv", total_num_devices, job_id);
			} else {
				snprintf(csv_name, sizeof csv_name, "celerity-nbody-%d.csv", total_num_devices);
			}
			*csv = fopen(csv_name, "wb");
			if(!*csv) {
				fprintf(stderr, "error opening output file: %s: %s\n", csv_name, strerror(errno));
				abort();
			}
			fprintf(*csv, "devices;benchmark;range;system;configuration;nanoseconds\n");
			fflush(*csv);
		});
	});

	const int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

	const int n_warmup = 2;
	const int n_passes = 10;

	for(const bool collectives : {false, true}) {
		celerity::detail::task_manager::NOMERGE_generate_collectives = collectives;
		const auto configuration = collectives ? "collectives" : "p2p";

		const int bytes = nBodies * sizeof(VAL4_TYPE) * 2;
		std::vector<char> h_buf(bytes);
		VAL_TYPE* const buf = (VAL_TYPE*)(h_buf.data());
		const BodySystem p = {(VAL4_TYPE*)buf, ((VAL4_TYPE*)buf) + nBodies};

		randomizeBodies(buf, 8 * nBodies); // init pos / vel data
		if(debug) print_system(p, nBodies);

		for(int i = 0; i < n_warmup + n_passes; ++i) {
			celerity::buffer<VAL4_TYPE, 1> posBuff(p.pos, nBodies);
			celerity::buffer<VAL4_TYPE, 1> velBuff(p.vel, nBodies);
			queue.slow_full_sync();

			const auto start = std::chrono::steady_clock::now();
			for(int iter = 1; iter <= nIters; iter++) {
				bodyForce(queue, posBuff, velBuff);

				queue.submit([=](celerity::handler& cgh) {
					celerity::accessor rwPos{posBuff, cgh, celerity::access::one_to_one{}, celerity::read_write};
					celerity::accessor readVel{velBuff, cgh, celerity::access::one_to_one{}, celerity::read_only};
					cgh.parallel_for<class BodyPosUpdate>(celerity::range<1>(nBodies), [=](celerity::item<1> i) {
						rwPos[i].x() += readVel[i].x() * DT;
						rwPos[i].y() += readVel[i].y() * DT;
						rwPos[i].z() += readVel[i].z() * DT;
					});
				});
			}
			queue.slow_full_sync();
			const auto end = std::chrono::steady_clock::now();

			queue.submit([=](celerity::handler& cgh) {
				celerity::experimental::side_effect csv(csv_obj, cgh);
				cgh.host_task(celerity::on_master_node, [=] {
					const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
					fprintf(*csv, "%d;%s;%d;MPI+CUDA;%s;%llu\n", total_num_devices, "NBody", nBodies, configuration, (unsigned long long)ns.count());
					fflush(*csv);
				});
			});

			if(i == n_warmup + n_passes - 1) {
				queue.submit([=](celerity::handler& cgh) {
					celerity::accessor readPos{posBuff, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor readVel{velBuff, cgh, celerity::access::all{}, celerity::read_only_host_task};
					cgh.host_task(celerity::on_master_node, [=] {
						double sum = 0.0;
						for(int i = 0; i < nBodies; ++i) {
							p.pos[i] = readPos[i];
							sum += p.pos[i].x() + p.pos[i].y() + p.pos[i].z();
							p.vel[i] = readVel[i];
						}
						printf("%12s result hash: %16lX, position avg: %15.8f\n", configuration, wyhash(buf, bytes, 0, _wyp), sum / nBodies);
						if(debug) print_system(p, nBodies);
					});
				});
			}

			// TODO: workaround for current experimental Celerity branch, remove when no longer needed
			queue.slow_full_sync();
		}
	}

	queue.submit([=](celerity::handler& cgh) {
		celerity::experimental::side_effect csv(csv_obj, cgh);
		cgh.host_task(celerity::on_master_node, [=] { //
			fclose(*csv);
		});
	});

	// TODO: workaround for current experimental Celerity branch, remove when no longer needed
	queue.slow_full_sync();
}
