#include <celerity.h>

#include <chrono>
#include <stdio.h>
#include <vector>
using namespace std::chrono_literals;

#include "utils/wyhash.h"

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
constexpr int WARP_SIZE = 32;
constexpr VAL_TYPE SOFTENING = 1e-9f;
constexpr VAL_TYPE DT = 0.01f;

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

struct n_to_one {
	size_t n;
	celerity::subrange<1> operator()(const celerity::chunk<1>& ck) const { return {ck.offset[0] / n, ck.range[0] / n}; }
};

void bodyForce(celerity::distr_queue& queue, celerity::buffer<VAL_TYPE, 1>& pos_x_buf, celerity::buffer<VAL_TYPE, 1>& pos_y_buf,
    celerity::buffer<VAL_TYPE, 1>& pos_z_buf, celerity::buffer<VAL_TYPE, 1>& vel_x_buf, celerity::buffer<VAL_TYPE, 1>& vel_y_buf,
    celerity::buffer<VAL_TYPE, 1>& vel_z_buf) {
	const int n_bodies = pos_x_buf.get_range()[0];
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor pos_x{pos_x_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor pos_y{pos_y_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor pos_z{pos_z_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor vel_x{vel_x_buf, cgh, n_to_one{WARP_SIZE}, celerity::read_write};
		celerity::accessor vel_y{vel_y_buf, cgh, n_to_one{WARP_SIZE}, celerity::read_write};
		celerity::accessor vel_z{vel_z_buf, cgh, n_to_one{WARP_SIZE}, celerity::read_write};
		celerity::local_accessor<VAL_TYPE[BLOCK_SIZE]> stage(3, cgh);
		cgh.parallel_for<class BodyForce>(celerity::nd_range<1>(n_bodies * WARP_SIZE, BLOCK_SIZE), [=](celerity::nd_item<1> item) {
			const int input_tid = item.get_local_id(0);
			const int output_item = item.get_global_id(0) / WARP_SIZE;
			const int output_tid = item.get_local_id(0) % WARP_SIZE;

			const auto out_pos_x = pos_x[output_item];
			const auto out_pos_y = pos_y[output_item];
			const auto out_pos_z = pos_z[output_item];

			if(output_item < n_bodies) {
				VAL_TYPE force_x = 0.0f;
				VAL_TYPE force_y = 0.0f;
				VAL_TYPE force_z = 0.0f;

				for(int input_item = input_tid; input_item < n_bodies; input_item += BLOCK_SIZE) {
					stage[0][input_tid] = pos_x[input_item];
					stage[1][input_tid] = pos_y[input_item];
					stage[2][input_tid] = pos_z[input_item];
					celerity::group_barrier(item.get_group());

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
					celerity::group_barrier(item.get_group());
				}

				force_x = sycl::reduce_over_group(item.get_sub_group(), force_x, sycl::plus<VAL_TYPE>());
				force_y = sycl::reduce_over_group(item.get_sub_group(), force_y, sycl::plus<VAL_TYPE>());
				force_z = sycl::reduce_over_group(item.get_sub_group(), force_z, sycl::plus<VAL_TYPE>());

				if(output_tid == 0) {
					vel_x[output_item] += DT * force_x;
					vel_y[output_item] += DT * force_y;
					vel_z[output_item] += DT * force_z;
				}
			}
		});
	});
}

void bodyPos(celerity::distr_queue& queue, celerity::buffer<VAL_TYPE, 1>& pos_x_buf, celerity::buffer<VAL_TYPE, 1>& pos_y_buf,
		celerity::buffer<VAL_TYPE, 1>& pos_z_buf, celerity::buffer<VAL_TYPE, 1>& vel_x_buf, celerity::buffer<VAL_TYPE, 1>& vel_y_buf,
		celerity::buffer<VAL_TYPE, 1>& vel_z_buf) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor pos_x{pos_x_buf, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor pos_y{pos_y_buf, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor pos_z{pos_z_buf, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor vel_x{vel_x_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor vel_y{vel_y_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor vel_z{vel_z_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		cgh.parallel_for<class BodyPos>(pos_x_buf.get_range(), [=](celerity::item<1> i) {
			pos_x[i] += vel_x[i] * DT;
			pos_y[i] += vel_y[i] * DT;
			pos_z[i] += vel_z[i] * DT;
		});
	});
}

int main(const int argc, const char** argv) {
	const int nBodies = (argc > 1) ? atoi(argv[1]) : 262144;
	const int nIters = (argc > 2) ? atoi(argv[2]) : 10; // simulation iterations
	const bool debug = false && nBodies <= 32;

	printf("Block size %d, data type: %s.\n", BLOCK_SIZE, USE_FLOAT == 1 ? "float" : "double");

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

		const size_t buf_count = 2 * 3 * nBodies;
		std::vector<VAL_TYPE> h_buf(buf_count);
		VAL_TYPE* const buf = h_buf.data();
		randomizeBodies(buf, buf_count); // init pos / vel data

		for(int i = 0; i < n_warmup + n_passes; ++i) {
			celerity::buffer<VAL_TYPE, 1> pos_x_buf(buf + 0 * nBodies, nBodies);
			celerity::buffer<VAL_TYPE, 1> pos_y_buf(buf + 1 * nBodies, nBodies);
			celerity::buffer<VAL_TYPE, 1> pos_z_buf(buf + 2 * nBodies, nBodies);
			celerity::buffer<VAL_TYPE, 1> vel_x_buf(buf + 3 * nBodies, nBodies);
			celerity::buffer<VAL_TYPE, 1> vel_y_buf(buf + 4 * nBodies, nBodies);
			celerity::buffer<VAL_TYPE, 1> vel_z_buf(buf + 5 * nBodies, nBodies);
			queue.slow_full_sync();

			const auto start = std::chrono::steady_clock::now();
			for(int iter = 1; iter <= nIters; iter++) {
				bodyForce(queue, pos_x_buf, pos_y_buf, pos_z_buf, vel_x_buf, vel_y_buf, vel_z_buf);
				bodyPos(queue, pos_x_buf, pos_y_buf, pos_z_buf, vel_x_buf, vel_y_buf, vel_z_buf);
			}
			queue.slow_full_sync();
			const auto end = std::chrono::steady_clock::now();

			queue.submit([=](celerity::handler& cgh) {
				celerity::experimental::side_effect csv(csv_obj, cgh);
				cgh.host_task(celerity::on_master_node, [=] {
					const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
					fprintf(*csv, "%d;%s;%d;Celerity;%s;%llu\n", total_num_devices, "NBody", nBodies, configuration, (unsigned long long)ns.count());
					fflush(*csv);
				});
			});

			if(i == n_warmup + n_passes - 1) {
				queue.submit([=](celerity::handler& cgh) {
					celerity::accessor pos_x{pos_x_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor pos_y{pos_y_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor pos_z{pos_z_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor vel_x{vel_x_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor vel_y{vel_y_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					celerity::accessor vel_z{vel_z_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
					cgh.host_task(celerity::on_master_node, [=] {
						double sum = 0.0;
						for(int i = 0; i < nBodies; ++i) {
							sum += pos_x[i] + pos_y[i] + pos_z[i];
							buf[6 * i + 0] = pos_x[i];
							buf[6 * i + 1] = pos_y[i];
							buf[6 * i + 2] = pos_z[i];
							buf[6 * i + 3] = vel_x[i];
							buf[6 * i + 4] = vel_y[i];
							buf[6 * i + 5] = vel_z[i];
						}
						const auto hash = wyhash(buf, buf_count * sizeof(VAL_TYPE), 0, _wyp);
						printf("%12s result hash: %16lX, position avg: %15.8f\n", configuration, hash, sum / nBodies);
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
