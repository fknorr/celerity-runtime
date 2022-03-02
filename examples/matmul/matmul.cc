#include <cstdio>

#include <celerity.h>

const size_t MAT_SIZE = 1024;

template <typename T>
void set_identity(celerity::distr_queue& queue, celerity::buffer<T, 2> mat) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class set_identity_kernel>(mat.get_range(), [=](celerity::item<2> item) { dw[item] = item[0] == item[1]; });
	});
}

template <typename T>
void multiply(celerity::distr_queue& queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

		// Use local-memory tiling to avoid waiting on global memory too often
		// Note: We assume a local range size of 64 here, this should be supported by most devices.
		const size_t GROUP_SIZE = 8;
		celerity::local_accessor<T, 2> scratch_a{{GROUP_SIZE, GROUP_SIZE}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{GROUP_SIZE, GROUP_SIZE}, cgh};

		cgh.parallel_for<class mat_mul>(celerity::nd_range<2>{{MAT_SIZE, MAT_SIZE}, {GROUP_SIZE, GROUP_SIZE}}, [=](celerity::nd_item<2> item) {
			T sum{};
			const auto lid = item.get_local_id();
			for(size_t j = 0; j < MAT_SIZE; j += GROUP_SIZE) {
				scratch_a[lid] = a[item.get_group(0) * GROUP_SIZE + lid[0]][j + lid[1]];
				scratch_b[lid] = b[j + lid[0]][item.get_group(1) * GROUP_SIZE + lid[1]];
				celerity::group_barrier(item.get_group());

				for(size_t k = 0; k < GROUP_SIZE; ++k) {
					const auto a_ik = scratch_a[lid[0]][k];
					const auto b_kj = scratch_b[k][lid[1]];
					sum += a_ik * b_kj;
				}
				celerity::group_barrier(item.get_group());
			}
			c[item.get_global_id()] = sum;
		});
	});
}

template <typename T>
void verify(celerity::distr_queue& queue, celerity::buffer<T, 2> mat_a_buf, celerity::experimental::host_object<bool> verification) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor result{mat_a_buf, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};
		celerity::experimental::side_effect passed{verification, cgh};

		cgh.host_task(mat_a_buf.get_range(), [=](celerity::partition<2> part) {
			*passed = [&] {
				auto sr = part.get_subrange();
				for(size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i) {
					for(size_t j = sr.offset[0]; j < sr.offset[0] + sr.range[0]; ++j) {
						const float received = result[{i, j}];
						const float expected = i == j;
						if(expected != received) {
							fprintf(stderr, "VERIFICATION FAILED for element %zu,%zu: %f (received) != %f (expected)\n", i, j, received, expected);
							return false;
						}
					}
				}
				printf("VERIFICATION PASSED!\n");
				return true;
			}();
		});
	});
}

int main() {
	celerity::distr_queue queue;

	const auto range = celerity::range<2>(MAT_SIZE, MAT_SIZE);
	celerity::buffer<float, 2> mat_a_buf(range);
	celerity::buffer<float, 2> mat_b_buf(range);
	celerity::buffer<float, 2> mat_c_buf(range);

	set_identity(queue, mat_a_buf);
	set_identity(queue, mat_b_buf);

	multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
	multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

	celerity::experimental::host_object<bool> verification;
	verify(queue, mat_a_buf, verification);

	const auto passed = celerity::experimental::drain(std::move(queue), celerity::experimental::capture{verification});
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
