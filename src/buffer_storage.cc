#include "buffer_storage.h"

namespace celerity {
namespace detail {

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<1>& source_range,
	    const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range, const cl::sycl::id<1>& target_offset,
	    const cl::sycl::range<1>& copy_range) {
		const size_t line_size = elem_size * copy_range[0];
		std::memcpy(static_cast<std::byte*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
		    static_cast<const std::byte*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size);
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<2>& source_range,
	    const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range, const cl::sycl::id<2>& target_offset,
	    const cl::sycl::range<2>& copy_range) {
		const size_t line_size = elem_size * copy_range[1];
		const auto source_base_offset = get_linear_index(source_range, source_offset);
		const auto target_base_offset = get_linear_index(target_range, target_offset);
		for(size_t i = 0; i < copy_range[0]; ++i) {
			std::memcpy(static_cast<std::byte*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
			    static_cast<const std::byte*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		}
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<3>& source_range,
	    const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& target_offset,
	    const cl::sycl::range<3>& copy_range) {
		// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
		const auto source_base_offset = get_linear_index(source_range, source_offset)
		                                - get_linear_index(cl::sycl::range<2>{source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
		const auto target_base_offset = get_linear_index(target_range, target_offset)
		                                - get_linear_index(cl::sycl::range<2>{target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});
		for(size_t i = 0; i < copy_range[0]; ++i) {
			const auto source_ptr = static_cast<const std::byte*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
			const auto target_ptr = static_cast<std::byte*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
			memcpy_strided(source_ptr, target_ptr, elem_size, cl::sycl::range<2>{source_range[1], source_range[2]}, {source_offset[1], source_offset[2]},
			    {target_range[1], target_range[2]}, {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]});
		}
	}

	// NOCOMMIT Copy pasta of host variant. Unify with above.
	// NOCOMMIT Don't wait here - return event (needs solution for 2D/3D with multiple events!)
	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<1>& source_range, const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range,
	    const cl::sycl::id<1>& target_offset, const cl::sycl::range<1>& copy_range) {
		const size_t line_size = elem_size * copy_range[0];
#if defined(__HIPSYCL__)
		const auto ret = cudaMemcpy(reinterpret_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
		    reinterpret_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault);
		if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy2D failed");
		// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
		cudaStreamSynchronize(0);
#else
		queue
		    .memcpy(reinterpret_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
		        reinterpret_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size)
		    .wait();
#endif
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<2>& source_range, const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range,
	    const cl::sycl::id<2>& target_offset, const cl::sycl::range<2>& copy_range) {
		const auto source_base_offset = get_linear_index(source_range, source_offset);
		const auto target_base_offset = get_linear_index(target_range, target_offset);

// NOCOMMIT Move into backend-specific module
#if defined(__HIPSYCL__)
		const auto ret = cudaMemcpy2D(reinterpret_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
		    reinterpret_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size,
		    copy_range[0], cudaMemcpyDefault);
		if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy2D failed");
		// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
		cudaStreamSynchronize(0);
#else
		const size_t line_size = elem_size * copy_range[1];
		std::vector<cl::sycl::event> wait_list;
		wait_list.reserve(copy_range[0]);
		for(size_t i = 0; i < copy_range[0]; ++i) {
			auto e = queue.memcpy(reinterpret_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
			    reinterpret_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
			wait_list.push_back(e);
		}
		for(auto& e : wait_list) {
			e.wait();
		}
#endif
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<3>& source_range, const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range,
	    const cl::sycl::id<3>& target_offset, const cl::sycl::range<3>& copy_range) {
// NOCOMMIT Move into backend-specific module
#if defined(__HIPSYCL__)
		// NOCOMMIT TODO This needs thorough testing. I don't think current unit tests exercise strided 3D copies much (if at all)
		cudaMemcpy3DParms parms = {};
		parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
		parms.srcPtr = make_cudaPitchedPtr(const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]);
		parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
		parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
		parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
		parms.kind = cudaMemcpyDefault;
		const auto ret = cudaMemcpy3D(&parms);
		if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy3D failed");
		// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
		cudaStreamSynchronize(0);
#else
		// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
		const auto source_base_offset =
		    get_linear_index(source_range, source_offset) - get_linear_index({source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
		const auto target_base_offset =
		    get_linear_index(target_range, target_offset) - get_linear_index({target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});

		for(size_t i = 0; i < copy_range[0]; ++i) {
			const auto source_ptr = reinterpret_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
			const auto target_ptr = reinterpret_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
			memcpy_strided_device(queue, source_ptr, target_ptr, elem_size, {source_range[1], source_range[2]}, {source_offset[1], source_offset[2]},
			    {target_range[1], target_range[2]}, {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]});
		}
#endif
	}

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr) {
		assert((id_cast<3>(copy_sr.offset) < id_cast<3>(source_range)) == cl::sycl::id<3>(1, 1, 1));
		assert((id_cast<3>(copy_sr.offset + copy_sr.range) <= id_cast<3>(source_range)) == cl::sycl::id<3>(1, 1, 1));

		if(source_range[2] == 1) {
			if(source_range[1] == 1) {
				memcpy_strided(source_base_ptr, target_ptr, elem_size, range_cast<1>(source_range), range_cast<1>(copy_sr.offset), range_cast<1>(copy_sr.range),
				    cl::sycl::id<1>(0), range_cast<1>(copy_sr.range));
			} else {
				memcpy_strided(source_base_ptr, target_ptr, elem_size, range_cast<2>(source_range), range_cast<2>(copy_sr.offset), range_cast<2>(copy_sr.range),
				    cl::sycl::id<2>(0, 0), range_cast<2>(copy_sr.range));
			}
		} else {
			memcpy_strided(
			    source_base_ptr, target_ptr, elem_size, range_cast<3>(source_range), copy_sr.offset, copy_sr.range, cl::sycl::id<3>(0, 0, 0), copy_sr.range);
		}
	}

} // namespace detail
} // namespace celerity
