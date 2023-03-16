#pragma once

#include <cassert>
#include <cstring>
#include <memory>

#include <CL/sycl.hpp>
#include <gch/small_vector.hpp>

// clang-format off
#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                               \
	if (const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {      \
		CELERITY_CRITICAL(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));   \
		abort();                                                                                  \
	}
// clang-format on

#define USE_NDVBUFFER 0

// TODO: Works for now, but really needs to be a runtime switch depending on selected device
#if !defined(USE_NDVBUFFER) && defined(__HIPSYCL__) && defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)
#define USE_NDVBUFFER 1
#include "ndvbuffer.h"
#else
#define USE_NDVBUFFER 0
#endif

#include "device_queue.h"
#include "payload.h"
#include "ranges.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	class native_event_wrapper {
	  public:
		virtual ~native_event_wrapper() = default;
		virtual bool is_done() const = 0;
		// virtual void wait() = 0;
	};

	class sycl_event_wrapper final : public native_event_wrapper {
	  public:
		sycl_event_wrapper(sycl::event evt) : m_event(std::move(evt)) {}

		bool is_done() const override { return m_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; }
		// void wait() override { m_event.wait(); }

	  private:
		sycl::event m_event;
	};

#if defined(__HIPSYCL__)
	inline cudaEvent_t create_and_record_cuda_event(cudaStream_t stream) {
		// TODO: Perf considerations - we should probably have an event pool
		cudaEvent_t result;
		CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &result, cudaEventDisableTiming);
		CELERITY_CUDA_CHECK(cudaEventRecord, result, stream);
		return result;
	}

	class cuda_event_wrapper final : public native_event_wrapper {
	  public:
		cuda_event_wrapper(cudaEvent_t evt, cudaStream_t stream) : m_event(evt) {}
		~cuda_event_wrapper() { CELERITY_CUDA_CHECK(cudaEventDestroy, m_event); }

		bool is_done() const override {
			const auto ret = cudaEventQuery(m_event);
			if(ret != cudaSuccess && ret != cudaErrorNotReady) {
				CELERITY_CRITICAL("cudaEventQuery: {}", cudaGetErrorString(ret));
				abort();
			}
			return ret == cudaSuccess;
		}

	  private:
		cudaEvent_t m_event;
	};
#endif

	// TODO: Naming: Future, Promise, ..?
	// FIXME: We probably want this to be copyable, right..? (Currently not possible due to payload attachment hack)
	class [[nodiscard]] async_event {
	  public:
		async_event() = default;
		async_event(const async_event&) = delete;
		async_event(async_event&&) = default;
		async_event(std::shared_ptr<native_event_wrapper> native_event) { add(std::move(native_event)); }

		async_event& operator=(async_event&&) = default;

		void merge(async_event other) {
			for(size_t i = 0; i < other.m_native_events.size(); ++i) {
				m_done_cache.push_back(other.m_done_cache[i]);
				m_native_events.emplace_back(std::move(other.m_native_events[i]));
			}
			for(auto& p : other.m_attached_payloads) {
				m_attached_payloads.emplace_back(std::move(p));
			}
		}

		void add(std::shared_ptr<native_event_wrapper> native_event) {
			m_done_cache.push_back(false);
			m_native_events.emplace_back(std::move(native_event));
		}

		bool is_done() const {
			for(size_t i = 0; i < m_native_events.size(); ++i) {
				if(!m_done_cache[i]) {
					const bool is_done = m_native_events[i]->is_done();
					if(is_done) {
						m_done_cache[i] = true;
						continue;
					}
					return false;
				}
			}
			return true;
		}

		void wait() const {
			while(!is_done()) {}
		}

		// FIXME: Workaround to extend lifetime of temporary staging copies for asynchronous transfers
		void hack_attach_payload(shared_payload_ptr ptr) { m_attached_payloads.emplace_back(std::move(ptr)); }

	  private:
		mutable gch::small_vector<bool> m_done_cache;
		gch::small_vector<std::shared_ptr<native_event_wrapper>> m_native_events;
		// FIXME: For some reason (old libstdc++?) this doesn't compile as a gch::small_vector on Marconi-100...
		std::vector<shared_payload_ptr> m_attached_payloads;
	};

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<1>& source_range,
	    const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range, const cl::sycl::id<1>& target_offset,
	    const cl::sycl::range<1>& copy_range);

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<2>& source_range,
	    const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range, const cl::sycl::id<2>& target_offset,
	    const cl::sycl::range<2>& copy_range);

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<3>& source_range,
	    const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& target_offset,
	    const cl::sycl::range<3>& copy_range);

	// NOCOMMIT ONLY TEMPORARY
	// SYCL 2020 Provisional doesn't include any strided overloads for memcpy, so much like on the host, we have to roll our own.
	// TODO: Review once SYCL 2020 final has been released.
	// NOCOMMIT Copy pasta of host variant. Unify with above.
	async_event memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<1>& source_range, const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range,
	    const cl::sycl::id<1>& target_offset, const cl::sycl::range<1>& copy_range, device_id did, cudaStream_t stream);

	async_event memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<2>& source_range, const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range,
	    const cl::sycl::id<2>& target_offset, const cl::sycl::range<2>& copy_range, device_id did, cudaStream_t stream);

	async_event memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<3>& source_range, const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range,
	    const cl::sycl::id<3>& target_offset, const cl::sycl::range<3>& copy_range, device_id did, cudaStream_t stream);

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr);

#if !USE_NDVBUFFER
	template <typename DataT, int Dims>
	class device_buffer {
	  public:
		device_buffer(const range<Dims>& range, device_queue& queue) : m_range(range), m_queue(queue) {
			if(m_range.size() != 0) { m_device_allocation = m_queue.malloc<DataT>(m_range.size()); }
		}

		~device_buffer() { m_queue.free(std::move(m_device_allocation)); }

		device_buffer(const device_buffer&) = delete;

		range<Dims> get_range() const { return m_range; }

		DataT* get_pointer() { return static_cast<DataT*>(m_device_allocation.ptr); }

		const DataT* get_pointer() const { return static_cast<DataT*>(m_device_allocation.ptr); }

		bool operator==(const device_buffer& rhs) const {
			return m_device_allocation == rhs.m_device_allocation && m_queue == rhs.m_queue && m_range == rhs.m_range;
		}

	  private:
		sycl::range<Dims> m_range;
		device_queue& m_queue;
		device_allocation m_device_allocation;
	};
#endif

	template <typename DataT, int Dims>
	class host_buffer {
	  public:
		explicit host_buffer(cl::sycl::range<Dims> range) : m_range(range) {
			auto r3 = range_cast<3>(range);
			m_data = std::make_unique<DataT[]>(r3[0] * r3[1] * r3[2]);
		}

		cl::sycl::range<Dims> get_range() const { return m_range; };

		DataT* get_pointer() { return m_data.get(); }

		const DataT* get_pointer() const { return m_data.get(); }

		bool operator==(const host_buffer& rhs) const { return m_data.get() == rhs.m_data.get(); }

	  private:
		cl::sycl::range<Dims> m_range;
		std::unique_ptr<DataT[]> m_data;
	};

	enum class buffer_type { device_buffer, host_buffer };

	class buffer_storage {
	  public:
		/**
		 * @param range The size of the buffer
		 */
		buffer_storage(celerity::range<3> range, buffer_type type) : m_range(range), m_type(type) {}

		celerity::range<3> get_range() const { return m_range; }

		buffer_type get_type() const { return m_type; }

		/**
		 * Returns the buffer size, in bytes.
		 */
		virtual size_t get_size() const = 0;

		virtual void* get_pointer() = 0;

		virtual const void* get_pointer() const = 0;

		// TODO: This is just a mockup of what a backend-specific integration of ndvbuffer might look like
		// TODO: Naming - this should signal two things: Buffer can be resized in-place, and supports sparse backing allocations
		virtual bool supports_dynamic_allocation() const { return false; }

		virtual void allocate(const subrange<3>& sr) { assert(supports_dynamic_allocation()); }

		virtual async_event get_data(const subrange<3>& sr, void* out_linearized) const = 0;

		virtual async_event set_data(
		    const void* in_base_ptr, const range<3>& in_range, const id<3>& in_offset, const id<3>& local_offset, const range<3>& copy_range) = 0;

		/**
		 * Copy data from the given source buffer into this buffer.
		 */
		virtual async_event copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) = 0;

		// FIXME Just hacking - this assumes source has same dimensionality
		// FIXME: Need to pass SYCL queue for copying to host... ugh
		virtual async_event copy_from_device_raw(sycl::queue& q, void* source_ptr, const range<3>& source_range, const id<3>& source_offset,
		    const id<3>& target_offset, const range<3>& copy_range, device_id did, cudaStream_t stream) = 0;

		virtual ~buffer_storage() = default;

	  private:
		cl::sycl::range<3> m_range;
		buffer_type m_type;
	};

	// FIXME: Remove this
	template <typename DataT, int Dims>
	class computecpp_get_data_workaround {};
	template <typename DataT, int Dims>
	class computecpp_set_data_workaround {};

	template <typename DataT, int Dims>
	class device_buffer_storage : public buffer_storage {
	  public:
		device_buffer_storage(range<Dims> range, device_queue& owning_queue, cudaStream_t copy_stream)
		    : buffer_storage(range_cast<3>(range), buffer_type::device_buffer), m_owning_queue(owning_queue.get_sycl_queue()),
#if USE_NDVBUFFER
		      m_device_buf(sycl::get_native<sycl::backend::cuda>(m_owning_queue.get_device()), ndv::extent<Dims>::make_from(range))
#else
		      m_device_buf(range, owning_queue)
#endif
		      ,
		      m_did(owning_queue.get_id()), m_copy_stream(copy_stream) {
		}

		~device_buffer_storage() {
#if USE_NDVBUFFER
			CELERITY_DEBUG("Destroying ndvbuffer. Total allocation size: {} bytes.\n", m_device_buf.get_allocated_size());
#endif
		}

		sycl::queue& get_owning_queue() const { return m_owning_queue; }

		// FIXME: This is no longer accurate for (sparsely allocated) ndv buffers (only an upper bound).
		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		void* get_pointer() override { return m_device_buf.get_pointer(); }

		const void* get_pointer() const override { return m_device_buf.get_pointer(); }

		bool supports_dynamic_allocation() const override { return USE_NDVBUFFER; }

#if USE_NDVBUFFER
		void allocate(const subrange<3>& sr) override {
			m_device_buf.access({ndv::point<Dims>::make_from(sr.offset), ndv::point<Dims>::make_from(sr.offset + sr.range)});
		}
#endif

		async_event get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			assert((id_cast<Dims>(sr.offset) + range_cast<Dims>(sr.range) <= m_device_buf.get_range()) == range_cast<Dims>(range<3>{true, true, true}));

#if USE_NDVBUFFER
			const ndv::box<Dims> src_box = {ndv::point<Dims>::make_from(sr.offset), ndv::point<Dims>::make_from(sr.offset + sr.range)};
			const ndv::box<Dims> dst_box = {{}, ndv::point<Dims>::make_from(sr.range)};
			m_device_buf.copy_to(static_cast<DataT*>(out_linearized), ndv::extent<Dims>::make_from(sr.range), src_box, dst_box);
			assert(false && "Figure out how to integrate with async_event");
#else
			return memcpy_strided_device(m_owning_queue, m_device_buf.get_pointer(), out_linearized, sizeof(DataT), m_device_buf.get_range(),
			    id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range), id<Dims>{}, range_cast<Dims>(sr.range), m_did, m_copy_stream);
#endif
		}

		async_event set_data(
		    const void* in_base_ptr, const range<3>& in_range, const id<3>& in_offset, const id<3>& local_offset, const range<3>& copy_range) override {
			assert(Dims > 1 || (in_offset[1] == 0 && in_range[1] == 1 && local_offset[1] == 0 && copy_range[1] == 1));
			assert(Dims > 2 || (in_offset[2] == 0 && in_range[2] == 1 && local_offset[2] == 0 && copy_range[2] == 1));
#if USE_NDVBUFFER
			const ndv::box<Dims> src_box = {{}, ndv::point<Dims>::make_from(sr.range)};
			const ndv::box<Dims> dst_box = {ndv::point<Dims>::make_from(sr.offset), ndv::point<Dims>::make_from(sr.offset + sr.range)};
			m_device_buf.copy_from(static_cast<const DataT*>(in_linearized), ndv::extent<Dims>::make_from(sr.range), src_box, dst_box);
			assert(false && "Figure out how to integrate with async_event");
#else
			// NOCOMMIT Use assert_copy_is_in_range from below?
			assert((id_cast<Dims>(local_offset) + range_cast<Dims>(copy_range) <= m_device_buf.get_range()) == range_cast<Dims>(range<3>{true, true, true}));
			return memcpy_strided_device(m_owning_queue, in_base_ptr, m_device_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(in_range),
			    range_cast<Dims>(in_offset), m_device_buf.get_range(), id_cast<Dims>(local_offset), range_cast<Dims>(copy_range), m_did, m_copy_stream);
#endif
		}

		async_event copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

		async_event copy_from_device_raw(sycl::queue& q, void* source_ptr, const range<3>& source_range, const id<3>& source_offset, const id<3>& target_offset,
		    const range<3>& copy_range, device_id did, cudaStream_t stream) override;

#if USE_NDVBUFFER
		// FIXME: Required for more efficient D->H copies (see host_buffer_storage::copy). Find cleaner API.
		const ndv::buffer<DataT, Dims>& get_ndv_buffer() const { return m_device_buf; }
#endif

	  private:
		mutable cl::sycl::queue m_owning_queue;
#if USE_NDVBUFFER
		ndv::buffer<DataT, Dims> m_device_buf;
#else
		device_buffer<DataT, Dims> m_device_buf;
#endif

		// NOCOMMIT HACK copy_from_device_raw() takes the stream by argument, but copy() doesn't, so we keep it as a member here
	  public:
		device_id m_did;
		cudaStream_t m_copy_stream;
	};

	template <typename DataT, int Dims>
	class host_buffer_storage : public buffer_storage {
	  public:
		explicit host_buffer_storage(cl::sycl::range<Dims> range) : buffer_storage(range_cast<3>(range), buffer_type::host_buffer), m_host_buf(range) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		void* get_pointer() override { return m_host_buf.get_pointer(); }

		const void* get_pointer() const override { return m_host_buf.get_pointer(); }

		async_event get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));

			memcpy_strided(m_host_buf.get_pointer(), out_linearized, sizeof(DataT), range_cast<Dims>(m_host_buf.get_range()), id_cast<Dims>(sr.offset),
			    range_cast<Dims>(sr.range), id_cast<Dims>(cl::sycl::id<3>{0, 0, 0}), range_cast<Dims>(sr.range));
			return async_event{};
		}

		async_event set_data(
		    const void* in_base_ptr, const range<3>& in_range, const id<3>& in_offset, const id<3>& local_offset, const range<3>& copy_range) override {
			assert(Dims > 1 || (in_offset[1] == 0 && in_range[1] == 1 && local_offset[1] == 0 && copy_range[1] == 1));
			assert(Dims > 2 || (in_offset[2] == 0 && in_range[2] == 1 && local_offset[2] == 0 && copy_range[2] == 1));
			assert((id_cast<Dims>(local_offset) + range_cast<Dims>(copy_range) <= m_host_buf.get_range()) == range_cast<Dims>(range<3>{true, true, true}));
			memcpy_strided(in_base_ptr, m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(in_range), range_cast<Dims>(in_offset),
			    m_host_buf.get_range(), id_cast<Dims>(local_offset), range_cast<Dims>(copy_range));
			return async_event{};
		}

		async_event copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

		async_event copy_from_device_raw(sycl::queue& q, void* source_ptr, const range<3>& source_range, const id<3>& source_offset, const id<3>& target_offset,
		    const range<3>& copy_range, device_id did, cudaStream_t copy_stream) override;

		host_buffer<DataT, Dims>& get_host_buffer() { return m_host_buf; }

		const host_buffer<DataT, Dims>& get_host_buffer() const { return m_host_buf; }

	  private:
		host_buffer<DataT, Dims> m_host_buf;
	};

	inline void assert_copy_is_in_range(const cl::sycl::range<3>& source_range, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& source_offset,
	    const cl::sycl::id<3>& target_offset, const cl::sycl::range<3>& copy_range) {
		assert(max_range(source_range, range_cast<3>(source_offset + copy_range)) == source_range);
		assert(max_range(target_range, range_cast<3>(target_offset + copy_range)) == target_range);
	}

	template <typename DataT, int Dims>
	async_event device_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		ZoneScopedN("device_buffer_storage::copy");

#if !USE_NDVBUFFER
		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_device_buf.get_range()), source_offset, target_offset, copy_range);
#endif

		if(source.get_type() == buffer_type::device_buffer) {
			auto& device_source = dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source);
#if TRACY_ENABLE
			const auto msg = fmt::format("d2d {} -> {}, {} bytes", device_source.m_did, m_did, copy_range.size() * sizeof(DataT));
			ZoneText(msg.c_str(), msg.size());
#endif

#if USE_NDVBUFFER
			m_device_buf.copy_from(device_source.m_device_buf,
			    {ndv::point<Dims>::make_from(source_offset), ndv::point<Dims>::make_from(source_offset + copy_range)},
			    {ndv::point<Dims>::make_from(target_offset), ndv::point<Dims>::make_from(target_offset + copy_range)});
#else
			return memcpy_strided_device(m_owning_queue, device_source.m_device_buf.get_pointer(), m_device_buf.get_pointer(), sizeof(DataT),
			    device_source.m_device_buf.get_range(), id_cast<Dims>(source_offset), m_device_buf.get_range(), id_cast<Dims>(target_offset),
			    range_cast<Dims>(copy_range), device_source.m_did, device_source.m_copy_stream);
#endif
		}

		// TODO: Optimize for contiguous copies - we could do a single SYCL H->D copy directly.
		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
#if TRACY_ENABLE
			const auto msg = fmt::format("h2d -> {}, {} bytes", m_did, copy_range.size() * sizeof(DataT));
			ZoneText(msg.c_str(), msg.size());
#endif

#if USE_NDVBUFFER
			m_device_buf.copy_from(static_cast<const DataT*>(host_source.get_pointer()), ndv::extent<Dims>::make_from(host_source.get_range()),
			    {ndv::point<Dims>::make_from(source_offset), ndv::point<Dims>::make_from(source_offset + copy_range)},
			    {ndv::point<Dims>::make_from(target_offset), ndv::point<Dims>::make_from(target_offset + copy_range)});
#else
			return memcpy_strided_device(m_owning_queue, host_source.get_pointer(), m_device_buf.get_pointer(), sizeof(DataT),
			    range_cast<Dims>(host_source.get_range()), id_cast<Dims>(source_offset), m_device_buf.get_range(), id_cast<Dims>(target_offset),
			    range_cast<Dims>(copy_range), m_did, m_copy_stream);
#endif
		}

		else {
			assert(false);
		}

		return async_event{};
	}

	template <typename DataT, int Dims>
	async_event device_buffer_storage<DataT, Dims>::copy_from_device_raw(sycl::queue& q, void* source_ptr, const range<3>& source_range,
	    const id<3>& source_offset, const id<3>& target_offset, const range<3>& copy_range, device_id did, cudaStream_t copy_stream) {
		return memcpy_strided_device(m_owning_queue, source_ptr, m_device_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(source_range),
		    id_cast<Dims>(source_offset), m_device_buf.get_range(), id_cast<Dims>(target_offset), range_cast<Dims>(copy_range), did, copy_stream);
	}

	template <typename DataT, int Dims>
	async_event host_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		ZoneScopedN("host_buffer_storage::copy");

		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_host_buf.get_range()), source_offset, target_offset, copy_range);

		// TODO: Optimize for contiguous copies - we could do a single SYCL D->H copy directly.
		// NOCOMMIT This is USM - can't we just call memcpy here as well?
		if(source.get_type() == buffer_type::device_buffer) {
			auto& device_source = dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source);
			const auto msg = fmt::format("d2h {}", copy_range.size() * sizeof(DataT));
			ZoneText(msg.c_str(), msg.size());

#if USE_NDVBUFFER
			// TODO: It may still be beneficial to first copy into a pinned, dense host allocation. Or should we just allocate host buffers as pinned memory?
			dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source).get_ndv_buffer().copy_to(static_cast<DataT*>(m_host_buf.get_pointer()),
			    ndv::extent<Dims>::make_from(m_host_buf.get_range()),
			    {ndv::point<Dims>::make_from(source_offset), ndv::point<Dims>::make_from(source_offset + copy_range)},
			    {ndv::point<Dims>::make_from(target_offset), ndv::point<Dims>::make_from(target_offset + copy_range)});
#else
			return memcpy_strided_device(device_source.get_owning_queue(), device_source.get_pointer(), m_host_buf.get_pointer(), sizeof(DataT),
			    range_cast<Dims>(device_source.get_range()), id_cast<Dims>(source_offset), range_cast<Dims>(m_host_buf.get_range()),
			    range_cast<Dims>(target_offset), range_cast<Dims>(copy_range), device_source.m_did, device_source.m_copy_stream);
#endif
		}

		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			const auto msg = fmt::format("h2h {}", copy_range.size() * sizeof(DataT));
			ZoneText(msg.c_str(), msg.size());
			memcpy_strided(host_source.get_host_buffer().get_pointer(), m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(host_source.get_range()),
			    id_cast<Dims>(source_offset), range_cast<Dims>(m_host_buf.get_range()), range_cast<Dims>(target_offset), range_cast<Dims>(copy_range));
		}

		else {
			assert(false);
		}

		return async_event{};
	}

	template <typename DataT, int Dims>
	async_event host_buffer_storage<DataT, Dims>::copy_from_device_raw(sycl::queue& q, void* source_ptr, const range<3>& source_range,
	    const id<3>& source_offset, const id<3>& target_offset, const range<3>& copy_range, device_id did, cudaStream_t copy_stream) {
		return memcpy_strided_device(q, source_ptr, m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(source_range), id_cast<Dims>(source_offset),
		    m_host_buf.get_range(), id_cast<Dims>(target_offset), range_cast<Dims>(copy_range), did, copy_stream);
	}

} // namespace detail
} // namespace celerity
