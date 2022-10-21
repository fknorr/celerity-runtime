#pragma once

#include <cassert>
#include <cstring>
#include <memory>

#include <CL/sycl.hpp>

#include "payload.h"
#include "ranges.h"
#include "workaround.h"

namespace celerity {
namespace detail {

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
	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<1>& source_range, const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range,
	    const cl::sycl::id<1>& target_offset, const cl::sycl::range<1>& copy_range);

	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<2>& source_range, const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range,
	    const cl::sycl::id<2>& target_offset, const cl::sycl::range<2>& copy_range);

	void memcpy_strided_device(cl::sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
	    const cl::sycl::range<3>& source_range, const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range,
	    const cl::sycl::id<3>& target_offset, const cl::sycl::range<3>& copy_range);

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr);

	template <typename DataT, int Dims>
	class device_buffer {
	  public:
		device_buffer(const range<Dims>& range, sycl::queue& queue) : m_range(range), m_queue(queue) {
			if(m_range.size() != 0) {
				// NOCOMMIT TODO Use aligned allocation?
				m_device_ptr = sycl::malloc_device<DataT>(m_range.size(), m_queue);
				// m_device_ptr = sycl::aligned_alloc_device<DataT>(alignof(DataT), m_range.size(), m_queue);
				assert(m_device_ptr != nullptr);
			}
		}

		~device_buffer() {
			if(m_range.size() != 0) { sycl::free(m_device_ptr, m_queue); }
		}

		device_buffer(const device_buffer&) = delete;

		range<Dims> get_range() const { return m_range; }

		DataT* get_pointer() { return m_device_ptr; }

		const DataT* get_pointer() const { return m_device_ptr; }

		bool operator==(const device_buffer& rhs) const { return m_device_ptr == rhs.m_device_ptr && m_queue == rhs.m_queue && m_range == rhs.m_range; }

	  private:
		sycl::range<Dims> m_range;
		sycl::queue m_queue;
		DataT* m_device_ptr = nullptr;
	};

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

		virtual void get_data(const subrange<3>& sr, void* out_linearized) const = 0;

		virtual void set_data(const subrange<3>& sr, const void* in_linearized) = 0;

		/**
		 * Convenience function to create new buffer_storages of the same (templated) type, useful in contexts where template type information is not available.
		 */
		virtual buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const = 0;

		/**
		 * Copy data from the given source buffer into this buffer.
		 *
		 * TODO: Consider making this non-blocking, returning an async handle instead.
		 */
		virtual void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) = 0;

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
		device_buffer_storage(range<Dims> range, sycl::queue owning_queue)
		    : buffer_storage(range_cast<3>(range), buffer_type::device_buffer), m_owning_queue(owning_queue), m_device_buf(range, m_owning_queue) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		device_buffer<DataT, Dims>& get_device_buffer() { return m_device_buf; }

		const device_buffer<DataT, Dims>& get_device_buffer() const { return m_device_buf; }

		void get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));

			// TODO: Ideally we'd make this non-blocking and return some sort of async handle that can be waited upon
			memcpy_strided_device(m_owning_queue, m_device_buf.get_pointer(), out_linearized, sizeof(DataT), m_device_buf.get_range(), id_cast<Dims>(sr.offset),
			    range_cast<Dims>(sr.range), id<Dims>{}, range_cast<Dims>(sr.range));
		}

		void set_data(const subrange<3>& sr, const void* in_linearized) override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			// NOCOMMIT Use assert_copy_is_in_range from below?
			assert((id_cast<Dims>(sr.offset) + range_cast<Dims>(sr.range) <= m_device_buf.get_range()) == range_cast<Dims>(range<3>{true, true, true}));

			// TODO: Ideally we'd make this non-blocking and return some sort of async handle that can be waited upon
			memcpy_strided_device(m_owning_queue, in_linearized, m_device_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(sr.range), id<Dims>{},
			    m_device_buf.get_range(), id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range));
		}

		buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const override {
			return new device_buffer_storage<DataT, Dims>(range_cast<Dims>(range), m_owning_queue);
		}

		void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

		sycl::queue& get_owning_queue() { return m_owning_queue; }

	  private:
		mutable cl::sycl::queue m_owning_queue;
		device_buffer<DataT, Dims> m_device_buf;
	};

	template <typename DataT, int Dims>
	class host_buffer_storage : public buffer_storage {
	  public:
		explicit host_buffer_storage(cl::sycl::range<Dims> range) : buffer_storage(range_cast<3>(range), buffer_type::host_buffer), m_host_buf(range) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		void get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));

			memcpy_strided(m_host_buf.get_pointer(), out_linearized, sizeof(DataT), range_cast<Dims>(m_host_buf.get_range()), id_cast<Dims>(sr.offset),
			    range_cast<Dims>(sr.range), id_cast<Dims>(cl::sycl::id<3>{0, 0, 0}), range_cast<Dims>(sr.range));
		}

		void set_data(const subrange<3>& sr, const void* in_linearized) override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));

			memcpy_strided(in_linearized, m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(sr.range), id_cast<Dims>(cl::sycl::id<3>(0, 0, 0)),
			    range_cast<Dims>(m_host_buf.get_range()), id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range));
		}

		buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const override { return new host_buffer_storage<DataT, Dims>(range_cast<Dims>(range)); }

		void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

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
	void device_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		ZoneScopedN("device_buffer_storage::copy");
		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_device_buf.get_range()), source_offset, target_offset, copy_range);

		if(source.get_type() == buffer_type::device_buffer) {
			auto& device_source = dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source);
			const auto msg = fmt::format("Copying {},{},{} elements", copy_range[0], copy_range[1], copy_range[2]);
			ZoneText(msg.c_str(), msg.size());
			memcpy_strided_device(m_owning_queue, device_source.m_device_buf.get_pointer(), m_device_buf.get_pointer(), sizeof(DataT),
			    device_source.m_device_buf.get_range(), id_cast<Dims>(source_offset), m_device_buf.get_range(), id_cast<Dims>(target_offset),
			    range_cast<Dims>(copy_range));
		}

		// TODO: Optimize for contiguous copies - we could do a single SYCL H->D copy directly.
		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			auto tmp = make_uninitialized_payload<DataT>(copy_range.size());
			host_source.get_data(subrange{source_offset, copy_range}, static_cast<DataT*>(tmp.get_pointer()));
			set_data(subrange{target_offset, copy_range}, static_cast<const DataT*>(tmp.get_pointer()));
		}

		else {
			assert(false);
		}
	}

	template <typename DataT, int Dims>
	void host_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_host_buf.get_range()), source_offset, target_offset, copy_range);

		// TODO: Optimize for contiguous copies - we could do a single SYCL D->H copy directly.
		// NOCOMMIT This is USM - can't we just call memcpy here as well?
		if(source.get_type() == buffer_type::device_buffer) {
			// This looks more convoluted than using a vector<DataT>, but that would break if DataT == bool
			auto tmp = make_uninitialized_payload<DataT>(copy_range.size());
			source.get_data(subrange{source_offset, copy_range}, static_cast<DataT*>(tmp.get_pointer()));
			set_data(subrange{target_offset, copy_range}, static_cast<const DataT*>(tmp.get_pointer()));
		}

		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			memcpy_strided(host_source.get_host_buffer().get_pointer(), m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(host_source.get_range()),
			    id_cast<Dims>(source_offset), range_cast<Dims>(m_host_buf.get_range()), range_cast<Dims>(target_offset), range_cast<Dims>(copy_range));
		}

		else {
			assert(false);
		}
	}

} // namespace detail
} // namespace celerity
