#pragma once

#include <memory>
#include <type_traits>

#include "accessor.h"
#include "buffer_manager.h"
#include "device_queue.h"
#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	class distr_queue_tracker {
	  public:
		~distr_queue_tracker() { runtime::get_instance().shutdown(); }
	};

	template <typename CGF>
	constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

	struct fence_builder;

} // namespace detail

struct allow_by_ref_t {};

inline constexpr allow_by_ref_t allow_by_ref{};

class distr_queue {
  public:
	distr_queue() { init(detail::auto_select_device{}); }

	[[deprecated("Use the overload with device selector instead, this will be removed in future release")]] distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(device);
	}

	template <typename DeviceSelector>
	distr_queue(const DeviceSelector& device_selector) {
		if(detail::runtime::is_initialized()) {
			throw std::runtime_error("Passing explicit device selector not possible, runtime has already been initialized.");
		}
		init(device_selector);
	}

	distr_queue(const distr_queue&) = default;
	distr_queue(distr_queue&&) = default;

	distr_queue& operator=(const distr_queue&) = delete;
	distr_queue& operator=(distr_queue&&) = delete;

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 *
	 * With this overload, CGF may capture by-reference. This may lead to lifetime issues with asynchronous execution, so using the `submit(cgf)` overload is
	 * preferred in the common case.
	 */
	template <typename CGF>
	void submit(allow_by_ref_t, CGF cgf) { // NOLINT(readability-convert-member-functions-to-static)
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		detail::runtime::get_instance().get_task_manager().submit_command_group(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * CGF must not capture by reference. This is a conservative safety check to avoid lifetime issues when command groups are executed asynchronously.
	 *
	 * If you know what you are doing, you can use the `allow_by_ref` overload of `submit` to bypass this check.
	 */
	template <typename CGF>
	void submit(CGF cgf) {
		static_assert(detail::is_safe_cgf<CGF>, "The provided command group function is not multi-pass execution safe. Please make sure to only capture by "
		                                        "value. If you know what you're doing, use submit(celerity::allow_by_ref, ...).");
		submit(allow_by_ref, std::move(cgf));
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes accross the entire cluster. }
	 */
	void slow_full_sync() { detail::runtime::get_instance().sync(); } // NOLINT(readability-convert-member-functions-to-static)

  private:
	std::shared_ptr<detail::distr_queue_tracker> m_tracker;

	void init(detail::device_or_selector device_or_selector) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, device_or_selector); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) {
			throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)");
		}
		m_tracker = std::make_shared<detail::distr_queue_tracker>();
	}
};

namespace experimental {

	template <typename T, int Dims>
	class buffer_subrange {
	  public:
		celerity::buffer<T, Dims> buffer;
		celerity::subrange<Dims> subrange;

		buffer_subrange(const celerity::buffer<T, Dims>& buffer, const celerity::subrange<Dims>& subrange) : buffer(buffer), subrange(subrange) {
			assert((detail::range_cast<3>(subrange.offset + subrange.range) <= detail::range_cast<3>(buffer.get_range())) == range<3>(true, true, true));
		}

		buffer_subrange(const celerity::buffer<T, Dims>& buffer) : buffer(buffer), subrange({}, buffer.get_range()) {}
	};

	template <typename T, int Dims>
	class buffer_snapshot {
	  public:
		buffer_snapshot() : m_sr({}, detail::zero_range) {}

		explicit operator bool() const { return !m_data.empty(); }

		range<Dims> get_offset() const { return m_sr.offset; }

		range<Dims> get_range() const { return m_sr.range; }

		subrange<Dims> get_subrange() const { return m_sr; }

		const std::vector<T>& get_data() const { return m_data; }

		std::vector<T> into_data() && { return std::move(m_data); }

		inline const T& operator[](id<Dims> index) const { return m_data[detail::get_linear_index(m_sr.range, index)]; }

		inline detail::subscript_result_t<Dims, const buffer_snapshot> operator[](size_t index) const { return detail::subscript<Dims>(*this, index); }

		friend bool operator==(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return lhs.m_sr == rhs.m_sr && lhs.m_data == rhs.m_data; }

		friend bool operator!=(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return !operator==(lhs, rhs); }

	  private:
		friend struct detail::fence_builder;

		subrange<Dims> m_sr;
		std::vector<T> m_data;

		explicit buffer_snapshot(subrange<Dims> sr, std::vector<T> data) : m_sr(sr), m_data(std::move(data)) { assert(m_data.size() == m_sr.range.size()); }
	};

} // namespace experimental

namespace detail {

	struct fence_builder {
		buffer_capture_map buffer_captures;
		side_effect_map side_effects;

		template <typename T>
		void add(const experimental::host_object<T>& ho) {
			side_effects.add_side_effect(detail::get_host_object_id(ho), experimental::side_effect_order::sequential);
		}

		template <typename T, int Dims>
		void add(const experimental::buffer_subrange<T, Dims>& bsr) {
			buffer_captures.add_read_access(detail::get_buffer_id(bsr.buffer), detail::subrange_cast<3>(bsr.subrange));
		}

		template <typename T, int Dims>
		void add(const buffer<T, Dims>& buf) {
			buffer_captures.add_read_access(detail::get_buffer_id(buf), subrange<3>({}, range_cast<3>(buf.get_range())));
		}

		fence_guard await_and_acquire() {
			auto& tm = detail::runtime::get_instance().get_task_manager();
			const auto tid = tm.generate_fence_task(std::move(buffer_captures), std::move(side_effects));
			return tm.get_task(tid)->get_fence().await_arrived_and_acquire();
		}

		template <typename T>
		T extract(const experimental::host_object<T>& ho) const {
			return detail::get_host_object_instance(ho);
		}

		template <typename T, int Dims>
		experimental::buffer_snapshot<T, Dims> extract(const experimental::buffer_subrange<T, Dims>& bsr) const {
			auto& bm = detail::runtime::get_instance().get_buffer_manager();
			const auto access_info = bm.begin_host_buffer_access<T, Dims>(
			    detail::get_buffer_id(bsr.buffer), access_mode::read, detail::range_cast<3>(bsr.subrange.range), detail::id_cast<3>(bsr.subrange.offset));

			// TODO this should be able to use host_buffer_storage::get_data
			const auto allocation_window = buffer_allocation_window<T, Dims>{
			    static_cast<T*>(access_info.ptr),
			    bsr.buffer.get_range(),
			    range_cast<Dims>(access_info.backing_buffer_range),
			    bsr.subrange.range,
			    id_cast<Dims>(access_info.backing_buffer_offset),
			    bsr.subrange.offset,
			};
			const auto allocation_range_3 = detail::range_cast<3>(allocation_window.get_allocation_range());
			const auto window_range_3 = detail::range_cast<3>(allocation_window.get_window_range());
			const auto read_offset_3 = detail::id_cast<3>(allocation_window.get_window_offset_in_allocation());
			std::vector<T> data(allocation_window.get_window_range().size());
			for(id<3> item{0, 0, 0}; item[0] < window_range_3[0]; ++item[0]) {
				for(item[1] = 0; item[1] < window_range_3[1]; ++item[1]) {
					for(item[2] = 0; item[2] < window_range_3[2]; ++item[2]) {
						data[detail::get_linear_index(window_range_3, item)] =
						    allocation_window.get_allocation()[detail::get_linear_index(allocation_range_3, item + read_offset_3)];
					}
				}
			}

			bm.end_buffer_access(buffer_manager::data_location{}.set(bm.get_host_memory_id()), detail::get_buffer_id(bsr.buffer), access_mode::read,
			    detail::range_cast<3>(bsr.subrange.range), detail::id_cast<3>(bsr.subrange.offset));

			return experimental::buffer_snapshot<T, Dims>(bsr.subrange, std::move(data));
		}

		template <typename T, int Dims>
		experimental::buffer_snapshot<T, Dims> extract(const buffer<T, Dims>& buf) const {
			return extract(experimental::buffer_subrange(buf, subrange({}, buf.get_range())));
		}
	};

	template <typename>
	struct captured_data;

	template <typename T>
	struct captured_data<experimental::host_object<T>> {
		using type = T;
	};

	template <typename T, int Dims>
	struct captured_data<buffer<T, Dims>> {
		using type = experimental::buffer_snapshot<T, Dims>;
	};

	template <typename T, int Dims>
	struct captured_data<experimental::buffer_subrange<T, Dims>> {
		using type = experimental::buffer_snapshot<T, Dims>;
	};

	template <typename T>
	using captured_data_t = typename captured_data<T>::type;

	template <typename... Captures, size_t... Indices>
	std::tuple<detail::captured_data_t<Captures>...> fence_internal(const std::tuple<Captures...>& cap, std::index_sequence<Indices...>) {
		detail::fence_builder builder;
		(builder.add(std::get<Indices>(cap)), ...);
		const auto guard = builder.await_and_acquire();
		return std::tuple(builder.extract(std::get<Indices>(cap))...);
	}

} // namespace detail

namespace experimental {

	template <typename Capture>
	detail::captured_data_t<Capture> fence(distr_queue&, const Capture& cap) {
		detail::fence_builder builder;
		builder.add(cap);
		const auto guard = builder.await_and_acquire();
		return builder.extract(cap);
	}

	template <typename... Captures>
	std::tuple<detail::captured_data_t<Captures>...> fence(distr_queue&, const std::tuple<Captures...>& cap) {
		return detail::fence_internal(cap, std::index_sequence_for<Captures...>());
	}

} // namespace experimental
} // namespace celerity
