#pragma once

#include <memory>
#include <type_traits>

#include "accessor.h"
#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	template <typename CGF>
	constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

	struct capture_inspector {
	  public:
		template <typename Capture>
		static void record_requirements(const Capture& cap, detail::buffer_access_map& accesses, detail::side_effect_map& side_effects) {
			cap.record_requirements(accesses, side_effects);
		}

		template <typename Captures>
		static void record_requirements_on_tuple(const Captures& caps, detail::buffer_access_map& accesses, detail::side_effect_map& side_effects) {
			record_requirements_on_tuple(caps, accesses, side_effects, std::make_index_sequence<std::tuple_size_v<Captures>>{});
		}

		template <typename Capture>
		static auto exfiltrate_by_copy(const Capture& cap) {
			return cap.exfiltrate_by_copy();
		}

		template <typename Capture>
		static auto exfiltrate_by_move(const Capture& cap) {
			return cap.exfiltrate_by_move();
		}

		template <typename Captures>
		static auto exfiltrate_tuple_by_copy(const Captures& caps) {
			return exfiltrate_tuple_by_copy(caps, std::make_index_sequence<std::tuple_size_v<Captures>>{});
		}

		template <typename Captures>
		static auto exfiltrate_tuple_by_move(const Captures& caps) {
			return exfiltrate_tuple_by_move(caps, std::make_index_sequence<std::tuple_size_v<Captures>>{});
		}

	  private:
		template <typename Captures, size_t... Is>
		static void record_requirements_on_tuple(
		    const Captures& caps, detail::buffer_access_map& accesses, detail::side_effect_map& side_effects, std::index_sequence<Is...>) {
			(std::get<Is>(caps).record_requirements(accesses, side_effects), ...);
		}

		template <typename Captures, size_t... Is>
		static auto exfiltrate_tuple_by_copy(const Captures& caps, std::index_sequence<Is...>) {
			return std::tuple{std::get<Is>(caps).exfiltrate_by_copy()...};
		}

		template <typename Captures, size_t... Is>
		static auto exfiltrate_tuple_by_move(const Captures& caps, std::index_sequence<Is...>) {
			return std::tuple{std::get<Is>(caps).exfiltrate_by_move()...};
		}
	};

	struct queue_inspector;

} // namespace detail

class distr_queue;

struct allow_by_ref_t {};

inline constexpr allow_by_ref_t allow_by_ref{};

namespace experimental {

	template <typename>
	class capture;

	template <typename T, int Dims>
	class buffer_data {
	  public:
		buffer_data() : range{detail::zero_range} {}

		explicit operator bool() const { return !data.empty(); }

		celerity::range<Dims> get_range() const { return range; }
		const T* get_pointer() const { return data.data(); }
		T* get_pointer() { return data.data(); }

		// TODO accessor semantics with operator[]; into_vector()

	  private:
		friend class capture<buffer<T, Dims>>;

		celerity::range<Dims> range;
		std::vector<T> data;

		explicit buffer_data(celerity::range<Dims> range, std::vector<T> data) : range{range}, data{std::move(data)} {
			assert(this->data.size() == this->range.size());
		}
	};

	template <typename T, int Dims>
	class capture<buffer<T, Dims>> {
	  public:
		using value_type = buffer_data<T, Dims>;

		explicit capture(buffer<T, Dims> buf) : buffer{std::move(buf)}, sr{{}, buffer.get_range()} {}
		explicit capture(buffer<T, Dims> buf, const subrange<Dims>& sr) : buffer{std::move(buf)}, sr{sr} {}

	  private:
		friend struct detail::capture_inspector;

		buffer<T, Dims> buffer;
		subrange<Dims> sr;

		void record_requirements(detail::buffer_access_map& accesses, detail::side_effect_map&) const {
			accesses.add_access(detail::get_buffer_id(buffer),
			    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(sr, access_mode::read, buffer.get_range()));
		}

		value_type exfiltrate_by_copy() const {
			auto& bm = detail::runtime::get_instance().get_buffer_manager();
			const auto access_info =
			    bm.get_host_buffer<T, Dims>(detail::get_buffer_id(buffer), access_mode::read, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));

			// TODO this should be able to use host_buffer_storage::get_data
			const auto allocation_window = buffer_allocation_window<T, Dims>{
			    access_info.buffer.get_pointer(),
			    buffer.get_range(),
			    access_info.buffer.get_range(),
			    sr.range,
			    access_info.offset,
			    sr.offset,
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

			return value_type{allocation_window.get_window_range(), std::move(data)};
		}

		value_type exfiltrate_by_move() const { return exfiltrate_by_copy(); }
	};

	template <typename T, int Dims>
	capture(buffer<T, Dims>) -> capture<buffer<T, Dims>>;

	template <typename T, int Dims>
	capture(buffer<T, Dims>, subrange<Dims>) -> capture<buffer<T, Dims>>;

	template <typename T>
	class capture<host_object<T>> {
	  public:
		static_assert(std::is_object_v<T>);

		using value_type = T;

		explicit capture(host_object<T> ho) : ho{std::move(ho)} {}

	  private:
		friend struct detail::capture_inspector;

		host_object<T> ho;

		void record_requirements(detail::buffer_access_map&, detail::side_effect_map& side_effects) const {
			side_effects.add_side_effect(ho.get_id(), side_effect_order::sequential);
		}

		value_type exfiltrate_by_copy() const { return value_type{std::as_const(*ho.get_object())}; }

		value_type exfiltrate_by_move() const { return value_type{std::move(*ho.get_object())}; }
	};

	template <typename T>
	capture(host_object<T>) -> capture<host_object<T>>;

} // namespace experimental

class distr_queue {
  public:
	distr_queue() { init(nullptr); }
	distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(&device);
	}

	distr_queue(const distr_queue&) = delete;
	distr_queue& operator=(const distr_queue&) = delete;

	~distr_queue() {
		if(!drained) { drain({}, {}); }
	}

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
		check_not_drained();
		detail::runtime::get_instance().get_task_manager().create_task(std::move(cgf));
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
	void slow_full_sync() { // NOLINT(readability-convert-member-functions-to-static)
		check_not_drained();
		detail::runtime::get_instance().sync({}, {});
	}

	template <typename T>
	typename experimental::capture<T>::value_type slow_full_sync(const experimental::capture<T>& cap) {
		check_not_drained();
		detail::buffer_access_map accesses;
		detail::side_effect_map side_effects;
		detail::capture_inspector::record_requirements(cap, accesses, side_effects);
		detail::runtime::get_instance().sync(std::move(accesses), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_by_copy(cap);
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> slow_full_sync(const std::tuple<experimental::capture<Ts>...>& caps) {
		check_not_drained();
		detail::buffer_access_map accesses;
		detail::side_effect_map side_effects;
		detail::capture_inspector::record_requirements_on_tuple(caps, accesses, side_effects);
		detail::runtime::get_instance().sync(std::move(accesses), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_tuple_by_copy(caps);
	}

  private:
	friend struct detail::queue_inspector;

	void init(cl::sycl::device* user_device) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, user_device); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	}

	void check_not_drained() const {
		if(drained) { throw std::runtime_error("distr_queue has already been drained"); }
	}

	void drain(detail::buffer_access_map buffer_accesses, detail::side_effect_map side_effects) {
		check_not_drained();
		detail::runtime::get_instance().shutdown(std::move(buffer_accesses), std::move(side_effects));
		drained = true;
	}

	bool drained = false;
};

namespace detail {

	struct queue_inspector {
		static void drain(distr_queue& q, buffer_access_map buffer_accesses, side_effect_map side_effects) {
			q.drain(std::move(buffer_accesses), std::move(side_effects));
		}
	};

} // namespace detail

namespace experimental {

	inline void drain(distr_queue&& q) { detail::queue_inspector::drain(q, {}, {}); }

	template <typename T>
	typename experimental::capture<T>::value_type drain(distr_queue&& q, const experimental::capture<T>& cap) {
		detail::buffer_access_map accesses;
		detail::side_effect_map side_effects;
		detail::capture_inspector::record_requirements(cap, accesses, side_effects);
		detail::queue_inspector::drain(q, std::move(accesses), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_by_move(cap);
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> drain(distr_queue&& q, const std::tuple<experimental::capture<Ts>...>& caps) {
		detail::buffer_access_map accesses;
		detail::side_effect_map side_effects;
		detail::capture_inspector::record_requirements_on_tuple(caps, accesses, side_effects);
		detail::queue_inspector::drain(q, std::move(accesses), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_tuple_by_move(caps);
	}

} // namespace experimental
} // namespace celerity
