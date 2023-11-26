#pragma once

#include <cstdlib>
#include <functional>
#include <utility>

namespace celerity {
namespace detail {

	/// Like `false`, but dependent on one or more template parameters. Use as the condition of always-failing static assertions in overloads, template
	/// specializations or `if constexpr` branches.
	template <typename...>
	constexpr bool constexpr_false = false;

	template <typename T, typename UniqueName>
	class PhantomType {
	  public:
		using underlying_t = T;

		constexpr PhantomType() = default;
		constexpr PhantomType(const T& value) : m_value(value) {}
		constexpr PhantomType(T&& value) : m_value(std::move(value)) {}

		// Allow implicit conversion to underlying type, otherwise it becomes too annoying to use.
		// Luckily compilers won't do more than one user-defined conversion, so something like
		// PhantomType1<T> -> T -> PhantomType2<T>, can't happen. Therefore we still retain
		// strong typesafety between phantom types with the same underlying type.
		constexpr operator T&() { return m_value; }
		constexpr operator const T&() const { return m_value; }

	  private:
		T m_value;
	};

} // namespace detail
} // namespace celerity

#define MAKE_PHANTOM_TYPE(TypeName, UnderlyingT)                                                                                                               \
	namespace celerity {                                                                                                                                       \
		namespace detail {                                                                                                                                     \
			using TypeName = PhantomType<UnderlyingT, class TypeName##_PhantomType>;                                                                           \
		}                                                                                                                                                      \
	}                                                                                                                                                          \
	namespace std {                                                                                                                                            \
		template <>                                                                                                                                            \
		struct hash<celerity::detail::TypeName> {                                                                                                              \
			std::size_t operator()(const celerity::detail::TypeName& t) const noexcept { return std::hash<UnderlyingT>{}(static_cast<UnderlyingT>(t)); }       \
		};                                                                                                                                                     \
	}

MAKE_PHANTOM_TYPE(task_id, size_t)
MAKE_PHANTOM_TYPE(buffer_id, size_t)
MAKE_PHANTOM_TYPE(node_id, size_t)
MAKE_PHANTOM_TYPE(command_id, size_t)
MAKE_PHANTOM_TYPE(collective_group_id, size_t)
MAKE_PHANTOM_TYPE(reduction_id, size_t)
MAKE_PHANTOM_TYPE(host_object_id, size_t)
MAKE_PHANTOM_TYPE(hydration_id, size_t)
MAKE_PHANTOM_TYPE(memory_id, size_t)
MAKE_PHANTOM_TYPE(device_id, size_t)
MAKE_PHANTOM_TYPE(allocation_id, size_t)
MAKE_PHANTOM_TYPE(instruction_id, size_t)


// declared in this header for include-dependency reasons
namespace celerity::experimental {

enum class side_effect_order { sequential };

}

namespace celerity::detail {

inline constexpr node_id master_node_id = 0;

inline constexpr memory_id user_memory_id = 0; // (unpinned) host memory allocated for or by the user
inline constexpr memory_id host_memory_id = 1; // (pinned) host memory for buffer-backing allocations
inline constexpr memory_id first_device_memory_id = 2;

inline constexpr allocation_id null_allocation_id = 0; // allocation_id equivalent of a null pointer

inline constexpr collective_group_id non_collective_group_id = 0;
inline constexpr collective_group_id root_collective_group_id = 1;

inline constexpr reduction_id no_reduction_id = 0;

struct transfer_id {
	task_id consumer_tid = -1;
	buffer_id bid = -1;
	reduction_id rid = no_reduction_id;

	transfer_id() = default;
	transfer_id(const task_id consumer_tid, const buffer_id bid, const reduction_id rid = no_reduction_id) : consumer_tid(consumer_tid), bid(bid), rid(rid) {}

	friend bool operator==(const transfer_id& lhs, const transfer_id& rhs) {
		return lhs.consumer_tid == rhs.consumer_tid && lhs.bid == rhs.bid && lhs.rid == rhs.rid;
	}
	friend bool operator!=(const transfer_id& lhs, const transfer_id& rhs) { return !(lhs == rhs); }
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::transfer_id> {
	std::size_t operator()(const celerity::detail::transfer_id& t) const noexcept; // defined in utils.cc
};

namespace celerity::detail {

enum class error_policy {
	ignore,
	log_warning,
	log_error,
	throw_exception,
};

enum class epoch_action {
	none,
	barrier,
	shutdown,
};

} // namespace celerity::detail
