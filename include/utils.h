#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <variant>

namespace celerity::detail::utils {

template <typename BitMaskT>
constexpr inline uint32_t popcount(const BitMaskT bit_mask) noexcept {
	static_assert(std::is_integral_v<BitMaskT> && std::is_unsigned_v<BitMaskT>, "popcount argument needs to be an unsigned integer type.");

	uint32_t counter = 0;
	for(auto b = bit_mask; b; b >>= 1) {
		counter += b & 1;
	}
	return counter;
}

template <typename... F>
struct overload : F... {
	using F::operator()...;
};

template <typename... F>
overload(F...) -> overload<F...>;

template <typename Variant, typename... Arms>
decltype(auto) match(Variant&& v, Arms&&... arms) {
	using overload_type = overload<std::decay_t<Arms>...>;
	return std::visit(overload_type{std::forward<Arms>(arms)...}, std::forward<Variant>(v));
}

} // namespace celerity::detail::utils

namespace celerity::detail::utils_detail {

template <typename T>
class declare_visit_fn {
  public:
	virtual void visit(T) = 0;
};

template <typename Base, typename Fn, typename Ret, typename... Ts>
class impl_visit_fn;

template <typename Base, typename Fn, typename Ret, typename T, typename... Ts>
class impl_visit_fn<Base, Fn, Ret, T, Ts...> : public impl_visit_fn<Base, Fn, Ret, Ts...> {
  private:
	using next = impl_visit_fn<Base, Fn, Ret, Ts...>;

  public:
	explicit impl_visit_fn(Fn&& fn) : next(std::move(fn)) {}
	void visit(T v) final { next::template invoke<T>(v); }
	using next::visit;
};

template <typename Base, typename Fn, typename Ret>
class impl_visit_fn<Base, Fn, Ret> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : m_fn(std::move(fn)) {}
	Ret get_result() { return std::move(*m_ret); }

  protected:
	template <typename T>
	void invoke(T v) {
		static_assert(std::is_same_v<std::invoke_result_t<Fn, T>, Ret>);
		m_ret.emplace(m_fn(static_cast<T>(v)));
	}

  private:
	Fn m_fn;
	std::optional<Ret> m_ret;
};

template <typename Base, typename Fn, typename Ret>
class impl_visit_fn<Base, Fn, Ret&> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : m_fn(std::move(fn)) {}
	Ret& get_result() { return *m_ret; }

  protected:
	template <typename T>
	void invoke(T v) {
		static_assert(std::is_same_v<std::invoke_result_t<Fn, T>, Ret&>);
		m_ret = &m_fn(static_cast<T>(v));
	}

  private:
	Fn m_fn;
	Ret* m_ret = nullptr;
};

template <typename Base, typename Fn, typename Ret>
class impl_visit_fn<Base, Fn, Ret&&> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : m_fn(std::move(fn)) {}
	Ret&& get_result() { return std::move(*m_ret); }

  protected:
	template <typename T>
	void invoke(T v) {
		static_assert(std::is_same_v<std::invoke_result_t<Fn, T>, Ret&&>);
		Ret&& name = fn(static_cast<T>(v));
		m_ret = &name;
	}

  private:
	Fn m_fn;
	Ret* m_ret;
};

template <typename Base, typename Fn>
class impl_visit_fn<Base, Fn, void> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : m_fn(std::move(fn)) {}
	void get_result() {}

  protected:
	template <typename T>
	void invoke(T v) {
		static_assert(std::is_void_v<std::invoke_result_t<Fn, T>>);
		m_fn(static_cast<T>(v));
	}

  private:
	Fn m_fn;
};

} // namespace celerity::detail::utils_detail

namespace celerity::detail::utils {

template <typename... Ts>
class visitor : public utils_detail::declare_visit_fn<Ts>... {
  public:
	virtual ~visitor() = default;
	using utils_detail::declare_visit_fn<Ts>::visit...;
};

} // namespace celerity::detail::utils

namespace celerity::detail::utils_detail {

template <typename Fn, typename T, typename... Ts>
struct visitor_result {
	using type = std::invoke_result_t<Fn, T>;
	static_assert((... && std::is_same_v<type, std::invoke_result_t<Fn, Ts>>), "all overloads must return the same type");
};

template <typename Fn, typename... Ts>
using visitor_result_t = typename visitor_result<Fn, Ts...>::type;

template <typename Visitor, typename Fn>
class visitor_impl;

template <typename... Ts, typename Fn>
class visitor_impl<utils::visitor<Ts...>, Fn> : public impl_visit_fn<utils::visitor<Ts...>, Fn, visitor_result_t<Fn, Ts...>, Ts...> {
  public:
	using result_type = visitor_result_t<Fn, Ts...>;
	using impl_visit_fn<utils::visitor<Ts...>, Fn, result_type, Ts...>::impl_visit_fn;
};

template <typename T, typename Enable = void>
inline constexpr bool declares_const_visitor_v = false;

template <typename T>
inline constexpr bool declares_const_visitor_v<T, std::void_t<typename T::const_visitor>> = true;

template <typename T, typename Enable = void>
inline constexpr bool declares_visitor_v = false;

template <typename T>
inline constexpr bool declares_visitor_v<T, std::void_t<typename T::visitor>> = true;

} // namespace celerity::detail::utils_detail

namespace celerity::detail::utils {

template <typename Visitor, typename T, typename... Arms>
decltype(auto) match(T& target, Arms&&... arms) {
	using overload_type = overload<std::decay_t<Arms>...>;
	utils_detail::visitor_impl<Visitor, overload_type> vis(overload_type{std::forward<Arms>(arms)...});
	target.accept(vis);
	return vis.get_result();
}

template <typename T, typename Enable = void>
struct default_visitor;

template <typename T>
struct default_visitor<T, std::enable_if_t<utils_detail::declares_const_visitor_v<T> && (std::is_const_v<T> || !utils_detail::declares_visitor_v<T>)>> {
	using type = typename T::const_visitor;
};

template <typename T>
struct default_visitor<T, std::enable_if_t<utils_detail::declares_visitor_v<T> && !std::is_const_v<T>>> {
	using type = typename T::visitor;
};

template <typename T>
using default_visitor_t = typename default_visitor<T>::type;

template <typename T, typename... Arms, typename Visitor = default_visitor_t<T>>
decltype(auto) match(T& target, Arms&&... arms) {
	return match<Visitor, T, Arms...>(target, std::forward<Arms>(arms)...);
}

// Implementation from Boost.ContainerHash, licensed under the Boost Software License, Version 1.0.
inline void hash_combine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

struct pair_hash {
	template <typename U, typename V>
	std::size_t operator()(const std::pair<U, V>& p) const {
		std::size_t seed = 0;
		hash_combine(seed, std::hash<U>{}(p.first));
		hash_combine(seed, std::hash<V>{}(p.second));
		return seed;
	}
};

} // namespace celerity::detail::utils
