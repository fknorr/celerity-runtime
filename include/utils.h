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
	explicit constexpr overload(F... f) : F(f)... {}
	using F::operator()...;
};

template <typename Variant, typename... Arms>
decltype(auto) match(Variant&& v, Arms&&... arms) {
	return std::visit(overload{std::forward<Arms>(arms)...}, std::forward<Variant>(v));
}

template <typename T>
class declare_visit_fn {
  public:
	virtual void visit(T) = 0;
};

template <typename... Ts>
class visitor : public declare_visit_fn<Ts>... {
  public:
	virtual ~visitor() = default;
	using declare_visit_fn<Ts>::visit...;
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
	explicit impl_visit_fn(Fn&& fn) : fn(std::move(fn)) {}
	Ret get_result() { return std::move(*ret); }

  protected:
	template <typename T>
	void invoke(T v) {
		ret.emplace(fn(static_cast<T>(v)));
	}

  private:
	Fn fn;
	std::optional<Ret> ret;
};

template <typename Base, typename Fn, typename Ret>
class impl_visit_fn<Base, Fn, Ret&> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : fn(std::move(fn)) {}
	Ret& get_result() { return *ret; }

  protected:
	template <typename T>
	void invoke(T v) {
		ret = &fn(static_cast<T>(v));
	}

  private:
	Fn fn;
	Ret* ret = nullptr;
};

template <typename Base, typename Fn, typename Ret>
class impl_visit_fn<Base, Fn, Ret&&> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : fn(std::move(fn)) {}
	Ret&& get_result() { return std::move(*ret); }

  protected:
	template <typename T>
	void invoke(T v) {
		Ret&& name = fn(static_cast<T>(v));
		ret = &name;
	}

  private:
	Fn fn;
	Ret* ret;
};

template <typename Base, typename Fn>
class impl_visit_fn<Base, Fn, void> : public Base {
  public:
	explicit impl_visit_fn(Fn&& fn) : fn(std::move(fn)) {}
	void get_result() {}

  protected:
	template <typename T>
	void invoke(T v) {
		fn(static_cast<T>(v));
	}

  private:
	Fn fn;
};

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
class visitor_impl<visitor<Ts...>, Fn> : public impl_visit_fn<visitor<Ts...>, Fn, visitor_result_t<Fn, Ts...>, Ts...> {
  public:
	using result_type = visitor_result_t<Fn, Ts...>;
	using impl_visit_fn<visitor<Ts...>, Fn, result_type, Ts...>::impl_visit_fn;
};

template <typename Visitor, typename T, typename... Fns>
decltype(auto) match(T& target, Fns&&... fns) {
	using overload_type = overload<std::decay_t<Fns>...>;
	visitor_impl<Visitor, overload_type> vis(overload_type{std::forward<Fns>(fns)...});
	target.accept(vis);
	return vis.get_result();
}

template <typename T, typename Enable = void>
struct default_visitor;

template <typename T>
struct default_visitor<T, std::enable_if_t<std::is_const_v<T>, std::void_t<typename T::const_visitor>>> {
	using type = typename T::const_visitor;
};

template <typename T>
struct default_visitor<T, std::enable_if_t<!std::is_const_v<T>, std::void_t<typename T::visitor>>> {
	using type = typename T::visitor;
};

template <typename T>
using default_visitor_t = typename default_visitor<T>::type;

template <typename T, typename... Fns, typename Visitor = default_visitor_t<T>>
decltype(auto) match(T& target, Fns&&... fns) {
	return match<Visitor, T, Fns...>(target, std::forward<Fns>(fns)...);
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
