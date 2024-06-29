#pragma once

#include "types.h"

#include <memory>
#include <numeric>

namespace celerity::detail {

class reducer {
  public:
	virtual ~reducer() = default;

	virtual void reduce(void* dest, const void* src, size_t src_count) const = 0;
	virtual void fill_identity(void* dest, size_t count) const = 0;

  protected:
	reducer() = default;
	reducer(const reducer&) = default;
	reducer(reducer&&) = default;
	reducer& operator=(const reducer&) = default;
	reducer& operator=(reducer&&) = default;
};

template <typename Scalar, typename BinaryOp>
class reducer_impl final : public reducer {
  public:
	reducer_impl(const BinaryOp& op, const Scalar& identity) : m_op(op), m_identity(identity) {}

	void reduce(void* const dest, const void* const src, const size_t src_count) const override {
		const auto v_dest = static_cast<Scalar*>(dest);
		const auto v_src = static_cast<const Scalar*>(src);
		*v_dest = std::reduce(v_src, v_src + src_count, m_identity, m_op);
	}

	void fill_identity(void* const dest, const size_t count) const override { //
		std::uninitialized_fill_n(static_cast<Scalar*>(dest), count, m_identity);
	}

  private:
	BinaryOp m_op;
	Scalar m_identity;
};

template <typename Scalar, typename BinaryOp>
std::unique_ptr<reducer> make_reducer(const BinaryOp& op, const Scalar& identity) {
	return std::make_unique<reducer_impl<Scalar, BinaryOp>>(op, identity);
}

struct reduction_info {
	reduction_id rid = 0;
	buffer_id bid = 0;
	bool init_from_buffer = false;
};

} // namespace celerity::detail
