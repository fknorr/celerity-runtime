#pragma once

#include "types.h"

#include <memory>

namespace celerity::detail {

class reduction_interface {
  public:
	virtual ~reduction_interface() = default;
	virtual void reduce_scalar(void* acc, const void* val) const = 0;
	virtual const void* get_scalar_identity() const = 0;

  protected:
	reduction_interface() = default;
	reduction_interface(const reduction_interface&) = default;
	reduction_interface(reduction_interface&&) = default;
	reduction_interface& operator=(const reduction_interface&) = default;
	reduction_interface& operator=(reduction_interface&&) = default;
};

template <typename Scalar, typename BinaryOp>
class reduction_implementation : public reduction_interface {
  public:
	reduction_implementation(const BinaryOp op, const Scalar identity) : m_op(op), m_identity(identity) {}

	void reduce_scalar(void* acc, const void* val) const override {
		*static_cast<Scalar*>(acc) = m_op(*static_cast<const Scalar*>(acc), *static_cast<const Scalar*>(val));
	}
	const void* get_scalar_identity() const override { return &m_identity; }

  private:
	BinaryOp m_op;
	Scalar m_identity;
};

template <typename Scalar, typename BinaryOp>
std::unique_ptr<reduction_interface> make_reduction_interface(const BinaryOp op, const Scalar identity) {
	return std::make_unique<reduction_implementation<Scalar, BinaryOp>>(op, identity);
}

struct reduction_info {
	reduction_id rid = 0;
	buffer_id bid = 0;
	bool init_from_buffer = false;
};

} // namespace celerity::detail
