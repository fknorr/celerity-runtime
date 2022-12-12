#pragma once

#include "intrusive_graph.h"
#include "types.h"

#include <unordered_map>

namespace celerity::detail {

enum class instruction_type {
	alloc_host,
	alloc_device,
	kernel,
	copy_h2h,
	copy_d2h,
	copy_h2d,
	copy_d2d,
};

enum class execution_unit {
	host,
	device0,
};

class instruction : public intrusive_graph_node<instruction> {
  public:
	virtual ~instruction() = default;

	virtual instruction_type get_type() const = 0;
};

class alloc_instruction : public instruction {
  public:
	instruction_type get_type() const override { return m_where == execution_unit::host ? instruction_type::alloc_host : instruction_type::alloc_device; }

  private:
	execution_unit m_where;
};

class copy_instruction : public instruction {
  public:
	instruction_type get_type() const override {
		const auto h1 = m_from == execution_unit::host;
		const auto h2 = m_to == execution_unit::host;
		return h1 && h2    ? instruction_type::copy_h2h
		       : h1 && !h2 ? instruction_type::copy_h2d
		       : !h1 && h2 ? instruction_type::copy_d2h
		                   : instruction_type::copy_d2d;
	}

  private:
	execution_unit m_from;
	execution_unit m_to;
};

class device_kernel_instruction : public instruction {
  public:
	instruction_type get_type() const override { return instruction_type::kernel; }
};

class instruction_graph {
  private:
	std::unordered_map<instruction_id, std::unique_ptr<instruction>> m_instructions;
};

} // namespace celerity::detail
