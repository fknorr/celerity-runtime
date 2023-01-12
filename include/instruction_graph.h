#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "types.h"

#include <unordered_map>

namespace celerity::detail {

class task;

enum class instruction_type {
	alloc_host,
	alloc_device,
	device_kernel,
	host_kernel,
	copy_h2h,
	copy_d2h,
	copy_h2d,
	copy_d2d,
	send,
	recv,
};

class instruction : public intrusive_graph_node<instruction> {
  public:
	virtual ~instruction() = default;

	virtual instruction_type get_type() const = 0;
};

class alloc_instruction : public instruction {
  public:
	explicit alloc_instruction(const buffer_id bid, const memory_id where, GridRegion<3> region) : m_bid(bid), m_memory(where), m_region(region) {}

	instruction_type get_type() const override { return m_memory == host_memory_id ? instruction_type::alloc_host : instruction_type::alloc_device; }

  private:
	buffer_id m_bid;
	memory_id m_memory;
	GridRegion<3> m_region;
};

class copy_instruction : public instruction {
  public:
	explicit copy_instruction(buffer_id bid, subrange<3> sr, memory_id from, memory_id to) : m_bid(bid), m_sr(sr), m_from(from), m_to(to) {}

	instruction_type get_type() const override {
		const auto h1 = m_from == host_memory_id;
		const auto h2 = m_to == host_memory_id;
		return h1 && h2    ? instruction_type::copy_h2h
		       : h1 && !h2 ? instruction_type::copy_h2d
		       : !h1 && h2 ? instruction_type::copy_d2h
		                   : instruction_type::copy_d2d;
	}

  private:
	buffer_id m_bid;
	subrange<3> m_sr;
	memory_id m_from;
	memory_id m_to;
};

class device_kernel_instruction : public instruction {
  public:
	explicit device_kernel_instruction(const task& tsk, const device_id did, const subrange<3>& execution_range)
	    : m_tsk(tsk), m_device_id(did), m_execution_range(execution_range) {}

	instruction_type get_type() const override { return instruction_type::device_kernel; }

  private:
	const task& m_tsk;
	device_id m_device_id;
	subrange<3> m_execution_range;
};

class send_instruction : public instruction {
  public:
	explicit send_instruction(const node_id to, const buffer_id bid, const subrange<3>& sr) : m_to(to), m_bid(bid), m_sr(sr) {}

	instruction_type get_type() const override { return instruction_type::send; }

  private:
	node_id m_to;
	buffer_id m_bid;
	subrange<3> m_sr;
};

class recv_instruction : public instruction {
  public:
	explicit recv_instruction(const transfer_id transfer, const buffer_id bid, const GridRegion<3>& region)
	    : m_transfer(transfer), m_bid(bid), m_region(region) {}

	instruction_type get_type() const override { return instruction_type::recv; }

  private:
	transfer_id m_transfer;
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class instruction_graph {
  public:
	template <typename Instruction, typename... CtorParams>
	instruction& create(CtorParams&&... ctor_args) {
		return *m_instructions.emplace_back(std::make_unique<Instruction>(std::forward<CtorParams>(ctor_args)...));
	}

  private:
	// TODO split vector into epochs for cleanup phase
	std::vector<std::unique_ptr<instruction>> m_instructions;
};

} // namespace celerity::detail
