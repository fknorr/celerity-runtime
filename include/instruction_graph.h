#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "types.h"

#include <unordered_map>

namespace celerity::detail {

class task;

class instruction;
class alloc_instruction;
class copy_instruction;
class device_kernel_instruction;
class host_kernel_instruction;
class send_instruction;
class recv_instruction;
class horizon_instruction;
class epoch_instruction;

class const_instruction_graph_visitor {
  public:
	virtual ~const_instruction_graph_visitor() = default;

	virtual void visit(const instruction& insn) {}
	virtual void visit_alloc(const alloc_instruction& ainsn);
	virtual void visit_copy(const copy_instruction& cinsn);
	virtual void visit_device_kernel(const device_kernel_instruction& dkinsn);
	virtual void visit_host_kernel(const host_kernel_instruction& hkinsn);
	virtual void visit_send(const send_instruction& sinsn);
	virtual void visit_recv(const recv_instruction& rinsn);
	virtual void visit_horizon(const horizon_instruction& hinsn);
	virtual void visit_epoch(const epoch_instruction& einsn);
};

class instruction : public intrusive_graph_node<instruction> {
  public:
	explicit instruction(const instruction_id id, const memory_id mid) : m_id(id), m_mid(mid) {}

	instruction(const instruction&) = delete;
	instruction& operator=(const instruction&) = delete;
	virtual ~instruction() = default;

	virtual void visit(const_instruction_graph_visitor& visitor) const = 0;

	instruction_id get_id() const { return m_id; }
	memory_id get_memory_id() const { return m_mid; }

  private:
	instruction_id m_id;
	memory_id m_mid;
};

struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
};

class alloc_instruction : public instruction {
  public:
	explicit alloc_instruction(const instruction_id id, const memory_id mid, const buffer_id bid, GridRegion<3> region)
	    : instruction(id, mid), m_bid(bid), m_region(region) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_alloc(*this); }

	buffer_id get_buffer_id() const { return m_bid; }
	GridRegion<3> get_region() const { return m_region; }

  private:
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class copy_instruction : public instruction {
  public:
	enum class side { source, dest };

	static std::pair<std::unique_ptr<copy_instruction>, std::unique_ptr<copy_instruction>> make_pair(const instruction_id source_id, const memory_id source_mid,
	    const instruction_id dest_id, const memory_id dest_mid, const buffer_id bid, GridRegion<3> region) {
		std::unique_ptr<copy_instruction> source(new copy_instruction(source_id, source_mid, bid, region, side::source));
		std::unique_ptr<copy_instruction> dest(new copy_instruction(dest_id, dest_mid, bid, std::move(region), side::dest));
		source->m_counterpart = dest.get();
		dest->m_counterpart = source.get();
		return std::pair(std::move(source), std::move(dest));
	}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_copy(*this); }

	const buffer_id& get_buffer_id() const { return m_bid; }
	const GridRegion<3>& get_region() const { return m_region; }
	memory_id get_source_memory() const { return m_side == side::source ? get_memory_id() : m_counterpart->get_memory_id(); }
	memory_id get_dest_memory() const { return m_side == side::dest ? get_memory_id() : m_counterpart->get_memory_id(); }
	side get_side() const { return m_side; }
	instruction& get_counterpart() const { return *m_counterpart; }

  private:
	buffer_id m_bid;
	GridRegion<3> m_region;
	side m_side;
	instruction* m_counterpart;

	explicit copy_instruction(const instruction_id id, const memory_id mid, const buffer_id bid, GridRegion<3> region, const side side)
	    : instruction(id, mid), m_bid(bid), m_region(region), m_side(side) {}
};

class device_kernel_instruction : public instruction {
  public:
	explicit device_kernel_instruction(const instruction_id id, const device_id did, const task& tsk, const subrange<3>& execution_range)
	    : instruction(id, memory_id(did + 1)), m_tsk(tsk), m_execution_range(execution_range) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_device_kernel(*this); }

	const subrange<3>& get_execution_range() const { return m_execution_range; }

  private:
	const task& m_tsk;
	device_id m_device_id;
	subrange<3> m_execution_range;
};

class host_kernel_instruction : public instruction {
  public:
	explicit host_kernel_instruction(const instruction_id id, const task& tsk, const subrange<3>& execution_range)
	    : instruction(id, host_memory_id), m_tsk(tsk), m_execution_range(execution_range) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_host_kernel(*this); }

	const subrange<3>& get_execution_range() const { return m_execution_range; }

  private:
	const task& m_tsk;
	subrange<3> m_execution_range;
};

class send_instruction : public instruction {
  public:
	explicit send_instruction(const instruction_id id, const node_id to, const buffer_id bid, const subrange<3>& sr)
	    : instruction(id, host_memory_id), m_to(to), m_bid(bid), m_sr(sr) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_send(*this); }

	buffer_id get_buffer_id() const { return m_bid; }
	subrange<3> get_subrange() const { return m_sr; }

  private:
	node_id m_to;
	buffer_id m_bid;
	subrange<3> m_sr;
};

class recv_instruction : public instruction {
  public:
	explicit recv_instruction(const instruction_id id, const transfer_id transfer, const buffer_id bid, const GridRegion<3>& region)
	    : instruction(id, host_memory_id), m_transfer(transfer), m_bid(bid), m_region(region) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_recv(*this); }

	buffer_id get_buffer_id() const { return m_bid; }
	GridRegion<3> get_region() const { return m_region; }

  private:
	transfer_id m_transfer;
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class horizon_instruction : public instruction {
  public:
	using instruction::instruction;

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_horizon(*this); }
};

class epoch_instruction : public instruction {
  public:
	using instruction::instruction;

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_epoch(*this); }
};

inline void const_instruction_graph_visitor::visit_alloc(const alloc_instruction& ainsn) { visit(ainsn); }
inline void const_instruction_graph_visitor::visit_copy(const copy_instruction& cinsn) { visit(cinsn); }
inline void const_instruction_graph_visitor::visit_device_kernel(const device_kernel_instruction& dkinsn) { visit(dkinsn); }
inline void const_instruction_graph_visitor::visit_host_kernel(const host_kernel_instruction& hkinsn) { visit(hkinsn); }
inline void const_instruction_graph_visitor::visit_send(const send_instruction& sinsn) { visit(sinsn); }
inline void const_instruction_graph_visitor::visit_recv(const recv_instruction& rinsn) { visit(rinsn); }
inline void const_instruction_graph_visitor::visit_horizon(const horizon_instruction& hinsn) { visit(hinsn); }
inline void const_instruction_graph_visitor::visit_epoch(const epoch_instruction& einsn) { visit(einsn); }

class instruction_graph {
  public:
	void insert(std::unique_ptr<instruction> instr) { m_instructions.push_back(std::move(instr)); }

	template <typename Predicate>
	void erase_if(Predicate&& p) {
		const auto last = std::remove_if(m_instructions.begin(), m_instructions.end(),
		    [&p](const std::unique_ptr<instruction>& item) { return p(static_cast<const instruction*>(item.get())); });
		m_instructions.erase(last, m_instructions.end());
	}

	void visit(const_instruction_graph_visitor& visitor) const {
		for(const auto& insn : m_instructions) {
			insn->visit(visitor);
		}
	}

	void visit(const_instruction_graph_visitor&& visitor) const { visit(/* lvalue */ visitor); }

  private:
	// TODO split vector into epochs for cleanup phase
	std::vector<std::unique_ptr<instruction>> m_instructions;
};

} // namespace celerity::detail
