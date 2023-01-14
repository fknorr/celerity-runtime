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
class send_instruction;
class recv_instruction;
class epoch_instruction;

class const_instruction_graph_visitor {
  public:
	virtual ~const_instruction_graph_visitor() = default;

	virtual void visit(const instruction& insn) {}
	virtual void visit_alloc(const alloc_instruction& ainsn);
	virtual void visit_copy(const copy_instruction& cinsn);
	virtual void visit_device_kernel(const device_kernel_instruction& dkinsn);
	virtual void visit_send(const send_instruction& sinsn);
	virtual void visit_recv(const recv_instruction& rinsn);
	virtual void visit_epoch(const epoch_instruction& einsn);
};

class instruction : public intrusive_graph_node<instruction> {
  public:
	explicit instruction(memory_id mid) : m_mid(mid) {}

	instruction(const instruction&) = delete;
	instruction& operator=(const instruction&) = delete;
	virtual ~instruction() = default;

	virtual void visit(const_instruction_graph_visitor& visitor) const = 0;

	memory_id get_memory() const { return m_mid; }

  private:
	memory_id m_mid;
};

class alloc_instruction : public instruction {
  public:
	explicit alloc_instruction(const memory_id mid, const buffer_id bid, GridRegion<3> region) : instruction(mid), m_bid(bid), m_region(region) {}

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

	static std::pair<std::unique_ptr<copy_instruction>, std::unique_ptr<copy_instruction>> make_pair(
	    const memory_id source_mid, const memory_id dest_mid, const buffer_id bid, const subrange<3> sr) {
		std::unique_ptr<copy_instruction> source(new copy_instruction(source_mid, bid, sr, side::source));
		std::unique_ptr<copy_instruction> dest(new copy_instruction(dest_mid, bid, sr, side::dest));
		source->m_counterpart = dest.get();
		dest->m_counterpart = source.get();
		return std::pair(std::move(source), std::move(dest));
	}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_copy(*this); }

	const buffer_id& get_buffer_id() const { return m_bid; }
	const subrange<3>& get_subrange() const { return m_sr; }
	memory_id get_source_memory() const { return m_side == side::source ? get_memory() : m_counterpart->get_memory(); }
	memory_id get_dest_memory() const { return m_side == side::dest ? get_memory() : m_counterpart->get_memory(); }
	side get_side() const { return m_side; }
	instruction& get_counterpart() const { return *m_counterpart; }

  private:
	buffer_id m_bid;
	subrange<3> m_sr;
	side m_side;
	instruction* m_counterpart;

	explicit copy_instruction(memory_id mid, buffer_id bid, subrange<3> sr, side side) : instruction(mid), m_bid(bid), m_sr(sr), m_side(side) {}
};

class device_kernel_instruction : public instruction {
  public:
	explicit device_kernel_instruction(const device_id did, const task& tsk, const subrange<3>& execution_range)
	    : instruction(memory_id(did + 1)), m_tsk(tsk), m_execution_range(execution_range) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_device_kernel(*this); }

	const subrange<3>& get_execution_range() const { return m_execution_range; }

  private:
	const task& m_tsk;
	device_id m_device_id;
	subrange<3> m_execution_range;
};

class send_instruction : public instruction {
  public:
	explicit send_instruction(const node_id to, const buffer_id bid, const subrange<3>& sr) : instruction(host_memory_id), m_to(to), m_bid(bid), m_sr(sr) {}

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
	explicit recv_instruction(const transfer_id transfer, const buffer_id bid, const GridRegion<3>& region)
	    : instruction(host_memory_id), m_transfer(transfer), m_bid(bid), m_region(region) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_recv(*this); }

	buffer_id get_buffer_id() const { return m_bid; }

	GridRegion<3> get_region() const { return m_region; }

  private:
	transfer_id m_transfer;
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class epoch_instruction : public instruction {
  public:
	using instruction::instruction;

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_epoch(*this); }
};

inline void const_instruction_graph_visitor::visit_alloc(const alloc_instruction& ainsn) { visit(ainsn); }
inline void const_instruction_graph_visitor::visit_copy(const copy_instruction& cinsn) { visit(cinsn); }
inline void const_instruction_graph_visitor::visit_device_kernel(const device_kernel_instruction& dkinsn) { visit(dkinsn); }
inline void const_instruction_graph_visitor::visit_send(const send_instruction& sinsn) { visit(sinsn); }
inline void const_instruction_graph_visitor::visit_recv(const recv_instruction& rinsn) { visit(rinsn); }
inline void const_instruction_graph_visitor::visit_epoch(const epoch_instruction& einsn) { visit(einsn); }

class instruction_graph {
  public:
	template <typename Instruction, typename... CtorParams>
	Instruction& create(CtorParams&&... ctor_args) {
		auto insn = std::make_unique<Instruction>(std::forward<CtorParams>(ctor_args)...);
		auto& ref = *insn;
		m_instructions.emplace_back(std::move(insn));
		return ref;
	}

	std::pair<copy_instruction&, copy_instruction&> create_copy(
	    const memory_id source_mid, const memory_id dest_mid, const buffer_id bid, const subrange<3> sr) {
		auto [source, dest] = copy_instruction::make_pair(source_mid, dest_mid, bid, sr);
		auto &source_ref = *source, &dest_ref = *dest;
		m_instructions.emplace_back(std::move(source));
		m_instructions.emplace_back(std::move(dest));
		return {source_ref, dest_ref};
	}

	static void add_dependency(instruction& from, instruction& to, const dependency_kind kind, const dependency_origin origin) {
		from.add_dependency({&to, kind, origin});
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
