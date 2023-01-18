#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h"
#include "types.h"

#include <unordered_map>

namespace celerity::detail {

class instruction;
class alloc_instruction;
class copy_instruction;
class kernel_instruction;
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
	virtual void visit_kernel(const kernel_instruction& kinsn);
	virtual void visit_device_kernel(const device_kernel_instruction& dkinsn);
	virtual void visit_host_kernel(const host_kernel_instruction& hkinsn);
	virtual void visit_send(const send_instruction& sinsn);
	virtual void visit_recv(const recv_instruction& rinsn);
	virtual void visit_horizon(const horizon_instruction& hinsn);
	virtual void visit_epoch(const epoch_instruction& einsn);
};

class instruction : public intrusive_graph_node<instruction> {
  public:
	explicit instruction(const instruction_id id, const std::optional<command_id> cid = std::nullopt) : m_id(id), m_cid(cid) {}

	instruction(const instruction&) = delete;
	instruction& operator=(const instruction&) = delete;
	virtual ~instruction() = default;

	virtual void visit(const_instruction_graph_visitor& visitor) const = 0;

	instruction_id get_id() const { return m_id; }
	std::optional<command_id> get_command_id() const { return m_cid; }

  private:
	instruction_id m_id;
	std::optional<command_id> m_cid;
};

struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
};

class alloc_instruction final : public instruction {
  public:
	explicit alloc_instruction(const instruction_id id, const buffer_id bid, const memory_id mid, GridRegion<3> region)
	    : instruction(id), m_bid(bid), m_mid(mid), m_region(std::move(region)) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_alloc(*this); }

	buffer_id get_buffer_id() const { return m_bid; }
	memory_id get_memory_id() const { return m_mid; }
	GridRegion<3> get_region() const { return m_region; }

  private:
	buffer_id m_bid;
	memory_id m_mid;
	GridRegion<3> m_region;
};

class copy_instruction final : public instruction {
  public:
	explicit copy_instruction(const instruction_id id, const buffer_id bid, GridRegion<3> region, const memory_id from_mid, const memory_id to_mid)
	    : instruction(id), m_bid(bid), m_region(std::move(region)), m_from_mid(from_mid), m_to_mid(to_mid) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_copy(*this); }

	const buffer_id& get_buffer_id() const { return m_bid; }
	const GridRegion<3>& get_region() const { return m_region; }
	memory_id get_source_memory_id() const { return m_from_mid; }
	memory_id get_dest_memory_id() const { return m_to_mid; }

  private:
	buffer_id m_bid;
	GridRegion<3> m_region;
	memory_id m_from_mid;
	memory_id m_to_mid;
};


struct reads_writes {
	GridRegion<3> reads;
	GridRegion<3> writes;

	bool empty() const { return reads.empty() && writes.empty(); }
};

// TODO this should eventually supersede buffer_access_map (we don't need per-access-mode granularity anywhere)
using buffer_read_write_map = std::unordered_map<buffer_id, reads_writes>;

class kernel_instruction : public instruction {
  public:
	explicit kernel_instruction(const instruction_id id, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map)
	    : instruction(id, cid), m_execution_range(execution_range), m_rw_map(std::move(rw_map)) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_kernel(*this); }

	const subrange<3>& get_execution_range() const { return m_execution_range; }

	const buffer_read_write_map& get_buffer_read_write_map() const { return m_rw_map; }

  private:
	subrange<3> m_execution_range;
	buffer_read_write_map m_rw_map;
};

class device_kernel_instruction final : public kernel_instruction {
  public:
	explicit device_kernel_instruction(
	    const instruction_id id, const device_id did, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map)
	    : kernel_instruction(id, cid, execution_range, std::move(rw_map)), m_device_id(did) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_device_kernel(*this); }

	device_id get_device_id() const { return m_device_id; }

  private:
	device_id m_device_id;
};

class host_kernel_instruction final : public kernel_instruction {
  public:
	explicit host_kernel_instruction(const instruction_id id, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map,
	    side_effect_map se_map, collective_group_id cgid)
	    : kernel_instruction(id, cid, execution_range, std::move(rw_map)), m_se_map(std::move(se_map)), m_cgid(cgid) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_host_kernel(*this); }

	const side_effect_map& get_side_effect_map() const { return m_se_map; }

	collective_group_id get_collective_group_id() const { return m_cgid; }

  private:
	side_effect_map m_se_map;
	collective_group_id m_cgid;
};

class send_instruction final : public instruction {
  public:
	explicit send_instruction(const instruction_id id, const command_id cid, const node_id to_nid, const buffer_id bid, GridRegion<3> region)
	    : instruction(id, cid), m_to_nid(to_nid), m_bid(bid), m_region(std::move(region)) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_send(*this); }

	node_id get_dest_node_id() const { return m_to_nid; }
	buffer_id get_buffer_id() const { return m_bid; }
	GridRegion<3> get_region() const { return m_region; }

  private:
	node_id m_to_nid;
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class recv_instruction final : public instruction {
  public:
	// We don't make the effort of tracking the command ids of (pending) await-pushes
	explicit recv_instruction(const instruction_id id, const transfer_id trid, const buffer_id bid, GridRegion<3> region)
	    : instruction(id), m_trid(trid), m_bid(bid), m_region(region) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_recv(*this); }

	transfer_id get_transfer_id() const { return m_trid; }
	buffer_id get_buffer_id() const { return m_bid; }
	GridRegion<3> get_region() const { return m_region; }

  private:
	transfer_id m_trid;
	buffer_id m_bid;
	GridRegion<3> m_region;
};

class horizon_instruction final : public instruction {
  public:
	explicit horizon_instruction(const instruction_id id, const command_id cid) : instruction(id, cid) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_horizon(*this); }
};

class epoch_instruction final : public instruction {
  public:
	explicit epoch_instruction(const instruction_id id, const command_id cid) : instruction(id, cid) {}

	void visit(const_instruction_graph_visitor& visitor) const override { visitor.visit_epoch(*this); }
};

inline void const_instruction_graph_visitor::visit_alloc(const alloc_instruction& ainsn) { visit(ainsn); }
inline void const_instruction_graph_visitor::visit_copy(const copy_instruction& cinsn) { visit(cinsn); }
inline void const_instruction_graph_visitor::visit_kernel(const kernel_instruction& dkinsn) { visit(dkinsn); }
inline void const_instruction_graph_visitor::visit_device_kernel(const device_kernel_instruction& dkinsn) { visit_kernel(dkinsn); }
inline void const_instruction_graph_visitor::visit_host_kernel(const host_kernel_instruction& hkinsn) { visit_kernel(hkinsn); }
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
