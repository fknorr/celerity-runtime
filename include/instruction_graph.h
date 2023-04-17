#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "utils.h"

#include <unordered_map>

namespace celerity::detail {

class instruction;
class alloc_instruction;
class free_instruction;
class copy_instruction;
class kernel_instruction;
class device_kernel_instruction;
class host_kernel_instruction;
class send_instruction;
class recv_instruction;
class horizon_instruction;
class epoch_instruction;

class instruction : public intrusive_graph_node<instruction> {
  public:
	using const_visitor = utils::visitor<const alloc_instruction&, const free_instruction&, const copy_instruction&, const device_kernel_instruction&,
	    const host_kernel_instruction&, const send_instruction&, const recv_instruction&, const horizon_instruction&, const epoch_instruction&>;

	explicit instruction(const instruction_id iid, const std::optional<command_id> cid = std::nullopt) : m_id(iid), m_cid(cid) {}

	instruction(const instruction&) = delete;
	instruction& operator=(const instruction&) = delete;
	virtual ~instruction() = default;

	virtual void accept(const_visitor& visitor) const = 0;

	instruction_id get_id() const { return m_id; }
	std::optional<command_id> get_command_id() const { return m_cid; }

  private:
	instruction_id m_id;
	std::optional<command_id> m_cid;
};

struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
};

struct nd_allocation {
	allocation_id allocation_id;
	int dimensions = 0;
	range<3> range = zero_range;
	size_t elem_size = 0;
};

class alloc_instruction final : public instruction {
  public:
	explicit alloc_instruction(const instruction_id iid, const allocation_id aid, const memory_id mid, const size_t size, const size_t alignment)
	    : instruction(iid), m_aid(aid), m_mid(mid), m_size(size), m_alignment(alignment) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

	allocation_id get_allocation_id() const { return m_aid; }
	memory_id get_memory_id() const { return m_mid; }
	size_t get_size() const { return m_size; }
	size_t get_alignment() const { return m_alignment; }

  private:
	allocation_id m_aid;
	memory_id m_mid;
	size_t m_size;
	size_t m_alignment;
};

class free_instruction final : public instruction {
  public:
	explicit free_instruction(const instruction_id iid, const allocation_id aid) : instruction(iid), m_aid(aid) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

	allocation_id get_allocation_id() const { return m_aid; }

  private:
	allocation_id m_aid;
};

class copy_instruction final : public instruction {
  public:
	explicit copy_instruction(const instruction_id iid, allocation_id source, const id<3>& source_offset, allocation_id dest, const id<3>& dest_offset,
	    int dims, const range<3>& range, const size_t elem_size)
	    : instruction(iid), m_source(source), m_source_offset(source_offset), m_dest(dest), m_dest_offset(dest_offset), m_dims(dims), m_range(range),
	      m_elem_size(elem_size) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

	allocation_id get_source() const { return m_source; }
	const id<3>& get_source_offset() const { return m_source_offset; }
	allocation_id get_dest() const { return m_dest; }
	const id<3>& get_dest_offset() const { return m_dest_offset; }
	int get_dimensions() const { return m_dims; }
	const range<3>& get_range() const { return m_range; }
	size_t get_elem_size() const { return m_elem_size; }

  private:
	allocation_id m_source;
	id<3> m_source_offset;
	allocation_id m_dest;
	id<3> m_dest_offset;
	int m_dims;
	range<3> m_range;
	size_t m_elem_size;
};

struct reads_writes {
	GridRegion<3> reads;
	GridRegion<3> writes;
	contiguous_box_set contiguous_boxes;

	bool empty() const { return reads.empty() && writes.empty(); }
};

// TODO this should eventually supersede buffer_access_map (we don't need per-access-mode granularity anywhere)
using buffer_read_write_map = std::unordered_map<buffer_id, reads_writes>;

class kernel_instruction : public instruction {
  public:
	explicit kernel_instruction(const instruction_id iid, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map)
	    : instruction(iid, cid), m_execution_range(execution_range), m_rw_map(std::move(rw_map)) {}

	const subrange<3>& get_execution_range() const { return m_execution_range; }

	const buffer_read_write_map& get_buffer_read_write_map() const { return m_rw_map; }

  private:
	subrange<3> m_execution_range;
	buffer_read_write_map m_rw_map;
};

class device_kernel_instruction final : public kernel_instruction {
  public:
	explicit device_kernel_instruction(
	    const instruction_id iid, const device_id did, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map)
	    : kernel_instruction(iid, cid, execution_range, std::move(rw_map)), m_device_id(did) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

	device_id get_device_id() const { return m_device_id; }

  private:
	device_id m_device_id;
};

class host_kernel_instruction final : public kernel_instruction {
  public:
	explicit host_kernel_instruction(const instruction_id iid, const command_id cid, const subrange<3>& execution_range, buffer_read_write_map rw_map,
	    side_effect_map se_map, collective_group_id cgid)
	    : kernel_instruction(iid, cid, execution_range, std::move(rw_map)), m_se_map(std::move(se_map)), m_cgid(cgid) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

	const side_effect_map& get_side_effect_map() const { return m_se_map; }

	collective_group_id get_collective_group_id() const { return m_cgid; }

  private:
	side_effect_map m_se_map;
	collective_group_id m_cgid;
};

class send_instruction final : public instruction {
  public:
	explicit send_instruction(const instruction_id iid, const command_id cid, const node_id to_nid, const buffer_id bid, GridRegion<3> region)
	    : instruction(iid, cid), m_to_nid(to_nid), m_bid(bid), m_region(std::move(region)) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

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
	explicit recv_instruction(const instruction_id iid, const transfer_id trid, const buffer_id bid, GridRegion<3> region)
	    : instruction(iid), m_trid(trid), m_bid(bid), m_region(region) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }

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
	explicit horizon_instruction(const instruction_id iid, const command_id cid) : instruction(iid, cid) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
};

class epoch_instruction final : public instruction {
  public:
	explicit epoch_instruction(const instruction_id iid, const command_id cid) : instruction(iid, cid) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
};

class instruction_graph {
  public:
	void insert(std::unique_ptr<instruction> instr) { m_instructions.push_back(std::move(instr)); }

	template <typename Predicate>
	void erase_if(Predicate&& p) {
		const auto last = std::remove_if(m_instructions.begin(), m_instructions.end(),
		    [&p](const std::unique_ptr<instruction>& item) { return p(static_cast<const instruction*>(item.get())); });
		m_instructions.erase(last, m_instructions.end());
	}

	template <typename Fn>
	void for_each(Fn&& fn) const {
		for(const auto& instr : m_instructions) {
			fn(std::as_const(*instr));
		}
	}

  private:
	// TODO split vector into epochs for cleanup phase
	std::vector<std::unique_ptr<instruction>> m_instructions;
};

} // namespace celerity::detail
