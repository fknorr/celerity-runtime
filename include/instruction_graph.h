#pragma once

#include "grid.h"
#include "instruction_backend.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h" // TODO the only dependencies on task.h are launcher types and contiguous_box_set, consider moving those
#include "types.h"
#include "utils.h"

#include <unordered_map>

namespace celerity::detail {

class instruction;
class alloc_instruction;
class free_instruction;
class copy_instruction;
class kernel_instruction;
class sycl_kernel_instruction;
class host_kernel_instruction;
class send_instruction;
class recv_instruction;
class horizon_instruction;
class epoch_instruction;

class instruction : public intrusive_graph_node<instruction> {
  public:
	using const_visitor = utils::visitor<const alloc_instruction&, const free_instruction&, const copy_instruction&, const sycl_kernel_instruction&,
	    const host_kernel_instruction&, const send_instruction&, const recv_instruction&, const horizon_instruction&, const epoch_instruction&>;

	explicit instruction(const instruction_id iid) : m_id(iid) {}

	virtual ~instruction() = default;

	virtual void accept(const_visitor& visitor) const = 0;
	virtual instruction_backend get_backend() const = 0;

	instruction_id get_id() const { return m_id; }

  private:
	instruction_id m_id;
};

struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
};

class alloc_instruction final : public instruction {
  public:
	explicit alloc_instruction(
	    const instruction_id iid, const instruction_backend backend, const allocation_id aid, const memory_id mid, const size_t size, const size_t alignment)
	    : instruction(iid), m_backend(backend), m_aid(aid), m_mid(mid), m_size(size), m_alignment(alignment) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return m_backend; }

	allocation_id get_allocation_id() const { return m_aid; }
	memory_id get_memory_id() const { return m_mid; }
	size_t get_size() const { return m_size; }
	size_t get_alignment() const { return m_alignment; }

  private:
	instruction_backend m_backend;
	allocation_id m_aid;
	memory_id m_mid;
	size_t m_size;
	size_t m_alignment;
};

class free_instruction final : public instruction {
  public:
	explicit free_instruction(const instruction_id iid, const instruction_backend backend, const allocation_id aid)
	    : instruction(iid), m_backend(backend), m_aid(aid) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return m_backend; }

	allocation_id get_allocation_id() const { return m_aid; }

  private:
	instruction_backend m_backend;
	allocation_id m_aid;
};

// copy_instruction: either copy or linearize
// TODO maybe template this on Dims?
class copy_instruction final : public instruction {
  public:
	explicit copy_instruction(const instruction_id iid, const instruction_backend backend, const int dims, const memory_id source_memory,
	    const allocation_id source_allocation, const range<3>& source_range, const id<3>& offset_in_source, const memory_id dest_memory,
	    const allocation_id dest_allocation, const range<3>& dest_range, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size)
	    : instruction(iid), m_backend(backend), m_source_memory(source_memory), m_source_allocation(source_allocation), m_dest_memory(dest_memory),
	      m_dest_allocation(dest_allocation), m_dims(dims), m_source_range(source_range), m_dest_range(dest_range), m_offset_in_source(offset_in_source),
	      m_offset_in_dest(offset_in_dest), m_copy_range(copy_range), m_elem_size(elem_size) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return m_backend; }

	memory_id get_source_memory() const { return m_source_memory; }
	allocation_id get_source_allocation() const { return m_source_allocation; }
	int get_dimensions() const { return m_dims; }
	const range<3>& get_source_range() const { return m_source_range; }
	const id<3>& get_offset_in_source() const { return m_offset_in_source; }
	memory_id get_dest_memory() const { return m_dest_memory; }
	allocation_id get_dest_allocation() const { return m_dest_allocation; }
	const range<3>& get_dest_range() const { return m_dest_range; }
	const id<3>& get_offset_in_dest() const { return m_offset_in_dest; }
	const range<3>& get_copy_range() const { return m_copy_range; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	instruction_backend m_backend;
	memory_id m_source_memory;
	allocation_id m_source_allocation;
	memory_id m_dest_memory;
	allocation_id m_dest_allocation;
	int m_dims;
	range<3> m_source_range;
	range<3> m_dest_range;
	id<3> m_offset_in_source;
	id<3> m_offset_in_dest;
	range<3> m_copy_range;
	size_t m_elem_size;
};

struct reads_writes {
	region<3> reads;
	region<3> writes;
	region<3> invalidations;
	bounding_box_set contiguous_boxes;

	bool empty() const { return reads.empty() && writes.empty(); }
};

struct access_allocation {
	allocation_id aid;
	range<3> allocation_range = detail::zeros;
	id<3> offset_in_allocation = detail::zeros;
	subrange<3> buffer_subrange;
};

using access_allocation_map = std::vector<access_allocation>;

// TODO maybe overhaul buffer_access_map to provide this functionality?
using buffer_read_write_map = std::unordered_map<buffer_id, reads_writes>;

// TODO is a common base class for host and device "kernels" the right thing to do? On the host these are not called kernels but "host tasks" everywhere else.
class kernel_instruction : public instruction {
  public:
	explicit kernel_instruction(const instruction_id iid, const subrange<3>& execution_range, access_allocation_map allocation_map)
	    : instruction(iid), m_allocation_map(std::move(allocation_map)) {}

	const subrange<3>& get_execution_range() const { return m_execution_range; }
	const access_allocation_map& get_allocation_map() const { return m_allocation_map; }

  private:
	subrange<3> m_execution_range;
	access_allocation_map m_allocation_map;
};

class sycl_kernel_instruction final : public kernel_instruction {
  public:
	explicit sycl_kernel_instruction(
	    const instruction_id iid, const device_id did, sycl_kernel_launcher launcher, const subrange<3>& execution_range, access_allocation_map allocation_map)
	    : kernel_instruction(iid, execution_range, std::move(allocation_map)), m_device_id(did), m_launcher(std::move(launcher)) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::sycl; }

	device_id get_device_id() const { return m_device_id; }
	void launch(sycl::handler& cgh) const {
		m_launcher(cgh, get_execution_range()); // TODO where does m_allocation_map go?
	}

  private:
	device_id m_device_id;
	sycl_kernel_launcher m_launcher;
};

class host_kernel_instruction final : public kernel_instruction {
  public:
	using kernel_instruction::kernel_instruction;
	host_kernel_instruction(const instruction_id iid, host_task_launcher launcher, const subrange<3>& execution_range, const range<3>& global_range,
	    access_allocation_map allocation_map)
	    : kernel_instruction(iid, execution_range, std::move(allocation_map)), m_launcher(std::move(launcher)), m_global_range(global_range) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::host; }
	const range<3>& get_global_range() const { return m_global_range; }

	std::function<void()> bind(MPI_Comm comm) const {
		return [l = m_launcher, er = get_execution_range(), gr = m_global_range, comm] { l(er, gr, comm); };
	}

  private:
	host_task_launcher m_launcher;
	range<3> m_global_range;
};

struct pilot_message {
	int tag;
	buffer_id buffer;
	transfer_id transfer;
	box<3> box;
};

class send_instruction final : public instruction {
  public:
	explicit send_instruction(
	    const instruction_id iid, const transfer_id trid, const node_id to_nid, const int tag, const allocation_id aid, const size_t bytes)
	    : instruction(iid), m_transfer_id(trid), m_to_nid(to_nid), m_tag(tag), m_aid(aid), m_bytes(bytes) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::mpi; }

	transfer_id get_transfer_id() const { return m_transfer_id; }
	node_id get_dest_node_id() const { return m_to_nid; }
	int get_tag() const { return m_tag; }
	allocation_id get_allocation_id() const { return m_aid; }
	size_t get_size_bytes() const { return m_bytes; }
	// TODO offset_bytes, if we send directly from a host buffer allocation?

  private:
	transfer_id m_transfer_id;
	node_id m_to_nid;
	int m_tag;
	allocation_id m_aid;
	size_t m_bytes;
};

class recv_instruction final : public instruction {
  public:
	explicit recv_instruction(const instruction_id iid, const buffer_id bid, const transfer_id trid, const memory_id dest_memory,
	    const allocation_id dest_allocation, const int dims, const range<3>& alloc_range, const id<3>& offset_in_alloc, const id<3>& offset_in_buffer,
	    const range<3>& recv_range, const size_t elem_size)
	    : instruction(iid), m_buffer_id(bid), m_transfer_id(trid), m_dest_memory(dest_memory), m_dest_allocation(dest_allocation), m_dims(dims),
	      m_alloc_range(alloc_range), m_offset_in_alloc(offset_in_alloc), m_offset_in_buffer(offset_in_buffer), m_recv_range(recv_range),
	      m_elem_size(elem_size) {}

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::mpi; }

	buffer_id get_buffer_id() const { return m_buffer_id; }
	transfer_id get_transfer_id() const { return m_transfer_id; }
	allocation_id get_dest_allocation_id() const { return m_dest_allocation; }
	memory_id get_dest_memory_id() const { return m_dest_memory; }
	int get_dimensions() const { return m_dims; }
	const range<3>& get_allocation_range() const { return m_alloc_range; }
	const id<3>& get_offset_in_allocation() const { return m_offset_in_alloc; }
	const id<3>& get_offset_in_buffer() const { return m_offset_in_buffer; }
	const range<3>& get_recv_range() const { return m_recv_range; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	buffer_id m_buffer_id;
	transfer_id m_transfer_id;
	memory_id m_dest_memory;
	allocation_id m_dest_allocation;
	int m_dims;
	range<3> m_alloc_range;
	id<3> m_offset_in_alloc;
	id<3> m_offset_in_buffer;
	range<3> m_recv_range;
	size_t m_elem_size;
};

class horizon_instruction final : public instruction {
  public:
	explicit horizon_instruction(const instruction_id iid, const task_id horizon_tid) : instruction(iid), m_horizon_tid(horizon_tid) {}

	task_id get_horizon_task_id() const { return m_horizon_tid; }

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::host; }

  private:
	task_id m_horizon_tid;
};

class epoch_instruction final : public instruction {
  public:
	explicit epoch_instruction(const instruction_id iid, const task_id epoch_tid) : instruction(iid), m_epoch_tid(epoch_tid) {}

	task_id get_epoch_task_id() const { return m_epoch_tid; }

	void accept(const_visitor& visitor) const override { visitor.visit(*this); }
	instruction_backend get_backend() const override { return instruction_backend::host; }

  private:
	task_id m_epoch_tid;
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
