#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h" // TODO the only dependencies on task.h are launcher types and contiguous_box_set, consider moving those
#include "types.h"
#include "utils.h"

#include <unordered_map>

#include <matchbox.hh>

namespace celerity::detail {

class instruction : public intrusive_graph_node<instruction>,
                    public matchbox::acceptor<class alloc_instruction, class free_instruction, class init_buffer_instruction, class export_instruction,
                        class copy_instruction, class sycl_kernel_instruction, class host_kernel_instruction, class send_instruction, class recv_instruction,
                        class fence_instruction, class destroy_host_object_instruction, class horizon_instruction, class epoch_instruction> {
  public:
	explicit instruction(const instruction_id iid) : m_id(iid) {}

	instruction_id get_id() const { return m_id; }

  private:
	instruction_id m_id;
};

struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
};

class alloc_instruction final : public matchbox::implement_acceptor<instruction, alloc_instruction> {
  public:
	explicit alloc_instruction(const instruction_id iid, const allocation_id aid, const memory_id mid, const size_t size, const size_t alignment)
	    : acceptor_base(iid), m_aid(aid), m_mid(mid), m_size(size), m_alignment(alignment) {}

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

class free_instruction final : public matchbox::implement_acceptor<instruction, free_instruction> {
  public:
	explicit free_instruction(const instruction_id iid, const allocation_id aid) : acceptor_base(iid), m_aid(aid) {}

	allocation_id get_allocation_id() const { return m_aid; }

  private:
	allocation_id m_aid;
};

// TODO temporary until IGGEN supports user_memory_id and multi-hop copies (then this will become a copy_instruction)
class init_buffer_instruction final : public matchbox::implement_acceptor<instruction, init_buffer_instruction> {
  public:
	explicit init_buffer_instruction(const instruction_id iid, const buffer_id bid, const allocation_id host_aid, const size_t size_bytes)
	    : acceptor_base(iid), m_bid(bid), m_host_aid(host_aid), m_size_bytes(size_bytes) {}

	buffer_id get_buffer_id() const { return m_bid; }
	allocation_id get_host_allocation_id() const { return m_host_aid; }
	size_t get_size() const { return m_size_bytes; }

  private:
	buffer_id m_bid;
	allocation_id m_host_aid;
	size_t m_size_bytes;
};

// TODO temporary until IGGEN supports user_memory_id (then this will become a copy_instruction)
class export_instruction final : public matchbox::implement_acceptor<instruction, export_instruction> {
  public:
	explicit export_instruction(const instruction_id iid, const allocation_id host_aid, const int dims, const range<3>& allocation_range,
	    const id<3>& offset_in_allocation, const range<3>& copy_range, size_t elem_size, void* out_pointer)
	    : acceptor_base(iid), m_host_aid(host_aid), m_dims(dims), m_allocation_range(allocation_range), m_offset_in_allocation(offset_in_allocation),
	      m_copy_range(copy_range), m_elem_size(elem_size), m_out_pointer(out_pointer) {}

	allocation_id get_host_allocation_id() const { return m_host_aid; }
	int get_dimensions() const { return m_dims; }
	range<3> get_allocation_range() const { return m_allocation_range; }
	id<3> get_offset_in_allocation() const { return m_offset_in_allocation; }
	range<3> get_copy_range() const { return m_copy_range; }
	size_t get_element_size() const { return m_elem_size; }
	void* get_out_pointer() const { return m_out_pointer; }

  private:
	allocation_id m_host_aid;
	int m_dims; // TODO does this actually need to know dimensions or can we just copy by effective_dims?
	range<3> m_allocation_range;
	id<3> m_offset_in_allocation;
	range<3> m_copy_range;
	size_t m_elem_size;
	void* m_out_pointer; // TODO very naughty
};

// copy_instruction: either copy or linearize
// TODO maybe template this on Dims?
class copy_instruction final : public matchbox::implement_acceptor<instruction, copy_instruction> {
  public:
	explicit copy_instruction(const instruction_id iid, const int dims, const memory_id source_memory, const allocation_id source_allocation,
	    const range<3>& source_range, const id<3>& offset_in_source, const memory_id dest_memory, const allocation_id dest_allocation,
	    const range<3>& dest_range, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size)
	    : acceptor_base(iid), m_source_memory(source_memory), m_source_allocation(source_allocation), m_dest_memory(dest_memory),
	      m_dest_allocation(dest_allocation), m_dims(dims), m_source_range(source_range), m_dest_range(dest_range), m_offset_in_source(offset_in_source),
	      m_offset_in_dest(offset_in_dest), m_copy_range(copy_range), m_elem_size(elem_size) {}

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
	memory_id m_source_memory;
	allocation_id m_source_allocation;
	memory_id m_dest_memory;
	allocation_id m_dest_allocation;
	int m_dims; // TODO does this actually need to know dimensions or can we just copy by effective_dims?
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
	box<3> allocated_box_in_buffer;
	box<3> accessed_box_in_buffer;
};

using access_allocation_map = std::vector<access_allocation>;

// TODO maybe overhaul buffer_access_map to provide this functionality?
using buffer_read_write_map = std::unordered_map<buffer_id, reads_writes>;

// TODO is a common base class for host and device "kernels" the right thing to do? On the host these are not called kernels but "host tasks" everywhere else.
class kernel_instruction : public instruction {
  public:
	explicit kernel_instruction(const instruction_id iid, const subrange<3>& execution_range, access_allocation_map allocation_map)
	    : instruction(iid), m_execution_range(execution_range), m_allocation_map(std::move(allocation_map)) {}

	const subrange<3>& get_execution_range() const { return m_execution_range; }
	const access_allocation_map& get_allocation_map() const { return m_allocation_map; }

  private:
	subrange<3> m_execution_range;
	access_allocation_map m_allocation_map;
};

class sycl_kernel_instruction final : public matchbox::implement_acceptor<kernel_instruction, sycl_kernel_instruction> {
  public:
	explicit sycl_kernel_instruction(
	    const instruction_id iid, const device_id did, sycl_kernel_launcher launcher, const subrange<3>& execution_range, access_allocation_map allocation_map)
	    : acceptor_base(iid, execution_range, std::move(allocation_map)), m_device_id(did), m_launcher(std::move(launcher)) {}

	device_id get_device_id() const { return m_device_id; }

	const sycl_kernel_launcher& get_launcher() const { return m_launcher; }

  private:
	device_id m_device_id;
	sycl_kernel_launcher m_launcher;
};

// TODO rename to host_task_instruction
class host_kernel_instruction final : public matchbox::implement_acceptor<kernel_instruction, host_kernel_instruction> {
  public:
	using acceptor_base::acceptor_base;

	host_kernel_instruction(const instruction_id iid, host_task_launcher launcher, const subrange<3>& execution_range, const range<3>& global_range,
	    access_allocation_map allocation_map)
	    : acceptor_base(iid, execution_range, std::move(allocation_map)), m_launcher(std::move(launcher)), m_global_range(global_range) {}

	const range<3>& get_global_range() const { return m_global_range; }

	const host_task_launcher& get_launcher() const { return m_launcher; }

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

struct outbound_pilot {
	node_id to;
	pilot_message message;
};

struct inbound_pilot {
	node_id from;
	pilot_message message;
};

class send_instruction final : public matchbox::implement_acceptor<instruction, send_instruction> {
  public:
	explicit send_instruction(const instruction_id iid, const transfer_id trid, const node_id to_nid, const int tag, const allocation_id source_allocation,
	    const range<3>& alloc_range, const id<3>& offset_in_alloc, const range<3>& send_range, const size_t elem_size)
	    : acceptor_base(iid), m_transfer_id(trid), m_to_nid(to_nid), m_tag(tag), m_source_allocation(source_allocation), m_alloc_range(alloc_range),
	      m_offset_in_alloc(offset_in_alloc), m_send_range(send_range), m_elem_size(elem_size) {}

	transfer_id get_transfer_id() const { return m_transfer_id; }
	node_id get_dest_node_id() const { return m_to_nid; }
	int get_tag() const { return m_tag; }
	allocation_id get_source_allocation_id() const { return m_source_allocation; }
	const range<3>& get_allocation_range() const { return m_alloc_range; }
	const id<3>& get_offset_in_allocation() const { return m_offset_in_alloc; }
	const range<3>& get_send_range() const { return m_send_range; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	transfer_id m_transfer_id;
	node_id m_to_nid;
	int m_tag;
	allocation_id m_source_allocation;
	range<3> m_alloc_range;
	id<3> m_offset_in_alloc;
	range<3> m_send_range;
	size_t m_elem_size;
};

class recv_instruction final : public matchbox::implement_acceptor<instruction, recv_instruction> {
  public:
	explicit recv_instruction(const instruction_id iid, const buffer_id bid, const transfer_id trid, const memory_id dest_memory,
	    const allocation_id dest_allocation, const range<3>& alloc_range, const id<3>& offset_in_alloc, const id<3>& offset_in_buffer,
	    const range<3>& recv_range, const size_t elem_size)
	    : acceptor_base(iid), m_buffer_id(bid), m_transfer_id(trid), m_dest_memory(dest_memory), m_dest_allocation(dest_allocation), m_alloc_range(alloc_range),
	      m_offset_in_alloc(offset_in_alloc), m_offset_in_buffer(offset_in_buffer), m_recv_range(recv_range), m_elem_size(elem_size) {}

	buffer_id get_buffer_id() const { return m_buffer_id; }
	transfer_id get_transfer_id() const { return m_transfer_id; }
	allocation_id get_dest_allocation_id() const { return m_dest_allocation; }
	memory_id get_dest_memory_id() const { return m_dest_memory; }
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
	range<3> m_alloc_range;
	id<3> m_offset_in_alloc;
	id<3> m_offset_in_buffer;
	range<3> m_recv_range;
	size_t m_elem_size;
};

class fence_instruction final : public matchbox::implement_acceptor<instruction, fence_instruction> {
  public:
	explicit fence_instruction(const instruction_id iid, fence_promise* promise) : acceptor_base(iid), m_promise(promise) {}

	fence_promise* get_promise() const { return m_promise; };

  private:
	fence_promise* m_promise;
};

class destroy_host_object_instruction final : public matchbox::implement_acceptor<instruction, destroy_host_object_instruction> {
  public:
	explicit destroy_host_object_instruction(const instruction_id iid, const host_object_id hoid) : acceptor_base(iid), m_hoid(hoid) {}

	host_object_id get_host_object_id() const { return m_hoid; }

  private:
	host_object_id m_hoid;
};

class horizon_instruction final : public matchbox::implement_acceptor<instruction, horizon_instruction> {
  public:
	explicit horizon_instruction(const instruction_id iid, const task_id horizon_tid) : acceptor_base(iid), m_horizon_tid(horizon_tid) {}

	task_id get_horizon_task_id() const { return m_horizon_tid; }

  private:
	task_id m_horizon_tid;
};

class epoch_instruction final : public matchbox::implement_acceptor<instruction, epoch_instruction> {
  public:
	explicit epoch_instruction(const instruction_id iid, const task_id epoch_tid, const epoch_action action)
	    : acceptor_base(iid), m_epoch_tid(epoch_tid), m_epoch_action(action) {}

	task_id get_epoch_task_id() const { return m_epoch_tid; }
	epoch_action get_epoch_action() const { return m_epoch_action; }

  private:
	task_id m_epoch_tid;
	epoch_action m_epoch_action;
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
