#pragma once

#include <cstddef>
#include <variant>

#include "intrusive_graph.h"
#include "mpi_support.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "utils.h"

namespace celerity {
namespace detail {

	enum class command_type { epoch, horizon, execution, data_request, push, await_push, reduction, gather, allgather, broadcast, scatter, alltoall };

	// ----------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------ COMMAND GRAPH -------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	// TODO: Consider using LLVM-style RTTI for better performance
	template <typename T, typename P>
	bool isa(P* p) {
		return dynamic_cast<T*>(const_cast<std::remove_const_t<P>*>(p)) != nullptr;
	}

	// TODO: Consider adding a mechanism (during debug builds?) to assert that dependencies can only exist between commands on the same node
	class abstract_command : public intrusive_graph_node<abstract_command> {
		friend class command_graph;

	  protected:
		abstract_command(command_id cid, node_id nid) : m_cid(cid), m_nid(nid) {}

	  public:
		virtual ~abstract_command() = 0;

		command_id get_cid() const { return m_cid; }

		node_id get_nid() const { return m_nid; }

		void mark_as_flushed() {
			assert(!m_flushed);
			m_flushed = true;
		}
		bool is_flushed() const { return m_flushed; }

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id m_cid;
		node_id m_nid;
		bool m_flushed = false;
	};
	inline abstract_command::~abstract_command() {}

	class push_command final : public abstract_command {
		friend class command_graph;
		push_command(command_id cid, node_id nid, buffer_id bid, reduction_id rid, node_id target, transfer_id trid, subrange<3> push_range)
		    : abstract_command(cid, nid), m_bid(bid), m_rid(rid), m_target(target), m_trid(trid), m_push_range(push_range) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		reduction_id get_rid() const { return m_rid; }
		node_id get_target() const { return m_target; }
		transfer_id get_transfer_id() const { return m_trid; }
		const subrange<3>& get_range() const { return m_push_range; }

	  private:
		buffer_id m_bid;
		reduction_id m_rid;
		node_id m_target;
		transfer_id m_trid;
		subrange<3> m_push_range;
	};

	class await_push_command final : public abstract_command {
		friend class command_graph;
		await_push_command(command_id cid, node_id nid, buffer_id bid, transfer_id trid, GridRegion<3> region)
		    : abstract_command(cid, nid), m_bid(bid), m_trid(trid), m_region(region) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		transfer_id get_transfer_id() const { return m_trid; }
		GridRegion<3> get_region() const { return m_region; }

	  private:
		buffer_id m_bid;
		transfer_id m_trid;
		GridRegion<3> m_region;
	};

	class data_request_command final : public abstract_command {
		friend class command_graph;
		data_request_command(command_id cid, node_id nid, buffer_id bid, node_id source, subrange<3> data_range)
		    : abstract_command(cid, nid), m_bid(bid), m_source(source), m_data_range(data_range) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		node_id get_source() const { return m_source; }
		const subrange<3>& get_range() const { return m_data_range; }

	  private:
		buffer_id m_bid;
		node_id m_source;
		subrange<3> m_data_range;
	};

	class reduction_command final : public abstract_command {
		friend class command_graph;
		reduction_command(command_id cid, node_id nid, const reduction_info& info) : abstract_command(cid, nid), m_info(info) {}

	  public:
		const reduction_info& get_reduction_info() const { return m_info; }

	  private:
		reduction_info m_info;
	};

	class task_command : public abstract_command {
	  protected:
		task_command(command_id cid, node_id nid, task_id tid) : abstract_command(cid, nid), m_tid(tid) {}

	  public:
		task_id get_tid() const { return m_tid; }

	  private:
		task_id m_tid;
	};

	class epoch_command final : public task_command {
		friend class command_graph;
		epoch_command(const command_id& cid, const node_id& nid, const task_id& tid, epoch_action action) : task_command(cid, nid, tid), m_action(action) {}

	  public:
		epoch_action get_epoch_action() const { return m_action; }

	  private:
		epoch_action m_action;
	};

	class horizon_command final : public task_command {
		friend class command_graph;
		using task_command::task_command;
	};

	class execution_command final : public task_command {
		friend class command_graph;

	  protected:
		execution_command(command_id cid, node_id nid, task_id tid, subrange<3> execution_range)
		    : task_command(cid, nid, tid), m_execution_range(execution_range) {}

	  public:
		const subrange<3>& get_execution_range() const { return m_execution_range; }

		void set_is_reduction_initializer(bool is_initializer) { m_initialize_reductions = is_initializer; }

		bool is_reduction_initializer() const { return m_initialize_reductions; }

		// Sets the device this command should run on. The idea behind this is that assigning the same chunk (in split order)
		// to the same device will likely reduce the amount of required inter-device transfers for typical kernels.
		// NOTE: This is a temporary solution until we have better local scheduling.
		void set_device_id(device_id id) { m_device_id = id; }
		device_id get_device_id() const { return m_device_id; }

	  private:
		subrange<3> m_execution_range;
		bool m_initialize_reductions = false;
		device_id m_device_id = 0;
	};

	class gather_command final : public abstract_command {
		friend class command_graph;
		gather_command(const command_id cid, const node_id nid, const buffer_id bid, std::vector<GridRegion<3>> source_regions, GridRegion<3> dest_region,
		    const node_id root)
		    : abstract_command(cid, nid), m_bid(bid), m_source_regions(std::move(source_regions)), m_dest_region(std::move(dest_region)), m_root(root) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		const std::vector<GridRegion<3>>& get_source_regions() const { return m_source_regions; }
		const GridRegion<3>& get_dest_region() const { return m_dest_region; }
		node_id get_root() const { return m_root; }

	  private:
		buffer_id m_bid;
		std::vector<GridRegion<3>> m_source_regions;
		GridRegion<3> m_dest_region;
		node_id m_root;
	};

	class allgather_command final : public abstract_command {
		friend class command_graph;
		allgather_command(const command_id cid, const node_id nid, const buffer_id bid, std::vector<GridRegion<3>> source_regions, GridRegion<3> dest_region)
		    : abstract_command(cid, nid), m_bid(bid), m_source_regions(std::move(source_regions)), m_dest_region(std::move(dest_region)) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		const std::vector<GridRegion<3>>& get_source_regions() const { return m_source_regions; }
		const GridRegion<3>& get_dest_region() const { return m_dest_region; }

	  private:
		buffer_id m_bid;
		std::vector<GridRegion<3>> m_source_regions;
		GridRegion<3> m_dest_region;
	};

	class broadcast_command final : public abstract_command {
		friend class command_graph;
		broadcast_command(const command_id cid, const node_id nid, const buffer_id bid, const node_id root, GridRegion<3> region)
		    : abstract_command(cid, nid), m_bid(bid), m_root(root), m_region(std::move(region)) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		const GridRegion<3>& get_region() const { return m_region; }
		node_id get_root() const { return m_root; }

	  private:
		buffer_id m_bid;
		node_id m_root;
		GridRegion<3> m_region;
	};

	class scatter_command final : public abstract_command {
		friend class command_graph;
		scatter_command(const command_id cid, const node_id nid, const buffer_id bid, const node_id root, GridRegion<3> source_region,
		    std::vector<GridRegion<3>> dest_regions)
		    : abstract_command(cid, nid), m_bid(bid), m_root(root), m_source_region(std::move(source_region)), m_dest_regions(std::move(dest_regions)) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		node_id get_root() const { return m_root; }
		const GridRegion<3>& get_source_region() const { return m_source_region; }
		const std::vector<GridRegion<3>>& get_dest_regions() const { return m_dest_regions; }

	  private:
		buffer_id m_bid;
		node_id m_root;
		GridRegion<3> m_source_region;
		std::vector<GridRegion<3>> m_dest_regions;
	};

	class alltoall_command final : public abstract_command {
		friend class command_graph;
		alltoall_command(
		    const command_id cid, const node_id nid, const buffer_id bid, std::vector<GridRegion<3>> send_regions, std::vector<GridRegion<3>> recv_regions)
		    : abstract_command(cid, nid), m_bid(bid), m_send_regions(std::move(send_regions)), m_recv_regions(std::move(recv_regions)) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		const std::vector<GridRegion<3>>& get_send_regions() const { return m_send_regions; }
		const std::vector<GridRegion<3>>& get_recv_regions() const { return m_recv_regions; }

	  private:
		buffer_id m_bid;
		std::vector<GridRegion<3>> m_send_regions;
		std::vector<GridRegion<3>> m_recv_regions;
	};

	// ----------------------------------------------------------------------------------------------------------------
	// -------------------------------------------- SERIALIZED COMMANDS -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	struct horizon_data {
		task_id tid;
	};

	struct epoch_data {
		task_id tid;
		epoch_action action;
	};

	struct execution_data {
		task_id tid;
		subrange<3> sr;
		bool initialize_reductions;
		device_id did;
	};

	struct push_data {
		buffer_id bid;
		reduction_id rid;
		node_id target;
		transfer_id trid;
		subrange<3> sr;
	};

	struct await_push_data {
		buffer_id bid;
		reduction_id rid;
		transfer_id trid;

		// NOCOMMIT HACK: This is just so we don't have to rip the whole serialization mechanism out right now (GridRegion contains variable length data)
		static constexpr size_t max_subranges = 32;
		size_t num_subranges;
		subrange<3> region[max_subranges];
	};

	struct data_request_data { // ...
		buffer_id bid;
		node_id source;
		subrange<3> sr;
	};

	struct reduction_data {
		reduction_id rid;
	};

	struct gather_data {
		buffer_id bid;
		std::vector<GridRegion<3>> source_regions;
		GridRegion<3> dest_region;
		node_id root;
	};

	struct allgather_data {
		buffer_id bid;
		std::vector<GridRegion<3>> source_regions;
		GridRegion<3> dest_region;
	};

	struct broadcast_data {
		buffer_id bid;
		GridRegion<3> region;
		node_id root;
	};

	struct scatter_data {
		buffer_id bid;
		node_id root;
		GridRegion<3> source_region;
		std::vector<GridRegion<3>> dest_regions;
	};

	struct alltoall_data {
		buffer_id bid;
		std::vector<GridRegion<3>> send_regions;
		std::vector<GridRegion<3>> recv_regions;
	};

	using command_data = std::variant<std::monostate, horizon_data, epoch_data, execution_data, push_data, await_push_data, data_request_data, reduction_data,
	    gather_data, allgather_data, broadcast_data, scatter_data, alltoall_data>;

	/**
	 * A command package is what is actually transferred between nodes.
	 */
	struct command_pkg {
		command_id cid{};
		command_data data;

		// FIXME: Just a quick hack to get command serialization to work w/o command_frame
		std::vector<command_id> dependencies;

		std::optional<task_id> get_tid() const {
			// clang-format off
			return utils::match(data,
				[](const horizon_data& d) { return std::optional{d.tid}; },
				[](const epoch_data& d) { return std::optional{d.tid}; },
				[](const execution_data& d) { return std::optional{d.tid}; },
				[](const auto&) { return std::optional<task_id>{}; }
			);
			// clang-format on
		}

		command_type get_command_type() const {
			// clang-format off
			return utils::match(data,
			    [](const std::monostate&) -> command_type {
				    assert(!"calling get_command_type() on an empty command_pkg");
				    std::terminate();
			    },
			    [](const horizon_data&) { return command_type::horizon; },
			    [](const epoch_data&) { return command_type::epoch; },
			    [](const execution_data&) { return command_type::execution; },
			    [](const push_data&) { return command_type::push; },
			    [](const await_push_data&) { return command_type::await_push; },
				[](const data_request_data&) { return command_type::data_request; },
			    [](const reduction_data&) { return command_type::reduction; },
				[](const gather_data&) { return command_type::gather; },
				[](const allgather_data&) { return command_type::allgather; },
				[](const broadcast_data&) { return command_type::broadcast; },
				[](const scatter_data&) { return command_type::scatter; },
				[](const alltoall_data&) { return command_type::alltoall; }
			);
			// clang-format on
		}
	};

	// NOCOMMIT: No longer needed
	struct command_frame {
		using payload_type = command_id;

		command_pkg pkg;
		size_t num_dependencies = 0;
		payload_type dependencies[];

		// variable-sized structure
		command_frame() = default;
		command_frame(const command_frame&) = delete;
		command_frame& operator=(const command_frame&) = delete;

		iterable_range<const command_id*> iter_dependencies() const { return {dependencies, dependencies + num_dependencies}; }
	};

	// unique_frame_ptr assumes that the flexible payload member begins at exactly sizeof(Frame) bytes
	static_assert(offsetof(command_frame, dependencies) == sizeof(command_frame));

} // namespace detail
} // namespace celerity
