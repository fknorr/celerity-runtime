#pragma once

#include <cstddef>
#include <variant>

#include "intrusive_graph.h"
#include "mpi_support.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "utils.h"

#include <matchbox.hh>

namespace celerity {
namespace detail {

	enum class command_type { epoch, horizon, execution, push, await_push, reduction, fence };


	// ----------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------ COMMAND GRAPH -------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	// TODO: Consider adding a mechanism (during debug builds?) to assert that dependencies can only exist between commands on the same node
	class abstract_command : public intrusive_graph_node<abstract_command>,
	                         public matchbox::acceptor<class epoch_command, class horizon_command, class execution_command, class push_command,
	                             class await_push_command, class reduction_command, class fence_command> {
		friend class command_graph;

	  protected:
		abstract_command(command_id cid) : m_cid(cid) {}

	  public:
		virtual command_type get_type() const = 0;

		command_id get_cid() const { return m_cid; }

		// TODO remove together with graph serialization
		void mark_as_flushed() {
			assert(!m_flushed);
			m_flushed = true;
		}

		// TODO remove together with graph serialization
		bool is_flushed() const { return m_flushed; }

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id m_cid;
		bool m_flushed = false;
	};

	class push_command final : public matchbox::implement_acceptor<abstract_command, push_command> {
		friend class command_graph;
		push_command(command_id cid, buffer_id bid, reduction_id rid, node_id target, transfer_id trid, subrange<3> push_range)
		    : acceptor_base(cid), m_bid(bid), m_rid(rid), m_target(target), m_trid(trid), m_push_range(push_range) {}

		command_type get_type() const override { return command_type::push; }

	  public:
		buffer_id get_bid() const { return m_bid; }
		reduction_id get_reduction_id() const { return m_rid; }
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

	class await_push_command final : public matchbox::implement_acceptor<abstract_command, await_push_command> {
		friend class command_graph;
		await_push_command(command_id cid, buffer_id bid, reduction_id rid, transfer_id trid, region<3> region)
		    : acceptor_base(cid), m_bid(bid), m_rid(rid), m_trid(trid), m_region(std::move(region)) {}

		command_type get_type() const override { return command_type::await_push; }

	  public:
		buffer_id get_bid() const { return m_bid; }
		reduction_id get_reduction_id() const { return m_rid; }
		transfer_id get_transfer_id() const { return m_trid; }
		region<3> get_region() const { return m_region; }

	  private:
		buffer_id m_bid;
		// Having the reduction ID here isn't strictly required for matching against incoming pushes,
		// but it allows us to sanity check that they match as well as include the ID during graph printing.
		reduction_id m_rid;
		transfer_id m_trid;
		region<3> m_region;
	};

	class reduction_command final : public matchbox::implement_acceptor<abstract_command, reduction_command> {
		friend class command_graph;
		reduction_command(command_id cid, const reduction_info& info) : acceptor_base(cid), m_info(info) {}

		command_type get_type() const override { return command_type::reduction; }

	  public:
		const reduction_info& get_reduction_info() const { return m_info; }

	  private:
		reduction_info m_info;
	};

	class task_command : public abstract_command {
	  protected:
		task_command(command_id cid, task_id tid) : abstract_command(cid), m_tid(tid) {}

	  public:
		task_id get_tid() const { return m_tid; }

	  private:
		task_id m_tid;
	};

	class epoch_command final : public matchbox::implement_acceptor<task_command, epoch_command> {
		friend class command_graph;
		epoch_command(const command_id& cid, const task_id& tid, epoch_action action) : acceptor_base(cid, tid), m_action(action) {}

		command_type get_type() const override { return command_type::epoch; }

	  public:
		epoch_action get_epoch_action() const { return m_action; }

	  private:
		epoch_action m_action;
	};

	class horizon_command final : public matchbox::implement_acceptor<task_command, horizon_command> {
		friend class command_graph;
		using acceptor_base::acceptor_base;

		command_type get_type() const override { return command_type::horizon; }
	};

	class execution_command final : public matchbox::implement_acceptor<task_command, execution_command> {
		friend class command_graph;

	  protected:
		execution_command(command_id cid, task_id tid, subrange<3> execution_range) : acceptor_base(cid, tid), m_execution_range(execution_range) {}

	  public:
		command_type get_type() const override { return command_type::execution; }

		const subrange<3>& get_execution_range() const { return m_execution_range; }

		void set_is_reduction_initializer(bool is_initializer) { m_initialize_reductions = is_initializer; }

		bool is_reduction_initializer() const { return m_initialize_reductions; }

	  private:
		subrange<3> m_execution_range;
		bool m_initialize_reductions = false;
	};

	class fence_command final : public matchbox::implement_acceptor<task_command, fence_command> {
		friend class command_graph;
		using acceptor_base::acceptor_base;

		command_type get_type() const override { return command_type::fence; }
	};

	// ----------------------------------------------------------------------------------------------------------------
	// -------------------------------------------- SERIALIZED COMMANDS -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	// TODO: These are a holdover from the master/worker scheduling model. Remove at some point.
	// The only reason we keep them around for now is that they allow us to persist commands beyond graph pruning.
	// They no longer have to be network-serializable though.

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
		region<3> region;
	};

	struct reduction_data {
		reduction_id rid;
	};

	struct fence_data {
		task_id tid;
	};

	using command_data = std::variant<std::monostate, horizon_data, epoch_data, execution_data, push_data, await_push_data, reduction_data, fence_data>;

	struct command_pkg {
		command_id cid{};
		command_data data{};
		std::vector<command_id> dependencies;

		std::optional<task_id> get_tid() const {
			return matchbox::match<std::optional<task_id>>(
			    data,                                          //
			    [](const horizon_data& d) { return d.tid; },   //
			    [](const epoch_data& d) { return d.tid; },     //
			    [](const execution_data& d) { return d.tid; }, //
			    [](const fence_data& d) { return d.tid; },     //
			    [](const auto&) { return std::nullopt; }       //
			);
		}

		command_type get_command_type() const {
			return matchbox::match(
			    data,
			    [](const std::monostate&) -> command_type {
				    assert(!"calling get_command_type() on an empty command_pkg");
				    std::terminate();
			    },
			    [](const horizon_data&) { return command_type::horizon; },       //
			    [](const epoch_data&) { return command_type::epoch; },           //
			    [](const execution_data&) { return command_type::execution; },   //
			    [](const push_data&) { return command_type::push; },             //
			    [](const await_push_data&) { return command_type::await_push; }, //
			    [](const reduction_data&) { return command_type::reduction; },   //
			    [](const fence_data&) { return command_type::fence; });
		}
	};

} // namespace detail
} // namespace celerity
