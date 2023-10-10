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

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id m_cid;
	};

	class push_command final : public matchbox::implement_acceptor<abstract_command, push_command> {
		friend class command_graph;
		push_command(const command_id cid, const node_id target, const receive_id& rcvid, const subrange<3>& push_range)
		    : acceptor_base(cid), m_target(target), m_rcvid(rcvid), m_push_range(push_range) {}

		command_type get_type() const override { return command_type::push; }

	  public:
		node_id get_target() const { return m_target; }
		const receive_id& get_receive_id() const { return m_rcvid; }
		const subrange<3>& get_range() const { return m_push_range; }

	  private:
		node_id m_target;
		receive_id m_rcvid;
		subrange<3> m_push_range;
	};

	class await_push_command final : public matchbox::implement_acceptor<abstract_command, await_push_command> {
		friend class command_graph;
		await_push_command(const command_id cid, const receive_id& rcvid, region<3> region) : acceptor_base(cid), m_rcvid(rcvid), m_region(std::move(region)) {}

		command_type get_type() const override { return command_type::await_push; }

	  public:
		const receive_id& get_receive_id() const { return m_rcvid; }
		const region<3>& get_region() const { return m_region; }

	  private:
		receive_id m_rcvid;
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

} // namespace detail
} // namespace celerity
