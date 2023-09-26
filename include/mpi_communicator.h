#pragma once

#include "communicator.h"
#include "instruction_graph.h"

#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <mpi.h>

namespace celerity::detail {

class mpi_communicator final : public communicator {
  public:
	class event final : public communicator::event {
	  public:
		event(MPI_Request req);
		event(const event&) = delete;
		event(event&&) = delete;
		event& operator=(const event&) = delete;
		event& operator=(event&&) = delete;
		~event() override;

		bool is_complete() const override;

	  private:
		mutable MPI_Request m_req;
	};

	mpi_communicator(MPI_Comm comm, delegate* delegate);
	mpi_communicator(const mpi_communicator&) = delete;
	mpi_communicator(mpi_communicator&&) = delete;
	mpi_communicator& operator=(const mpi_communicator&) = delete;
	mpi_communicator& operator=(mpi_communicator&&) = delete;
	~mpi_communicator() override;

	size_t get_num_nodes() const override;
	node_id get_local_node_id() const override;
	void send_pilot_message(node_id to, const pilot_message& pilot) override;
	std::unique_ptr<communicator::event> send_payload(node_id to, int tag, const void* base, const stride& stride) override;
	std::unique_ptr<communicator::event> receive_payload(node_id from, int tag, void* base, const stride& stride) override;

  private:
	inline constexpr static int pilot_tag = 0; // TODO have a celerity pilot_id and translate it to an MPI tag on this level

	struct datatype_deleter {
		void operator()(MPI_Datatype dtype) const;
	};
	using unique_datatype = std::unique_ptr<std::remove_pointer_t<MPI_Datatype>, datatype_deleter>;

	// immutable
	MPI_Comm m_comm;
	delegate* m_delegate;

	// accesed only by owning thread
	std::vector<std::pair<std::unique_ptr<pilot_message>, MPI_Request>> m_outbound_pilots;
	mutable std::unordered_map<size_t, unique_datatype> m_scalar_type_cache;
	mutable std::unordered_map<stride, unique_datatype> m_array_type_cache;

	// accessed by both threads
	std::atomic<bool> m_shutdown{false};

	std::thread m_listener_thread;

	void listen();
	MPI_Datatype get_scalar_type(size_t bytes) const;
	MPI_Datatype get_array_type(const stride& stride) const;
};

class mpi_communicator_factory final : public communicator_factory {
  public:
	explicit mpi_communicator_factory(const MPI_Comm comm) : m_comm(comm) {}

	std::unique_ptr<communicator> make_communicator(communicator::delegate* delegate) const override;

  private:
	MPI_Comm m_comm;
};

} // namespace celerity::detail
