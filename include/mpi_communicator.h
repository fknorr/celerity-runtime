#pragma once

#include "communicator.h"
#include "instruction_graph.h" // for pilot_message TODO

#include <memory>
#include <unordered_map>
#include <vector>

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

	explicit mpi_communicator(MPI_Comm comm);

	mpi_communicator(mpi_communicator&&) = default;
	mpi_communicator& operator=(mpi_communicator&&) = default;
	mpi_communicator(const mpi_communicator&) = delete;
	mpi_communicator& operator=(const mpi_communicator&) = delete;
	~mpi_communicator() override;

	size_t get_num_nodes() const override;
	node_id get_local_node_id() const override;
	void send_outbound_pilot(const outbound_pilot& pilot) override;
	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override;
	[[nodiscard]] std::unique_ptr<communicator::event> send_payload(node_id to, int outbound_pilot_tag, const void* base, const stride& stride) override;
	[[nodiscard]] std::unique_ptr<communicator::event> receive_payload(node_id from, int inbound_pilot_tag, void* base, const stride& stride) override;

  private:
	inline constexpr static int pilot_tag = 0; // TODO have a celerity pilot_id and translate it to an MPI tag on this level

	struct datatype_deleter {
		void operator()(MPI_Datatype dtype) const;
	};
	using unique_datatype = std::unique_ptr<std::remove_pointer_t<MPI_Datatype>, datatype_deleter>;

	struct in_flight_pilot {
		// std::unique_ptr: pointer must be stable
		std::unique_ptr<pilot_message> message = std::make_unique<pilot_message>();
		MPI_Request request = MPI_REQUEST_NULL;
	};

	MPI_Comm m_comm;

	in_flight_pilot m_inbound_pilot; // TODO do we want to have multiple of these buffers around to increase throughput?
	std::vector<in_flight_pilot> m_outbound_pilots;

	std::unordered_map<size_t, unique_datatype> m_scalar_type_cache;
	std::unordered_map<stride, unique_datatype> m_array_type_cache;

	void begin_receive_pilot();
	MPI_Datatype get_scalar_type(size_t bytes);
	MPI_Datatype get_array_type(const stride& stride);
};

} // namespace celerity::detail
