#include "mpi_communicator.h"
#include "instruction_graph.h"
#include "ranges.h"
#include <atomic>
#include <cstddef>
#include <mpi.h>

namespace celerity::detail {

mpi_communicator::event::event(const MPI_Request req) : m_req(req) {}

mpi_communicator::event::~event() {
	// MPI_Request_free is always incorrect for our use case: events originate from an Isend or Irecv, which must ensure that the user-provided buffer remains
	// until the operation has completed.
	MPI_Wait(&m_req, MPI_STATUS_IGNORE);
}

bool mpi_communicator::event::is_complete() const {
	int flag = -1;
	MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE);
	return flag != 0;
}

mpi_communicator::mpi_communicator(const MPI_Comm comm, delegate* const delegate)
    : m_comm(comm), m_delegate(delegate), m_listener_thread(&mpi_communicator::listen, this) {}

mpi_communicator::~mpi_communicator() {
	m_shutdown.store(true, std::memory_order_relaxed);
	m_listener_thread.join();
	for(auto& [_, req] : m_outbound_pilots) {
		MPI_Wait(&req, MPI_STATUS_IGNORE);
	}
}

size_t mpi_communicator::get_num_nodes() const {
	int size = -1;
	MPI_Comm_size(m_comm, &size);
	return static_cast<size_t>(size);
}

node_id mpi_communicator::get_local_node_id() const {
	int rank = -1;
	MPI_Comm_rank(m_comm, &rank);
	return static_cast<node_id>(rank);
}

void mpi_communicator::send_pilot_message(const node_id to, const pilot_message& pilot) {
	// initiate Isend as early as possible
	auto stable_pilot = std::make_unique<pilot_message>(pilot);
	MPI_Request req = MPI_REQUEST_NULL;
	MPI_Isend(stable_pilot.get(), sizeof *stable_pilot.get(), MPI_BYTE, static_cast<int>(to), pilot_tag, m_comm, &req);
	for(auto& [_, req] : m_outbound_pilots) {
		int flag = -1;
		MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
	}

	// collect finished sends (TODO rate-limit this to avoid quadratic behavior)
	const auto last_incomplete_outbound_pilot =
	    std::remove_if(m_outbound_pilots.begin(), m_outbound_pilots.end(), [](const auto& pair) { return pair.second == MPI_REQUEST_NULL; });
	m_outbound_pilots.erase(last_incomplete_outbound_pilot, m_outbound_pilots.end());

	// keep allocation until Isend has completed
	m_outbound_pilots.emplace_back(std::move(stable_pilot), req);
}

std::unique_ptr<communicator::event> mpi_communicator::send_payload(const node_id to, const int tag, const void* const base, const stride& stride) {
	MPI_Request req;
	// TODO normalize stride and adjust base in order to re-use more datatypes
	MPI_Isend(base, 1, get_array_type(stride), static_cast<int>(to), tag, m_comm, &req);
	return std::make_unique<event>(req);
}

std::unique_ptr<communicator::event> mpi_communicator::receive_payload(const node_id from, const int tag, void* const base, const stride& stride) {
	MPI_Request req;
	// TODO normalize stride and adjust base in order to re-use more datatypes
	MPI_Irecv(base, 1, get_array_type(stride), static_cast<int>(from), tag, m_comm, &req);
	return std::make_unique<event>(req);
}

MPI_Datatype mpi_communicator::get_scalar_type(const size_t bytes) const {
	if(const auto it = m_scalar_type_cache.find(bytes); it != m_scalar_type_cache.end()) { return it->second.get(); }

	MPI_Datatype type = MPI_DATATYPE_NULL;
	MPI_Type_contiguous(bytes, MPI_BYTE, &type);
	MPI_Type_commit(&type);
	m_scalar_type_cache.emplace(bytes, unique_datatype(type));
	return type;
}

MPI_Datatype mpi_communicator::get_array_type(const stride& stride) const {
	if(const auto it = m_array_type_cache.find(stride); it != m_array_type_cache.end()) { return it->second.get(); }

	const int dims = stride.allocation.get_min_dimensions();
	assert(stride.subrange.get_min_dimensions() <= dims);

	int size_array[3];
	int subsize_array[3];
	int start_array[3];
	for(int d = 0; d < 3; ++d) {
		// TODO support transfers > 2Gi elements, at least in the 1d case - either through typing magic here, or by splitting sends / recvs in the iggen
		assert(stride.allocation[d] <= INT_MAX);
		size_array[d] = static_cast<int>(stride.allocation[d]);
		assert(stride.subrange.range[d] <= INT_MAX);
		subsize_array[d] = static_cast<int>(stride.subrange.range[d]);
		assert(stride.subrange.offset[d] <= INT_MAX);
		start_array[d] = static_cast<int>(stride.subrange.offset[d]);
	}

	MPI_Datatype type = MPI_DATATYPE_NULL;
	MPI_Type_create_subarray(dims, size_array, subsize_array, start_array, MPI_ORDER_C, get_scalar_type(stride.element_size), &type);
	MPI_Type_commit(&type);
	m_array_type_cache.emplace(stride, unique_datatype(type));
	return type;
}

void mpi_communicator::datatype_deleter::operator()(MPI_Datatype dtype) const { MPI_Type_free(&dtype); }

void mpi_communicator::listen() {
	pilot_message in_pilot;
	MPI_Request req = MPI_REQUEST_NULL;
	MPI_Irecv(&in_pilot, sizeof in_pilot, MPI_BYTE, MPI_ANY_SOURCE, pilot_tag, m_comm, &req);
	while(!m_shutdown.load(std::memory_order_relaxed)) {
		int flag = -1;
		MPI_Status status;
		MPI_Test(&req, &flag, &status);
		if(flag != 0) {
			const auto from = static_cast<node_id>(status.MPI_SOURCE);
			const auto pilot = in_pilot;
			// immediately re-start MPI_Irecv to overlap with call to delegate
			MPI_Irecv(&in_pilot, sizeof in_pilot, MPI_BYTE, MPI_ANY_SOURCE, pilot_tag, m_comm, &req);
			if(m_delegate != nullptr) { m_delegate->pilot_message_received(from, pilot); }
		}
	}
	MPI_Request_free(&req);
}

std::unique_ptr<communicator> mpi_communicator_factory::make_communicator(communicator::delegate* delegate) const {
	return std::make_unique<mpi_communicator>(m_comm, delegate);
}

} // namespace celerity::detail
