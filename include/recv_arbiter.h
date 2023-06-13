#pragma once

#include "instruction_queue.h"
#include "types.h"

#include <memory>

namespace celerity::detail {

class recv_instruction;
struct pilot_message;

class recv_arbiter {
  public:
	class delegate {
	  public:
		delegate() = default;
		delegate(const delegate&) = delete;
		delegate& operator=(const delegate&) = delete;
		virtual ~delegate() = default;

		// TODO from live discussion: We can get away without any staging by using MPI_Type_create_subarray. MPI_Irecvs could be dispatched ASAP depending
		// on when pilots and instructions arrive, with all recvs being attached to the same queue event. We should later benchmark whether theses strided recvs
		// work with RDMA or if manual staging can outperform (some) implementations (which would require more elaborate dynamic IDAG generation).
		virtual instruction_queue_event begin_recv(allocation_id dest, size_t offset_bytes, size_t size_bytes, node_id source, int tag) = 0;
	};

	explicit recv_arbiter(delegate* delegate);
    recv_arbiter(recv_arbiter&&) = default;
    recv_arbiter &operator=(recv_arbiter&&) = default;
    ~recv_arbiter();

	void prepare_recv(const recv_instruction& rinstr);
	[[nodiscard]] instruction_queue_event submit_recv(const recv_instruction& rinstr);
	void accept_pilot(const node_id source, const pilot_message& pilot);

  private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}
