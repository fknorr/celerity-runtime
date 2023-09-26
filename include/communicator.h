#pragma once

#include "instruction_graph.h" // for pilot_message - TODO does not feel right
#include "utils.h"

namespace celerity::detail {

class communicator {
  public:
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		virtual void inbound_pilot_received(const inbound_pilot& pilot) = 0;
	};

	class event {
	  public:
		virtual ~event() = default;

	  protected:
		event() = default;
		event(const event&) = default;
		event(event&&) = default;
		event& operator=(const event&) = default;
		event& operator=(event&&) = default;

	  public:
		virtual bool is_complete() const = 0;
	};

	struct stride {
		range<3> allocation;
		subrange<3> subrange;
		size_t element_size = 1;

		friend bool operator==(const stride& lhs, const stride& rhs) {
			return lhs.allocation == rhs.allocation && lhs.subrange == rhs.subrange && lhs.element_size == rhs.element_size;
		}
		friend bool operator!=(const stride& lhs, const stride& rhs) { return !(lhs == rhs); }
	};

	virtual ~communicator() = default;

	virtual size_t get_num_nodes() const = 0;
	virtual node_id get_local_node_id() const = 0;
	virtual void send_outbound_pilot(const outbound_pilot& pilot) = 0;
	[[nodiscard]] virtual std::unique_ptr<event> send_payload(node_id to, int outbound_pilot_tag, const void* base, const stride& stride) = 0;
	[[nodiscard]] virtual std::unique_ptr<event> receive_payload(node_id from, int inbound_pilot_tag, void* base, const stride& stride) = 0;

  protected:
	communicator() = default;
	communicator(const communicator&) = default;
	communicator(communicator&&) = default;
	communicator& operator=(const communicator&) = default;
	communicator& operator=(communicator&&) = default;
};

class communicator_factory {
  public:
	virtual ~communicator_factory() = default;

  protected:
	communicator_factory() = default;
	communicator_factory(const communicator_factory&) = delete;
	communicator_factory(communicator_factory&&) = delete;
	communicator_factory& operator=(const communicator_factory&) = delete;
	communicator_factory& operator=(communicator_factory&&) = delete;

  public:
	virtual std::unique_ptr<communicator> make_communicator(communicator::delegate* delegate) const = 0;
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::communicator::stride> {
	size_t operator()(const celerity::detail::communicator::stride& stride) const {
		size_t h = 0;
		for(int d = 0; d < 3; ++d) {
			celerity::detail::utils::hash_combine(h, stride.allocation[d]);
			celerity::detail::utils::hash_combine(h, stride.subrange.offset[d]);
			celerity::detail::utils::hash_combine(h, stride.subrange.range[d]);
		}
		celerity::detail::utils::hash_combine(h, stride.element_size);
		return h;
	}
};
