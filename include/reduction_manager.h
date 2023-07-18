#pragma once

#include "buffer_manager.h"
#include "runtime.h"
#include "types.h"

#include <vector>

namespace celerity {
namespace detail {

	class abstract_buffer_reduction {
	  public:
		explicit abstract_buffer_reduction(const buffer_id output_bid) : m_output_bid(output_bid) {}
		virtual ~abstract_buffer_reduction() = default;

		void push_overlapping_data(node_id source_nid, unique_payload_ptr data) { m_overlapping_data.emplace_back(source_nid, std::move(data)); }

		virtual unique_payload_ptr HACK_reduce_per_gpu_results(
		    const reduction_id rid, const node_id local_nid, unique_payload_ptr data, const size_t result_count) = 0;
		virtual void reduce_to_buffer() = 0;

	  protected:
		buffer_id m_output_bid;
		std::vector<std::pair<node_id, unique_payload_ptr>> m_overlapping_data;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class buffer_reduction final : public abstract_buffer_reduction {
	  public:
		buffer_reduction(buffer_id output_bid, BinaryOperation op, DataT identity) : abstract_buffer_reduction(output_bid), m_op(op), m_init(identity) {}

		virtual unique_payload_ptr HACK_reduce_per_gpu_results(
		    const reduction_id rid, const node_id local_nid, unique_payload_ptr data, const size_t result_count) override {
			DataT acc = m_init;
			for(size_t i = 0; i < result_count; ++i) {
				acc = m_op(acc, static_cast<const DataT*>(data.get_pointer())[i]);
			}
			auto result = make_uninitialized_payload<DataT>(1);
			*static_cast<DataT*>(result.get_pointer()) = acc;
			return result;
		}

		void reduce_to_buffer() override {
			std::sort(m_overlapping_data.begin(), m_overlapping_data.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });

			DataT acc = m_init;
			for(auto& [nid, data] : m_overlapping_data) {
				acc = m_op(acc, *static_cast<const DataT*>(data.get_pointer()));
			}

			const auto info = runtime::get_instance().get_buffer_manager().access_host_buffer<DataT, Dims>(
			    m_output_bid, access_mode::discard_write, detail::subrange_cast<Dims>(subrange<3>{{}, {1, 1, 1}}));
			*static_cast<DataT*>(info.ptr) = acc;
		}

	  private:
		BinaryOperation m_op;
		DataT m_init;
	};

	class reduction_manager {
	  public:
		template <typename DataT, int Dims, typename BinaryOperation>
		reduction_id create_reduction(const buffer_id bid, BinaryOperation op, DataT identity) {
			std::lock_guard lock{m_mutex};

			if(runtime::is_initialized()) {
				const auto info = runtime::get_instance().get_buffer_manager().get_buffer_info(bid);
				const auto num_devices = runtime::get_instance().get_local_devices().num_compute_devices();
				if(info.range[0] < num_devices) {
					CELERITY_CRITICAL(
					    "HACK: Multi-GPU reductions currently require the reduction buffer to have at least one element per local GPU (need: {}, have: {})",
					    num_devices, info.range[0]);
					abort();
				}
			}

			const auto rid = m_next_rid++;
			m_reductions.emplace(rid, std::make_unique<buffer_reduction<DataT, Dims, BinaryOperation>>(bid, op, identity));
			return rid;
		}

		bool has_reduction(reduction_id rid) const {
			std::lock_guard lock{m_mutex};
			return m_reductions.count(rid) != 0;
		}

		unique_payload_ptr HACK_reduce_per_gpu_results(const reduction_id rid, const node_id local_nid, unique_payload_ptr data, const size_t result_count) {
			std::lock_guard lock{m_mutex};
			return m_reductions.at(rid)->HACK_reduce_per_gpu_results(rid, local_nid, std::move(data), result_count);
		}

		void push_overlapping_reduction_data(reduction_id rid, node_id source_nid, unique_payload_ptr data) {
			std::lock_guard lock{m_mutex};
			m_reductions.at(rid)->push_overlapping_data(source_nid, std::move(data));
		}

		void finish_reduction(reduction_id rid) {
			std::lock_guard lock{m_mutex};
			m_reductions.at(rid)->reduce_to_buffer();
			m_reductions.erase(rid);
		}

	  private:
		mutable std::mutex m_mutex;
		reduction_id m_next_rid = 1;
		std::unordered_map<reduction_id, std::unique_ptr<abstract_buffer_reduction>> m_reductions;
	};

} // namespace detail
} // namespace celerity
