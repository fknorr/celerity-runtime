#include "task.h"

#include <algorithm>

namespace celerity {
namespace detail {

	std::unordered_set<buffer_id> buffer_access_map::get_accessed_buffers() const {
		std::unordered_set<buffer_id> result;
		for(auto& [bid, _] : m_accesses) {
			result.emplace(bid);
		}
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> buffer_access_map::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(auto& [_, rm] : m_accesses) {
			result.insert(rm->get_access_mode());
		}
		return result;
	}

	template <int KernelDims>
	subrange<3> apply_range_mapper(range_mapper_base const* rm, const chunk<KernelDims>& chnk) {
		switch(rm->get_buffer_dimensions()) {
		case 1: return subrange_cast<3>(rm->map_1(chnk));
		case 2: return subrange_cast<3>(rm->map_2(chnk));
		case 3: return rm->map_3(chnk);
		default: assert(false);
		}
		return subrange<3>{};
	}

	GridRegion<3> buffer_access_map::get_mode_requirements(
	    const buffer_id bid, const access_mode mode, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		GridRegion<3> result;
		for(size_t i = 0; i < m_accesses.size(); ++i) {
			if(m_accesses[i].first != bid || m_accesses[i].second->get_access_mode() != mode) continue;
			result = GridRegion<3>::merge(result, get_requirements_for_nth_access(i, kernel_dims, sr, global_size));
		}
		return result;
	}

	GridBox<3> buffer_access_map::get_requirements_for_nth_access(
	    const size_t n, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		const auto& [_, rm] = m_accesses[n];

		chunk<3> chnk{sr.offset, sr.range, global_size};
		subrange<3> req;
		switch(kernel_dims) {
		case 0:
			[[fallthrough]]; // sycl::range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
			                 // anyway, we require range mappers to take at least 1d subranges
		case 1: req = apply_range_mapper<1>(rm.get(), chunk_cast<1>(chnk)); break;
		case 2: req = apply_range_mapper<2>(rm.get(), chunk_cast<2>(chnk)); break;
		case 3: req = apply_range_mapper<3>(rm.get(), chunk_cast<3>(chnk)); break;
		default: assert(!"Unreachable");
		}
		return subrange_to_grid_box(req);
	}

	void side_effect_map::add_side_effect(const host_object_id hoid, const experimental::side_effect_order order) {
		// TODO for multiple side effects on the same hoid, find the weakest order satisfying all of them
		emplace(hoid, order);
	}

	fence_guard::~fence_guard() { m_fence->release(); }

	void fence::notify_arrived() {
		{
			const std::lock_guard lock(m_mutex);
			assert(m_state.load(std::memory_order_relaxed) == created);
			m_state.store(arrived, std::memory_order_relaxed);
		}
		m_state_change.notify_all();
	}

	fence_guard fence::await_arrived_and_acquire() {
		{
			std::unique_lock lock(m_mutex);
			state state_before;
			while((state_before = m_state.load(std::memory_order_relaxed)) != arrived) {
				assert(state_before == created);
				m_state_change.wait(lock);
			}
			m_state.store(acquired, std::memory_order_relaxed);
		}
		m_state_change.notify_all();
		return fence_guard(this);
	}

	bool fence::poll_released() {
		const auto state = m_state.load(std::memory_order_relaxed);
		assert(state == arrived || state == acquired || state == released);
		return state == released;
	}

	void fence::release() {
		{
			const std::lock_guard lock(m_mutex);
			assert(m_state.load(std::memory_order_relaxed) == acquired);
			m_state.store(released, std::memory_order_relaxed);
		}
		m_state_change.notify_all();
	}

} // namespace detail
} // namespace celerity
