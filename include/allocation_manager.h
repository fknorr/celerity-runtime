#pragma once

#include "types.h"

#include <mutex>
#include <unordered_map>

namespace celerity::detail {

class allocation_manager {
  public:
	void begin_tracking(const allocation_id aid, void* const ptr) {
		std::lock_guard lock(m_mutex);
		const auto [_, inserted] = m_pointers.emplace(aid, ptr);
		assert(inserted);
	}

	void end_tracking(const allocation_id aid) {
		std::lock_guard lock(m_mutex);
		const auto erased = m_pointers.erase(aid);
		assert(erased == 1);
	}

	void* get_pointer(const allocation_id aid) const {
		std::lock_guard lock(m_mutex);
		return m_pointers.at(aid);
	}

  private:
	mutable std::mutex m_mutex;
	std::unordered_map<allocation_id, void*> m_pointers;
};

} // namespace celerity::detail
