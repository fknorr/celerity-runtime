#pragma once

#include "ranges.h"
#include "sycl_wrappers.h"
#include "types.h"

namespace celerity::detail {

// Consider this API:
// There is a function called hydrate (rename current) that receives the closure as well as all access infos.
// It then does the copy and ensures that all infos have been consumed.
//
// TODO: Make this work for reductions as well!
// TODO: Just make everything static? No need for a singleton really?
class closure_hydrator {
  public:
	struct NOCOMMIT_info {
		target tgt;
		void* ptr;
		range<3> buffer_range;
		id<3> buffer_offset;
		subrange<3> accessor_sr;
	};

	static void enable() {
		assert(m_instance == nullptr);
		m_instance = std::unique_ptr<closure_hydrator>(new closure_hydrator());
	}

	static bool is_available() { return m_instance != nullptr; }

	static closure_hydrator& get_instance() {
		assert(m_instance != nullptr);
		return *m_instance;
	}

	void prepare(std::vector<NOCOMMIT_info> infos) {
		assert(m_next_idx == m_infos.size() && "Unconsumed pointers left");
		m_infos = std::move(infos);
		m_next_idx = 0;
	}

	template <typename Closure>
	Closure hydrate_local_accessors(const Closure& closure, sycl::handler& cgh) {
		m_sycl_cgh = &cgh;
		Closure hydrated{closure};
		m_sycl_cgh = nullptr;
		return hydrated;
	}

	sycl::handler& get_sycl_handler() {
		assert(m_sycl_cgh != nullptr);
		return *m_sycl_cgh;
	}

	bool has_sycl_handler() const { return m_sycl_cgh != nullptr; }

	bool can_hydrate() const { return !m_infos.empty(); }

	// NOCOMMIT Change return type
	NOCOMMIT_info hydrate() {
		assert(!m_infos.empty());
		assert(m_next_idx < m_infos.size());
		return m_infos[m_next_idx++];
	}

  private:
	inline static thread_local std::unique_ptr<closure_hydrator> m_instance;
	size_t m_next_idx = 0;
	std::vector<NOCOMMIT_info> m_infos;
	sycl::handler* m_sycl_cgh = nullptr;

	closure_hydrator() = default;
	closure_hydrator(const closure_hydrator&) = delete;
	closure_hydrator(closure_hydrator&&) = delete;
};

}; // namespace celerity::detail