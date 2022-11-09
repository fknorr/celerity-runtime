#pragma once

#include <vector>

#include "ranges.h"
#include "sycl_wrappers.h"
#include "types.h"

#include "buffer_storage.h" // NOCOMMIT JUST HACKING

namespace celerity::detail {

// To avoid additional register pressure, we embed closure object IDs into unhydrated
// accessor pointers, with the assumption that a real pointer will never be in the
// range [0, max_embedded_clsoure_id]. Embedding / extracting are currently no-ops
// and the associated helper functions only exist for documentation purposes.
// This number puts an effective limit on the number of closure objects (accessors
// etc.) that can be captured into a command function.
constexpr size_t max_embedded_closure_object_id = 128;

template <typename T>
using can_embed_closure_object_id = std::bool_constant<sizeof(closure_object_id) == sizeof(T)>;

template <typename T>
T embed_closure_object_id(const closure_object_id coid) {
	static_assert(can_embed_closure_object_id<T>::value);
	assert(coid < max_embedded_closure_object_id);
	T result;
	std::memcpy(&result, &coid, sizeof(coid));
	return result;
}

template <typename T>
closure_object_id extract_closure_object_id(const T value) {
	static_assert(can_embed_closure_object_id<T>::value);
	closure_object_id result;
	std::memcpy(&result, &value, sizeof(value));
	return result;
}

template <typename T>
bool is_embedded_closure_object_id(const T value) {
	static_assert(can_embed_closure_object_id<T>::value);
	return extract_closure_object_id(value) < max_embedded_closure_object_id;
}

// Consider this API:
// There is a function called hydrate (rename current) that receives the closure as well as all access infos.
// It then does the copy and ensures that all infos have been consumed.
//
// TODO: Make this work for reductions as well!
// TODO: Just make everything static? No need for a singleton really?
//
// !!!!!!!!!!!!!
// FIXME: Capturing a host accessor by reference into host_task does not cause it to be copied => must capture by value! We can at least try to detect this
//       (not distinguishable from unused accessor though...)
// !!!!!!!!!!!!!
//
class closure_hydrator {
  public:
	struct NOCOMMIT_info {
		target tgt;
		void* ptr;
		range<3> buffer_range;
		id<3> buffer_offset;
		subrange<3> accessor_sr;
		// NOCOMMIT JUST HACKING
		async_event pending_transfers;
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
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		assert(std::all_of(m_consumed_infos.cbegin(), m_consumed_infos.cend(), [](bool v) { return v; }));
		m_consumed_infos = std::vector<bool>(infos.size(), false);
#endif
		m_infos = std::move(infos);
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
	NOCOMMIT_info hydrate(const closure_object_id coid) {
		assert(!m_infos.empty());
		assert(coid < m_infos.size());
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		assert(m_consumed_infos[coid] == false);
		m_consumed_infos[coid] = true;
#endif
		return m_infos[coid];
	}

  private:
	inline static thread_local std::unique_ptr<closure_hydrator> m_instance;
	std::vector<NOCOMMIT_info> m_infos;
	sycl::handler* m_sycl_cgh = nullptr;

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
	std::vector<bool> m_consumed_infos;
#endif

	closure_hydrator() = default;
	closure_hydrator(const closure_hydrator&) = delete;
	closure_hydrator(closure_hydrator&&) = delete;
};

}; // namespace celerity::detail