#include "cuda_instruction_queue.h"

#include "allocation_manager.h"
#include "buffer_storage.h" // TODO included for CELERITY_CUDA_CHECK, consider moving that
#include "instruction_graph.h"

#include <cuda_runtime.h>

namespace celerity::detail {

class cuda_event final : public instruction_queue_event_impl {
  public:
	cuda_event() {
		cudaEvent_t event;
		CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
		m_event = std::unique_ptr<CUevent_st, deleter>(event);
	}

	bool has_completed() const override {
		switch(const auto result = cudaEventQuery(m_event.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

	void block_on() override { CELERITY_CUDA_CHECK(cudaEventSynchronize, m_event.get()); }
	cudaEvent_t get() const { return m_event.get(); }

  private:
	struct deleter {
		void operator()(cudaEvent_t event) { CELERITY_CUDA_CHECK(cudaEventDestroy, event); }
	};
	std::unique_ptr<CUevent_st, deleter> m_event;
};

class cuda_stream final : public in_order_instruction_queue {
  public:
	explicit cuda_stream(const int cuda_device_id, allocation_manager& am) : m_allocation_mgr(&am) {
		CELERITY_CUDA_CHECK(cudaSetDevice, cuda_device_id);
		cudaStream_t stream;
		CELERITY_CUDA_CHECK(cudaStreamCreate, &stream);
		m_stream = std::unique_ptr<CUstream_st, deleter>(stream);
	}

	instruction_queue_event submit(std::unique_ptr<instruction> instr) override {
		utils::match(
		    *instr, //
		    [this](const alloc_instruction& ainstr) { submit_malloc(ainstr); }, [this](const free_instruction& finstr) { submit_free(finstr); },
		    [this](const copy_instruction& cinstr) { submit_copy(cinstr); },
		    [&](const auto& /* other */) { panic("Invalid instruction type on cuda_stream"); });
		auto evt = std::make_shared<cuda_event>();
		CELERITY_CUDA_CHECK(cudaEventRecord, evt->get(), m_stream.get());
		return evt;
	}

	void wait_on(const instruction_queue_event& evt) override {
		CELERITY_CUDA_CHECK(cudaStreamWaitEvent, m_stream.get(), dynamic_cast<const cuda_event&>(*evt).get(), 0);
	}

  private:
	struct deleter {
		void operator()(cudaStream_t stream) { CELERITY_CUDA_CHECK(cudaStreamDestroy, stream); }
	};
	std::unique_ptr<CUstream_st, deleter> m_stream;
	allocation_manager* m_allocation_mgr;

	void submit_malloc(const alloc_instruction& ainstr) const {
		void* ptr;
		CELERITY_CUDA_CHECK(cudaMallocAsync, &ptr, ainstr.get_size(), m_stream.get());
		assert(reinterpret_cast<uintptr_t>(ptr) % ainstr.get_alignment() == 0); // TODO handle large alignments
		m_allocation_mgr->begin_tracking(ainstr.get_allocation_id(), ptr);
	}

	void submit_free(const free_instruction& finstr) const {
		const auto ptr = m_allocation_mgr->get_pointer(finstr.get_allocation_id());
		CELERITY_CUDA_CHECK(cudaFreeAsync, ptr, m_stream.get());
		m_allocation_mgr->end_tracking(finstr.get_allocation_id());
	}

	void submit_copy(const copy_instruction& cinstr) const {
		const auto source_base_ptr = m_allocation_mgr->get_pointer(cinstr.get_source_allocation());
		const auto dest_base_ptr = m_allocation_mgr->get_pointer(cinstr.get_dest_allocation());

		switch(cinstr.get_dimensions()) {
		case 0: {
			CELERITY_CUDA_CHECK(cudaMemcpyAsync, dest_base_ptr, source_base_ptr, cinstr.get_element_size(), cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 1: {
			const auto source_base_offset = get_linear_index(range_cast<1>(cinstr.get_source_range()), id_cast<1>(cinstr.get_offset_in_source()));
			const auto dest_base_offset = get_linear_index(range_cast<1>(cinstr.get_dest_range()), id_cast<1>(cinstr.get_offset_in_dest()));
			CELERITY_CUDA_CHECK(cudaMemcpyAsync, static_cast<char*>(dest_base_ptr) + cinstr.get_element_size() * dest_base_offset,
			    static_cast<const char*>(source_base_ptr) + cinstr.get_element_size() * source_base_offset,
			    cinstr.get_copy_range()[0] * cinstr.get_element_size(), cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 2: {
			const auto source_base_offset = get_linear_index(range_cast<2>(cinstr.get_source_range()), id_cast<2>(cinstr.get_offset_in_source()));
			const auto dest_base_offset = get_linear_index(range_cast<2>(cinstr.get_dest_range()), id_cast<2>(cinstr.get_offset_in_dest()));
			CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, static_cast<char*>(dest_base_ptr) + cinstr.get_element_size() * dest_base_offset,
			    cinstr.get_dest_range()[1] * cinstr.get_element_size(),
			    static_cast<const char*>(source_base_ptr) + cinstr.get_element_size() * source_base_offset,
			    cinstr.get_source_range()[1] * cinstr.get_element_size(), cinstr.get_copy_range()[1] * cinstr.get_element_size(), cinstr.get_copy_range()[0],
			    cudaMemcpyDefault, m_stream.get());
			break;
		}
		case 3: {
			cudaMemcpy3DParms parms = {};
			parms.srcPos =
			    make_cudaPos(cinstr.get_offset_in_source()[2] * cinstr.get_element_size(), cinstr.get_offset_in_source()[1], cinstr.get_offset_in_source()[0]);
			parms.srcPtr =
			    make_cudaPitchedPtr(const_cast<void*>(source_base_ptr), cinstr.get_source_range()[2] * cinstr.get_element_size(), cinstr.get_source_range()[2],
			        cinstr.get_source_range()[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
			parms.dstPos =
			    make_cudaPos(cinstr.get_offset_in_dest()[2] * cinstr.get_element_size(), cinstr.get_offset_in_dest()[1], cinstr.get_offset_in_dest()[0]);
			parms.dstPtr = make_cudaPitchedPtr(
			    dest_base_ptr, cinstr.get_dest_range()[2] * cinstr.get_element_size(), cinstr.get_dest_range()[2], cinstr.get_dest_range()[1]);
			parms.extent = {cinstr.get_copy_range()[2] * cinstr.get_element_size(), cinstr.get_copy_range()[1], cinstr.get_copy_range()[0]};
			parms.kind = cudaMemcpyDefault;
			CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, m_stream.get());
			break;
		}
		default: panic("copy dimensions out of range");
		}
	}
};

std::vector<std::unique_ptr<in_order_instruction_queue>> create_cuda_streams(const int cuda_device_id, const size_t num_streams, allocation_manager& am) {
	std::vector<std::unique_ptr<in_order_instruction_queue>> queues(num_streams);
	std::generate(queues.begin(), queues.end(), [&] { return std::make_unique<cuda_stream>(cuda_device_id, am); });
	return queues;
}

cuda_instruction_queue::cuda_instruction_queue(const int cuda_device_id, const size_t num_streams, allocation_manager &am)
    : multiplex_instruction_queue(create_cuda_streams(cuda_device_id, num_streams, am)) {}

} // namespace celerity::detail
