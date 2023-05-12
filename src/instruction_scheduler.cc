#include "instruction_scheduler.h"
#include "buffer_storage.h" // TODO included for CELERITY_CUDA_CHECK, consider moving that
#include "closure_hydrator.h"
#include <cuda_runtime.h> // TODO move CUDA stuff to separate source

namespace celerity::detail {

class cuda_event final : public instruction_queue_event_impl {
  public:
	cuda_event() {
		cudaEvent_t event;
		CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
		m_event = std::unique_ptr<CUevent_st, deleter>(event);
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
	explicit cuda_stream(allocation_manager& am) : m_allocation_mgr(&am) {
		cudaStream_t stream;
		CELERITY_CUDA_CHECK(cudaStreamCreate, &stream);
		m_stream = std::unique_ptr<CUstream_st, deleter>(stream);
	}

	instruction_queue_event submit(const instruction& instr) override {
		utils::match(
		    instr, //
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

class sycl_event_impl : public instruction_queue_event_impl {
  public:
	sycl_event_impl(sycl::event event) : m_event(std::move(event)) {}
	void block_on() override { m_event.wait(); }
	const sycl::event& get() const { return m_event; }

  private:
	sycl::event m_event;
};

class sycl_queue : public out_of_order_instruction_queue {
  public:
	explicit sycl_queue(sycl::queue q, allocation_manager& am) : m_queue(std::move(q)), m_allocation_mgr(&am) {}

	instruction_queue_event submit(const instruction& instr, const std::vector<instruction_queue_event>& dependencies) override {
		auto sycl_event = m_queue.submit([&](sycl::handler& cgh) {
			for(auto& dep : dependencies) {
				cgh.depends_on(dynamic_cast<const sycl_event_impl&>(*dep).get());
			}
			utils::match(
			    instr, //
			    [&](const sycl_kernel_instruction& skinstr) {
				    std::vector<closure_hydrator::NOCOMMIT_info> access_infos;
				    access_infos.reserve(skinstr.get_allocation_map().size());
				    for(const auto& aa : skinstr.get_allocation_map()) {
					    access_infos.push_back(closure_hydrator::NOCOMMIT_info{
					        target::device, m_allocation_mgr->get_pointer(aa.aid), aa.allocation_range, aa.offset_in_allocation, aa.buffer_subrange});
				    }
				    closure_hydrator::get_instance().prepare(std::move(access_infos));
				    skinstr.launch(cgh);
			    },
			    [](const auto& /* other */) { panic("invalid instruction type for sycl_queue"); });
		});
		return std::make_unique<sycl_event_impl>(std::move(sycl_event));
	}

  private:
	sycl::queue m_queue;
	allocation_manager* m_allocation_mgr;
};

void instruction_scheduler::submit(const instruction& instr) { instr.get_target_port(); }

} // namespace celerity::detail
