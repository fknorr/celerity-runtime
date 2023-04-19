#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "instruction_graph_generator.h"
#include "print_graph.h"
#include "task_ring_buffer.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;


class idag_test_context;

class idag_task_builder {
	friend class idag_test_context;

	using action = std::function<void(handler&)>;

	class step {
	  public:
		step(idag_test_context& ictx, std::deque<action> actions) : m_ictx(ictx), m_actions(std::move(actions)) {}
		virtual ~step() noexcept(false);

		task_id submit();

		step(const step&) = delete;

	  private:
		idag_test_context& m_ictx;
		std::deque<action> m_actions;

	  protected:
		template <typename StepT>
		StepT chain(action a) {
			static_assert(std::is_base_of_v<step, StepT>);
			m_actions.push_front(a);
			return StepT{m_ictx, std::move(m_actions)};
		}
	};

	class buffer_access_step : public step {
	  public:
		buffer_access_step(idag_test_context& ictx, std::deque<action> actions) : step(ictx, std::move(actions)) {}

		buffer_access_step(const buffer_access_step&) = delete;

		template <typename BufferT, typename RangeMapper>
		buffer_access_step read(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		buffer_access_step read_write(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read_write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		buffer_access_step discard_write(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
		}

		// FIXME: Misnomer (not a "buffer access")
		template <typename HostObjT>
		buffer_access_step affect(HostObjT& host_obj) {
			return chain<buffer_access_step>([&host_obj](handler& cgh) { host_obj.add_side_effect(cgh, experimental::side_effect_order::sequential); });
		}

		// FIXME: Misnomer (not a "buffer access")
		template <typename Hint>
		buffer_access_step hint(Hint hint) {
			return chain<buffer_access_step>([&hint](handler& cgh) { experimental::hint(cgh, hint); });
		}
	};

  public:
	template <typename Name, int Dims>
	buffer_access_step device_compute(const range<Dims>& global_size, const id<Dims>& global_offset) {
		std::deque<action> actions;
		actions.push_front([global_size, global_offset](handler& cgh) { cgh.parallel_for<Name>(global_size, global_offset, [](id<Dims>) {}); });
		return buffer_access_step(m_ictx, std::move(actions));
	}

	template <typename Name, int Dims>
	buffer_access_step device_compute(const nd_range<Dims>& nd_range) {
		std::deque<action> actions;
		actions.push_front([nd_range](handler& cgh) { cgh.parallel_for<Name>(nd_range, [](nd_item<Dims>) {}); });
		return buffer_access_step(m_ictx, std::move(actions));
	}

	template <int Dims>
	buffer_access_step host_task(const range<Dims>& global_size) {
		std::deque<action> actions;
		actions.push_front([global_size](handler& cgh) { cgh.host_task(global_size, [](partition<Dims>) {}); });
		return buffer_access_step(m_ictx, std::move(actions));
	}

  private:
	idag_test_context& m_ictx;

	idag_task_builder(idag_test_context& ictx) : m_ictx(ictx) {}
};

// According to Wikipedia https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
std::vector<abstract_command*> topsort(std::unordered_set<abstract_command*> unmarked) {
	std::unordered_set<abstract_command*> temporary_marked;
	std::unordered_set<abstract_command*> permanent_marked;
	std::vector<abstract_command*> sorted(unmarked.size());
	auto sorted_front = sorted.rbegin();

	const auto visit = [&](abstract_command* const cmd, auto& visit /* to allow recursion in lambda */) {
		if(permanent_marked.count(cmd) != 0) return;
		assert(temporary_marked.count(cmd) == 0 && "cyclic command graph");
		unmarked.erase(cmd);
		temporary_marked.insert(cmd);
		for(const auto dep_cmd : cmd->get_dependent_nodes()) {
			visit(dep_cmd, visit);
		}
		temporary_marked.erase(cmd);
		permanent_marked.insert(cmd);
		*sorted_front++ = cmd;
	};

	while(!unmarked.empty()) {
		visit(*unmarked.begin(), visit);
	}
	return sorted;
}


class idag_test_context {
	friend class idag_task_builder;

  public:
	explicit idag_test_context(size_t num_nodes, node_id my_nid, size_t num_devices_per_node) : m_num_nodes(num_nodes), m_my_nid(my_nid) {
		m_rm = std::make_unique<reduction_manager>();
		m_tm = std::make_unique<task_manager>(num_nodes, nullptr /* host_queue */);
		m_cdag = std::make_unique<command_graph>();
		m_dggen = std::make_unique<distributed_graph_generator>(num_nodes, 1 /* devices per node, TODO */, my_nid, *m_cdag, *m_tm);
		m_iggen = std::make_unique<instruction_graph_generator>(*m_tm, num_devices_per_node);
	}

	~idag_test_context() { maybe_print_graphs(); }

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm->add_buffer(bid, Dims, range_cast<3>(size), mark_as_host_initialized);
		m_dggen->add_buffer(bid, range_cast<3>(size), Dims);
		m_iggen->register_buffer(bid, Dims, range_cast<3>(size), 1 /* size */, 1 /* align */);
		return buf;
	}

	test_utils::mock_host_object create_host_object() { return test_utils::mock_host_object{m_next_host_object_id++}; }

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size, const id<Dims>& global_offset = {}) {
		return idag_task_builder(*this).device_compute<Name>(global_size, global_offset);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const nd_range<Dims>& nd_range) {
		return idag_task_builder(*this).device_compute<Name>(nd_range);
	}

	template <int Dims>
	auto host_task(const range<Dims>& global_size) {
		return idag_task_builder(*this).host_task(global_size);
	}

	void set_horizon_step(const int step) { m_tm->set_horizon_step(step); }

	const instruction_graph& get_instruction_graph() { return m_iggen->get_graph(); }

  private:
	size_t m_num_nodes;
	node_id m_my_nid;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	std::optional<task_id> m_most_recently_built_horizon;
	std::unique_ptr<reduction_manager> m_rm;
	std::unique_ptr<task_manager> m_tm;
	std::unique_ptr<command_graph> m_cdag;
	std::unique_ptr<distributed_graph_generator> m_dggen;
	std::unique_ptr<instruction_graph_generator> m_iggen;

	task_manager& get_task_manager() { return *m_tm; }

	void build_task(const task_id tid) { compile_commands(m_dggen->build_task(*m_tm->get_task(tid))); }

	void maybe_build_horizon() {
		const auto current_horizon = task_manager_testspy::get_current_horizon(*m_tm);
		if(m_most_recently_built_horizon != current_horizon) { build_task(current_horizon.value()); }
		m_most_recently_built_horizon = current_horizon;
	}

	void compile_commands(std::unordered_set<abstract_command*>&& cmds) {
		for(const auto cmd : topsort(std::move(cmds))) {
			m_iggen->compile(*cmd);
		}
	}

	void maybe_print_graphs() {
		if(test_utils::print_graphs) {
			test_utils::maybe_print_graph(*m_tm);
			CELERITY_INFO("Command graph:\n\n{}\n", m_cdag->print_graph(m_my_nid, std::numeric_limits<size_t>::max(), *m_tm, nullptr).value());
			CELERITY_INFO("Instruction graph:\n\n{}\n", print_instruction_graph(m_iggen->get_graph(), *m_cdag, *m_tm));
		}
	}
};

idag_task_builder::step::~step() noexcept(false) {
	if(!m_actions.empty()) { throw std::runtime_error("Found incomplete task build. Did you forget to call submit()?"); }
}

task_id idag_task_builder::step::submit() {
	assert(!m_actions.empty());
	const auto tid = m_ictx.get_task_manager().submit_command_group([this](handler& cgh) {
		while(!m_actions.empty()) {
			auto a = m_actions.front();
			a(cgh);
			m_actions.pop_front();
		}
	});
	m_ictx.build_task(tid);
	m_ictx.maybe_build_horizon();
	m_actions.clear();
	return tid;
}

TEST_CASE("trivial graph", "[instruction graph]") {
	idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	ictx.device_compute<class UKN(kernel)>(test_range).submit();
}

TEST_CASE("graph with only writes", "[instruction graph]") {
	idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
}

TEST_CASE("resize and overwrite", "[instruction graph]") {
	idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer(range<1>(256));
	ictx.device_compute<class UKN(writer)>(range<1>(1)).discard_write(buf1, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute<class UKN(writer)>(range<1>(1)).discard_write(buf1, acc::fixed<1>({64, 196})).submit();
	// TODO assert that we do not copy the overwritten buffer portion
}

TEST_CASE("communication-free dataflow", "[instruction graph]") {
	idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::one_to_one()).submit();
}

TEST_CASE("communication-free dataflow with copies", "[instruction graph]") {
	idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("simple communication", "[instruction graph]") {
	idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(writer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(reader)>(test_range).read(buf1, acc::all()).submit();
}

TEST_CASE("large graph", "[instruction graph]") {
	idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);

	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	auto buf2 = ictx.create_buffer(test_range);

	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf2, acc::all()).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf2, acc::all()).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf2, acc::all()).submit();
}

TEST_CASE("oversubscribe", "[instruction-graph]") {
	idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);

	const range<1> test_range = {256};
	auto buf = ictx.create_buffer(test_range);

	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf, acc::one_to_one()).hint(hints::oversubscribe(2)).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf, acc::all()).hint(hints::oversubscribe(2)).submit();
}

TEST_CASE("recv split", "[instruction-graph]") {
	idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);

	const auto reverse_one_to_one = [](chunk<1> ck) -> subrange<1> { return {ck.global_size[0] - ck.range[0] - ck.offset[0], ck.range[0]}; };

	const range<1> test_range = {256};
	auto buf = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf, reverse_one_to_one).submit();
}

TEST_CASE("transitive copy dependencies", "[instruction-graph]") {
	idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 1 /* devices */);

	const auto reverse_one_to_one = [](chunk<1> ck) -> subrange<1> { return {ck.global_size[0] - ck.range[0] - ck.offset[0], ck.range[0]}; };

	const range<1> test_range = {256};
	auto buf1 = ictx.create_buffer(test_range);
	auto buf2 = ictx.create_buffer(test_range);
	ictx.device_compute<class UKN(producer)>(test_range).discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(gather)>(test_range).read(buf1, acc::all()).discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute<class UKN(consumer)>(test_range).read(buf2, reverse_one_to_one).submit();
}
