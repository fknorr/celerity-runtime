#pragma once

#include <algorithm>
#include <cassert>
#include <list>
#include <optional>
#include <type_traits>

#include <gch/small_vector.hpp>

namespace celerity {
namespace detail {

	enum class dependency_kind {
		anti_dep = 0, // Data anti-dependency, can be resolved by duplicating buffers
		true_dep = 1, // True data flow or temporal dependency
	};

	enum class dependency_origin {
		dataflow,                       // buffer access dependencies generate task and command dependencies
		collective_group_serialization, // all nodes must execute kernels within the same collective group in the same order
		execution_front,                // horizons and epochs are temporally ordered after all preceding tasks or commands on the same node
		last_epoch,                     // nodes without other true-dependencies require an edge to the last epoch for temporal ordering
	};

	// TODO: Move to utility header..?
	template <typename Iterator>
	class iterable_range {
	  public:
		iterable_range(Iterator first, Iterator last) : m_first(first), m_last(last) {}

		Iterator begin() const { return m_first; }
		Iterator end() const { return m_last; }
		friend Iterator begin(const iterable_range& ir) { return ir.m_first; }
		friend Iterator end(const iterable_range& ir) { return ir.m_last; }

		auto& front() const { return *m_first; }
		bool empty() const { return m_first == m_last; }

	  private:
		Iterator m_first;
		Iterator m_last;
	};

	template <typename Node>
	class intrusive_graph_node {
	  public:
		struct dependency {
			Node* node;
			dependency_kind kind;
			dependency_origin origin; // context information for graph printing
		};

	  public:
		intrusive_graph_node() { static_assert(std::is_base_of<intrusive_graph_node<Node>, Node>::value, "Node must be child class (CRTP)"); }

	  protected:
		~intrusive_graph_node() { // protected: Statically disallow destruction through base pointer, since dtor is not polymorphic
			for(const auto& dep : m_dependencies) {
				auto& back_edges = dep.node->m_dependent_nodes;
				back_edges.erase(std::remove(back_edges.begin(), back_edges.end(), this), back_edges.end());
			}
			for(const auto node : m_dependent_nodes) {
				auto& forward_edges = node->m_dependencies;
				forward_edges.erase(
				    std::remove_if(forward_edges.begin(), forward_edges.end(), [=](const dependency& dep) { return dep.node == this; }), forward_edges.end());
			}
		}

	  public:
		void add_dependency(dependency dep) {
			// Check for (direct) cycles
			assert(!has_dependent(dep.node));

			if(const auto it = find_by_node(m_dependencies, dep.node); it != m_dependencies.end()) {
				// We assume that for dependency kinds A and B, max(A, B) is strong enough to satisfy both.
				static_assert(dependency_kind::anti_dep < dependency_kind::true_dep);

				// Already exists, potentially upgrade to full dependency
				if(it->kind < dep.kind) {
					it->kind = dep.kind;
					it->origin = dep.origin; // This unfortunately loses origin information from the lesser dependency
				}
				return;
			}

			m_dependencies.emplace_back(dep);
			dep.node->m_dependent_nodes.emplace_back(static_cast<Node*>(this));

			m_pseudo_critical_path_length =
			    std::max(m_pseudo_critical_path_length, static_cast<intrusive_graph_node*>(dep.node)->m_pseudo_critical_path_length + 1);
		}

		void remove_dependency(Node* node) {
			const auto forward_it = find_by_node(m_dependencies, node);
			const auto backward_it = std::find(node->m_dependent_nodes.begin(), node->m_dependent_nodes.end(), this);
			assert((forward_it == m_dependencies.end()) == (backward_it == node->m_dependent_nodes.end()));
			if(forward_it != m_dependencies.end()) { m_dependencies.erase(forward_it); }
			if(backward_it != node->m_dependent_nodes.end()) { node->m_dependent_nodes.erase(backward_it); }
		}

		const dependency& get_dependency(const Node* const node) const {
			const auto it = find_by_node(m_dependencies, node);
			assert(it != m_dependencies.end());
			return *it;
		}

		bool has_dependency(const Node* const node, const std::optional<dependency_kind> kind = std::nullopt) const {
			const auto it = find_by_node(m_dependencies, node);
			if(it == m_dependencies.end()) return false;
			return kind != std::nullopt ? it->kind == kind : true;
		}

		bool has_dependent(const Node* const node, const std::optional<dependency_kind> kind = std::nullopt) const {
			return node->has_dependency(static_cast<const Node*>(this), kind);
		}

		auto get_dependencies() const { return iterable_range{m_dependencies.cbegin(), m_dependencies.cend()}; }
		auto get_dependent_nodes() const { return iterable_range{m_dependent_nodes.cbegin(), m_dependent_nodes.cend()}; }

		int get_pseudo_critical_path_length() const { return m_pseudo_critical_path_length; }

	  private:
		gch::small_vector<dependency> m_dependencies;
		gch::small_vector<Node*> m_dependent_nodes;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		// (that is all that is needed for celerity use).
		int m_pseudo_critical_path_length = 0;

		template <typename Range>
		static auto find_by_node(Range& rng, const Node* const node) {
			using std::begin, std::end;
			return std::find_if(begin(rng), end(rng), [=](auto& d) { return d.node == node; });
		}
	};

} // namespace detail
} // namespace celerity
