#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace mind_sim::macro::frontend {

struct LocalConnectivityEdge {
    int target_node{0};
    int source_node{0};
    double weight{0.0};
};

class LocalConnectivity {
  public:
    LocalConnectivity(int node_count,
                      std::vector<int> indptr,
                      std::vector<int> indices,
                      std::vector<double> weights);

    [[nodiscard]] static LocalConnectivity from_edges(
        int node_count,
        const std::vector<LocalConnectivityEdge>& edges);

    [[nodiscard]] int node_count() const noexcept;
    [[nodiscard]] std::size_t nnz() const noexcept;
    [[nodiscard]] const std::vector<int>& indptr() const noexcept;
    [[nodiscard]] const std::vector<int>& indices() const noexcept;
    [[nodiscard]] const std::vector<double>& weights() const noexcept;

  private:
    int node_count_{0};
    std::vector<int> indptr_{};
    std::vector<int> indices_{};
    std::vector<double> weights_{};
};

}  // namespace mind_sim::macro::frontend
