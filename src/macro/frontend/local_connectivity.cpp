#include "macro/frontend/local_connectivity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace mind_sim::macro::frontend {

LocalConnectivity::LocalConnectivity(int node_count,
                                     std::vector<int> indptr,
                                     std::vector<int> indices,
                                     std::vector<double> weights)
    : node_count_(node_count),
      indptr_(std::move(indptr)),
      indices_(std::move(indices)),
      weights_(std::move(weights)) {
    if (node_count_ <= 0) {
        throw std::runtime_error("LocalConnectivity requires a positive node_count");
    }
    if (indptr_.size() != static_cast<std::size_t>(node_count_ + 1)) {
        throw std::runtime_error("LocalConnectivity indptr size must be node_count + 1");
    }
    if (indices_.size() != weights_.size()) {
        throw std::runtime_error("LocalConnectivity indices and weights size mismatch");
    }
    if (indptr_.front() != 0 || indptr_.back() != static_cast<int>(indices_.size())) {
        throw std::runtime_error("LocalConnectivity indptr must start at 0 and end at nnz");
    }
    for (int row = 0; row < node_count_; ++row) {
        if (indptr_[static_cast<std::size_t>(row)] >
            indptr_[static_cast<std::size_t>(row + 1)]) {
            throw std::runtime_error("LocalConnectivity indptr must be nondecreasing");
        }
    }
    for (std::size_t edge = 0; edge < indices_.size(); ++edge) {
        const int source = indices_[edge];
        if (source < 0 || source >= node_count_) {
            throw std::runtime_error("LocalConnectivity source index out of range");
        }
        if (!std::isfinite(weights_[edge])) {
            throw std::runtime_error("LocalConnectivity weights must be finite");
        }
    }
}

LocalConnectivity LocalConnectivity::from_edges(
    int node_count,
    const std::vector<LocalConnectivityEdge>& edges) {
    if (node_count <= 0) {
        throw std::runtime_error("LocalConnectivity requires a positive node_count");
    }
    std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(node_count));
    for (const auto& edge: edges) {
        if (edge.target_node < 0 || edge.target_node >= node_count ||
            edge.source_node < 0 || edge.source_node >= node_count) {
            throw std::runtime_error("LocalConnectivity edge index out of range");
        }
        if (!std::isfinite(edge.weight)) {
            throw std::runtime_error("LocalConnectivity edge weights must be finite");
        }
        rows[static_cast<std::size_t>(edge.target_node)].push_back(
            {edge.source_node, edge.weight});
    }

    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> weights;
    indptr.reserve(static_cast<std::size_t>(node_count + 1));
    indices.reserve(edges.size());
    weights.reserve(edges.size());
    indptr.push_back(0);
    for (auto& row: rows) {
        std::sort(row.begin(), row.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        for (const auto& [source, weight]: row) {
            indices.push_back(source);
            weights.push_back(weight);
        }
        indptr.push_back(static_cast<int>(indices.size()));
    }
    return LocalConnectivity(node_count, std::move(indptr), std::move(indices), std::move(weights));
}

int LocalConnectivity::node_count() const noexcept {
    return node_count_;
}

std::size_t LocalConnectivity::nnz() const noexcept {
    return indices_.size();
}

const std::vector<int>& LocalConnectivity::indptr() const noexcept {
    return indptr_;
}

const std::vector<int>& LocalConnectivity::indices() const noexcept {
    return indices_;
}

const std::vector<double>& LocalConnectivity::weights() const noexcept {
    return weights_;
}

}  // namespace mind_sim::macro::frontend
