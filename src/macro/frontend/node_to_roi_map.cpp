#include "macro/frontend/node_to_roi_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mind_sim::macro::frontend {

namespace {

template <typename Value>
std::vector<Value> read_vector_file(const std::string& path, const char* name) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("could not open ") + name + ": " + path);
    }
    std::vector<Value> values;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        Value value{};
        while (row >> value) {
            values.push_back(value);
        }
    }
    return values;
}

}  // namespace

NodeToRoiMap::NodeToRoiMap(std::vector<int> node_to_roi,
                           std::vector<double> node_weights)
    : node_to_roi_(std::move(node_to_roi)),
      node_weights_(std::move(node_weights)) {
    if (node_to_roi_.empty()) {
        throw std::runtime_error("NodeToRoiMap requires at least one node");
    }
    for (int roi: node_to_roi_) {
        if (roi < 0) {
            throw std::runtime_error("NodeToRoiMap ROI indices must be non-negative");
        }
    }
    if (node_weights_.empty()) {
        node_weights_.assign(node_to_roi_.size(), 1.0);
    }
    if (node_weights_.size() != node_to_roi_.size()) {
        throw std::runtime_error("NodeToRoiMap node_weights size must match node_to_roi");
    }
    for (double weight: node_weights_) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::runtime_error("NodeToRoiMap node_weights must be finite and non-negative");
        }
    }
}

NodeToRoiMap NodeToRoiMap::from_file(const std::string& node_to_roi_path,
                                     const std::string& node_weights_path) {
    return NodeToRoiMap(
        read_vector_file<int>(node_to_roi_path, "NodeToRoiMap node_to_roi file"),
        node_weights_path.empty()
            ? std::vector<double>{}
            : read_vector_file<double>(node_weights_path, "NodeToRoiMap node_weights file"));
}

int NodeToRoiMap::node_count() const noexcept {
    return static_cast<int>(node_to_roi_.size());
}

const std::vector<int>& NodeToRoiMap::node_to_roi() const noexcept {
    return node_to_roi_;
}

const std::vector<double>& NodeToRoiMap::node_weights() const noexcept {
    return node_weights_;
}

}  // namespace mind_sim::macro::frontend
