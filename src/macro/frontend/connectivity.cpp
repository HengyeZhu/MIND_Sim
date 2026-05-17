#include "macro/frontend/connectivity.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mind_sim::macro::frontend {

Connectivity::Connectivity(std::vector<std::string> labels,
                           std::vector<std::vector<double>> weights,
                           std::vector<std::vector<double>> delays)
    : labels_(std::move(labels)) {
    if (labels_.empty()) {
        throw std::runtime_error("Connectivity requires at least one ROI");
    }
    rois_.reserve(labels_.size());
    for (int index = 0; index < static_cast<int>(labels_.size()); ++index) {
        auto& label = labels_[static_cast<std::size_t>(index)];
        if (label.empty()) {
            throw std::runtime_error("ROI label must be non-empty");
        }
        const auto [_, inserted] = label_to_index_.emplace(label, index);
        if (!inserted) {
            throw std::runtime_error("ROI labels must be unique");
        }
        rois_.push_back(ROI{.index = index, .label = label});
    }
    weights_ = flatten_square_matrix(weights, "weights", false);
    delays_ = flatten_square_matrix(delays, "delays", true);
}

int Connectivity::roi_count() const noexcept {
    return static_cast<int>(rois_.size());
}

const std::vector<ROI>& Connectivity::rois() const noexcept {
    return rois_;
}

const std::vector<std::string>& Connectivity::labels() const noexcept {
    return labels_;
}

const std::vector<double>& Connectivity::weights() const noexcept {
    return weights_;
}

const std::vector<double>& Connectivity::delays() const noexcept {
    return delays_;
}

double Connectivity::weight_at(int target_roi, int source_roi) const {
    return weights_[matrix_offset(target_roi, source_roi)];
}

double Connectivity::delay_at(int target_roi, int source_roi) const {
    return delays_[matrix_offset(target_roi, source_roi)];
}

double Connectivity::min_positive_delay() const noexcept {
    double value = std::numeric_limits<double>::infinity();
    for (double delay: delays_) {
        if (delay > 0.0 && delay < value) {
            value = delay;
        }
    }
    return std::isfinite(value) ? value : 0.0;
}

int Connectivity::roi_index(const std::string& label) const {
    const auto iter = label_to_index_.find(label);
    if (iter == label_to_index_.end()) {
        throw std::runtime_error("unknown ROI: " + label);
    }
    return iter->second;
}

std::vector<double> Connectivity::flatten_square_matrix(
    const std::vector<std::vector<double>>& matrix,
    const std::string& name,
    bool require_non_negative) const {
    const auto n = labels_.size();
    if (matrix.size() != n) {
        throw std::runtime_error(name + " row count must match ROI count");
    }
    std::vector<double> out;
    out.reserve(n * n);
    for (const auto& row: matrix) {
        if (row.size() != n) {
            throw std::runtime_error(name + " must be square");
        }
        for (double value: row) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(name + " must contain finite values");
            }
            if (require_non_negative && value < 0.0) {
                throw std::runtime_error(name + " must be non-negative");
            }
            out.push_back(value);
        }
    }
    return out;
}

std::size_t Connectivity::matrix_offset(int target_roi, int source_roi) const {
    const int n = roi_count();
    if (target_roi < 0 || target_roi >= n || source_roi < 0 || source_roi >= n) {
        throw std::runtime_error("connectivity matrix index out of range");
    }
    return static_cast<std::size_t>(target_roi * n + source_roi);
}

}  // namespace mind_sim::macro::frontend
