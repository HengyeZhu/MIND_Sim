#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_sim::macro::frontend {

struct ROI {
    int index{-1};
    std::string label{};
};

class Connectivity {
  public:
    Connectivity(std::vector<std::string> labels,
                 std::vector<std::vector<double>> weights,
                 std::vector<std::vector<double>> delays);

    [[nodiscard]] int roi_count() const noexcept;
    [[nodiscard]] const std::vector<ROI>& rois() const noexcept;
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept;
    [[nodiscard]] const std::vector<double>& weights() const noexcept;
    [[nodiscard]] const std::vector<double>& delays() const noexcept;
    [[nodiscard]] double weight_at(int target_roi, int source_roi) const;
    [[nodiscard]] double delay_at(int target_roi, int source_roi) const;
    [[nodiscard]] double min_positive_delay() const noexcept;
    [[nodiscard]] int roi_index(const std::string& label) const;

  private:
    [[nodiscard]] std::vector<double> flatten_square_matrix(
        const std::vector<std::vector<double>>& matrix,
        const std::string& name,
        bool require_non_negative) const;
    [[nodiscard]] std::size_t matrix_offset(int target_roi, int source_roi) const;

    std::vector<std::string> labels_{};
    std::vector<ROI> rois_{};
    std::unordered_map<std::string, int> label_to_index_{};
    std::vector<double> weights_{};
    std::vector<double> delays_{};
};

}  // namespace mind_sim::macro::frontend
