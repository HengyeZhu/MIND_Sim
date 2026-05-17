#pragma once

#include <string>
#include <vector>

namespace mind_sim::macro::frontend {

class NodeToRoiMap {
  public:
    explicit NodeToRoiMap(std::vector<int> node_to_roi,
                          std::vector<double> node_weights = {});

    [[nodiscard]] static NodeToRoiMap from_file(
        const std::string& node_to_roi_path,
        const std::string& node_weights_path = {});

    [[nodiscard]] int node_count() const noexcept;
    [[nodiscard]] const std::vector<int>& node_to_roi() const noexcept;
    [[nodiscard]] const std::vector<double>& node_weights() const noexcept;

  private:
    std::vector<int> node_to_roi_{};
    std::vector<double> node_weights_{};
};

}  // namespace mind_sim::macro::frontend
