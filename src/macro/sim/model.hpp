#pragma once

#include "macro/sim/types.hpp"
#include "utils/dynamic_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mind_sim::macro::sim {

class RegionRule {
  public:
    RegionRule(std::string name,
               std::string library_path,
               int input_count,
               int exposure_count,
               int state_count,
               int param_count);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void step_group(const std::vector<int>& roi_indices,
                    int roi_count,
                    const std::vector<float>& input_soa,
                    std::vector<float>& exposure_soa,
                    std::vector<double>& state_soa,
                    const std::vector<double>& params_soa,
                    double t,
                    double dt) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    void (*step_)(int,
                  const int*,
                  int,
                  int,
                  int,
                  const float*,
                  float*,
                  int,
                  double*,
                  int,
                  const double*,
                  double,
                  double){nullptr};
    int input_count_{0};
    int exposure_count_{0};
    int state_count_{0};
    int param_count_{0};
};

class CouplingRule {
  public:
    CouplingRule(std::string name,
                 std::string library_path,
                 int input_count,
                 int exposure_count,
                 int param_count);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_params(const std::vector<double>& params) const;
    void apply_flat(int roi_count,
                    int input_count,
                    int history_capacity,
                    int step,
                    const std::vector<int>& target_indices,
                    const std::vector<int>& target_edge_offsets,
                    const std::vector<int>& edge_sources,
                    const std::vector<float>& edge_weights,
                    const std::vector<int>& edge_delay_steps,
                    const std::vector<int>& edge_delay_offsets,
                    const std::vector<float>& history,
                    std::vector<float>& inputs,
                    const std::vector<double>& params) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    void (*apply_)(int,
                   int,
                   int,
                   int,
                   int,
                   int,
                   const int*,
                   const int*,
                   const int*,
                   const float*,
                   const int*,
                   const int*,
                   const float*,
                   float*,
                   int,
                   const double*){nullptr};
    int input_count_{0};
    int exposure_count_{0};
    int param_count_{0};
};

}  // namespace mind_sim::macro::sim
