#pragma once

#include "macro/sim/types.hpp"
#include "mind_mod/abi.hpp"
#include "utils/dynamic_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mind_sim::macro::sim {

class RegionRule {
  public:
    explicit RegionRule(std::string library_path);

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
                    const std::vector<double>& input_soa,
                    std::vector<double>& exposure_soa,
                    std::vector<double>& state_soa,
                    const std::vector<double>& params_soa,
                    const std::vector<int>& read_input_offsets,
                    const std::vector<int>& write_exposure_offsets,
                    double t,
                    double dt) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mind_mod::RegionApplyFn step_{nullptr};
    int input_count_{0};
    int exposure_count_{0};
    int state_count_{0};
    int param_count_{0};
};

class NeuralFieldRule {
  public:
    explicit NeuralFieldRule(std::string library_path);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state, int node_count) const;
    void validate_params(const std::vector<double>& params) const;
    void step(int node_count,
              const std::vector<int>& node_to_roi,
              int roi_count,
              const std::vector<double>& input_soa,
              std::vector<double>& previous_state_soa,
              std::vector<double>& state_soa,
              const std::vector<double>& params,
              const std::vector<int>& local_indptr,
              const std::vector<int>& local_indices,
              const std::vector<double>& local_weights,
              const std::vector<int>& read_input_offsets,
              double t,
              double dt) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mind_mod::NeuralFieldApplyFn step_{nullptr};
    int input_count_{0};
    int state_count_{0};
    int param_count_{0};
    std::vector<int> local_state_indices_{};
};

class CouplingRule {
  public:
    CouplingRule(std::string name,
                 std::string library_path,
                 int input_count,
                 int exposure_count,
                 int param_count);
    explicit CouplingRule(std::string library_path);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_params(const std::vector<double>& params) const;
    void apply_flat(int roi_count,
                    int input_count,
                    int exposure_count,
                    int history_capacity,
                    int step,
                    const std::vector<int>& target_indices,
                    const std::vector<int>& target_edge_offsets,
                    const std::vector<int>& edge_sources,
                    const std::vector<double>& edge_weights,
                    const std::vector<int>& edge_delay_steps,
                    const std::vector<int>& edge_delay_offsets,
                    const std::vector<double>& history,
                    std::vector<double>& inputs,
                    const std::vector<double>& params,
                    const std::vector<int>& read_exposure_offsets,
                    const std::vector<int>& write_input_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mind_mod::CouplingApplyFn apply_{nullptr};
    int input_count_{0};
    int exposure_count_{0};
    int param_count_{0};
};

}  // namespace mind_sim::macro::sim
