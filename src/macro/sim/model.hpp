#pragma once

#include "macro/sim/types.hpp"
#include "mod/rule_api.hpp"
#include "utils/dynamic_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mind_sim::macro::sim {

class RegionRule {
  public:
    RegionRule(std::string library_path, std::string rule_name);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;
    [[nodiscard]] const std::vector<std::string>& exposure_names() const noexcept;
    [[nodiscard]] const std::vector<std::string>& state_names() const noexcept;
    [[nodiscard]] const std::vector<double>& state_defaults() const noexcept;
    [[nodiscard]] const std::vector<std::string>& param_names() const noexcept;
    [[nodiscard]] const std::vector<double>& param_defaults() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void step_group(const std::vector<int>& roi_indices,
                    int roi_count,
                    std::vector<double>& exposure_soa,
                    std::vector<double>& state_soa,
                    const std::vector<double>& params_soa,
                    const std::vector<int>& exposure_offsets,
                    double t,
                    double dt) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mod::RegionApplyFn step_{nullptr};
    std::vector<std::string> exposure_names_{};
    std::vector<std::string> state_names_{};
    std::vector<double> state_defaults_{};
    std::vector<std::string> param_names_{};
    std::vector<double> param_defaults_{};
};

class MacroToMacroRule {
  public:
    MacroToMacroRule(std::string library_path, std::string rule_name);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int read_source_count() const noexcept;
    [[nodiscard]] int read_target_count() const noexcept;
    [[nodiscard]] int write_source_count() const noexcept;
    [[nodiscard]] int write_target_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;
    [[nodiscard]] const std::vector<std::string>& read_source_exposure_names() const noexcept;
    [[nodiscard]] const std::vector<std::string>& read_target_exposure_names() const noexcept;
    [[nodiscard]] const std::vector<std::string>& write_source_exposure_names() const noexcept;
    [[nodiscard]] const std::vector<std::string>& write_target_exposure_names() const noexcept;
    [[nodiscard]] const std::vector<std::string>& param_names() const noexcept;
    [[nodiscard]] const std::vector<double>& param_defaults() const noexcept;

    void validate_params(const std::vector<double>& params) const;
    void apply_flat(int roi_count,
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
                    const std::vector<double>& current_exposures,
                    std::vector<double>& exposures,
                    const std::vector<double>& params,
                    const std::vector<int>& read_source_offsets,
                    const std::vector<int>& read_target_offsets,
                    const std::vector<int>& write_source_offsets,
                    const std::vector<int>& write_target_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mod::MacroToMacroApplyFn apply_{nullptr};
    std::vector<std::string> read_source_exposure_names_{};
    std::vector<std::string> read_target_exposure_names_{};
    std::vector<std::string> write_source_exposure_names_{};
    std::vector<std::string> write_target_exposure_names_{};
    std::vector<std::string> param_names_{};
    std::vector<double> param_defaults_{};
};

}  // namespace mind_sim::macro::sim
