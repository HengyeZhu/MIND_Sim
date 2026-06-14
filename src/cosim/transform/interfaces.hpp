#pragma once

#include "mod/abi.hpp"
#include "micro/sim/types.hpp"
#include "utils/dynamic_library.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mind_sim::cosim::transform {

class MicroInputRule {
  public:
    MicroInputRule(std::string library_path, std::string rule_name);

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

    void validate_state(const std::vector<double>& state, int source_count) const;
    void validate_params(const std::vector<double>& params) const;
    void apply(const std::vector<double>& exposure_trace_soa,
               int sample_count,
               double sample_dt,
               int network_exposure_count,
               int roi_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               double start_time,
               double stop_time,
               std::uint64_t rng_seed,
               int exchange_start_step,
               const std::vector<int>& source_indices,
               const std::vector<int>& source_ids,
               mind_sim::micro::sim::MicroEventTable& events,
               const std::vector<int>& exposure_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mod::MicroInputApplyFn apply_{nullptr};
    int exposure_count_{0};
    int state_count_{0};
    int param_count_{0};
    std::vector<std::string> exposure_names_{};
    std::vector<std::string> state_names_{};
    std::vector<double> state_defaults_{};
    std::vector<std::string> param_names_{};
    std::vector<double> param_defaults_{};
};

class MicroOutputRule {
  public:
    MicroOutputRule(std::string library_path, std::string rule_name);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int output_count() const noexcept;
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
    void apply(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
               std::vector<double>& output_soa,
               std::vector<double>& output_trace_soa,
               int roi_count,
               int output_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               double start_time,
               double stop_time,
               int sample_count,
               double sample_dt,
               const std::vector<int>& exposure_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mod::MicroOutputApplyFn apply_{nullptr};
    int output_count_{0};
    int state_count_{0};
    int param_count_{0};
    std::vector<std::string> exposure_names_{};
    std::vector<std::string> state_names_{};
    std::vector<double> state_defaults_{};
    std::vector<std::string> param_names_{};
    std::vector<double> param_defaults_{};
};

}  // namespace mind_sim::cosim::transform
