#pragma once

#include "mind_mod/abi.hpp"
#include "micro/sim/types.hpp"
#include "utils/dynamic_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mind_sim::bridge::sim {

class RandomStreamRule {
  public:
    RandomStreamRule(std::string library_path, int state_count);

    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;
    double uniform(double* state, int index, int draw) const;

  private:
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    double (*uniform_)(double*, int, int, int){nullptr};
    int state_count_{0};
};

struct RandomStreamBinding {
    std::shared_ptr<RandomStreamRule> rule{};
    std::vector<double> state{};
};

class MicroInputRule {
  public:
    MicroInputRule(std::string name,
                   std::string library_path,
                   int input_count,
                   int state_count,
                   int param_count,
                   int input_port_count,
                   int random_count);
    explicit MicroInputRule(std::string library_path);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] int input_port_count() const noexcept;
    [[nodiscard]] int random_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void apply(const std::vector<double>& input_soa,
               int roi_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               std::vector<RandomStreamBinding>& random_streams,
               double start_time,
               double stop_time,
               const std::vector<int>& input_port_bases,
               mind_sim::micro::sim::MicroEventTable& events,
               const std::vector<int>& read_input_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mind_mod::MicroInputApplyFn apply_{nullptr};
    int input_count_{0};
    int state_count_{0};
    int param_count_{0};
    int input_port_count_{0};
    int random_count_{0};
};

class MicroOutputRule {
  public:
    MicroOutputRule(std::string name,
                    std::string library_path,
                    int exposure_count,
                    int state_count,
                    int param_count);
    explicit MicroOutputRule(std::string library_path);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void apply(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
               std::vector<double>& exposure_soa,
               int roi_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               double start_time,
               double stop_time,
               const std::vector<int>& write_exposure_offsets) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    mind_sim::mind_mod::MicroOutputApplyFn apply_{nullptr};
    int exposure_count_{0};
    int state_count_{0};
    int param_count_{0};
};

}  // namespace mind_sim::bridge::sim
