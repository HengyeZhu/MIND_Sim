#pragma once

#include "micro/sim/types.hpp"
#include "utils/dynamic_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mind_sim::bridge::sim {

struct AbiEventWriter {
    void* user{nullptr};
    void (*write)(void*, double, int){nullptr};
};

struct AbiSpikeTable {
    const double* time{nullptr};
    const int* gid{nullptr};
    int size{0};
};

class MicroInputRule {
  public:
    MicroInputRule(std::string name,
                   std::string library_path,
                   int input_count,
                   int state_count,
                   int param_count,
                   int input_port_count);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] int input_port_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void apply(const std::vector<float>& input_soa,
               int roi_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               double start_time,
               double stop_time,
               const std::vector<int>& input_port_bases,
               mind_sim::micro::sim::MicroEventTable& events) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    void (*apply_)(int,
                   int,
                   int,
                   const float*,
                   int,
                   double*,
                   int,
                   const double*,
                   double,
                   double,
                   int,
                   const int*,
                   AbiEventWriter*){nullptr};
    int input_count_{0};
    int state_count_{0};
    int param_count_{0};
    int input_port_count_{0};
};

class MicroOutputRule {
  public:
    MicroOutputRule(std::string name,
                    std::string library_path,
                    int exposure_count,
                    int state_count,
                    int param_count);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] int state_count() const noexcept;
    [[nodiscard]] int param_count() const noexcept;
    [[nodiscard]] const std::string& library_path() const noexcept;

    void validate_state(const std::vector<double>& state) const;
    void validate_params(const std::vector<double>& params) const;
    void apply(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
               std::vector<float>& exposure_soa,
               int roi_count,
               int roi,
               std::vector<double>& state,
               const std::vector<double>& params,
               double start_time,
               double stop_time) const;

  private:
    std::string name_{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library_{};
    void (*apply_)(int,
                   const AbiSpikeTable*,
                   int,
                   int,
                   float*,
                   int,
                   double*,
                   int,
                   const double*,
                   double,
                   double){nullptr};
    int exposure_count_{0};
    int state_count_{0};
    int param_count_{0};
};

}  // namespace mind_sim::bridge::sim
