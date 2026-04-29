#include "macro/sim/model.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::macro::sim {

namespace {

void validate_count(int count, const char* what) {
    if (count < 0) {
        throw std::runtime_error(std::string(what) + " must be non-negative");
    }
}

void validate_vector_size(const std::vector<double>& values, int expected, const char* what) {
    if (values.size() != static_cast<std::size_t>(expected)) {
        throw std::runtime_error(std::string(what) + " size mismatch");
    }
}

}  // namespace

RegionRule::RegionRule(std::string name,
                       std::string library_path,
                       int input_count,
                       int exposure_count,
                       int state_count,
                       int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      step_(reinterpret_cast<decltype(step_)>(library_->symbol("mind_region_rule_step"))),
      input_count_(input_count),
      exposure_count_(exposure_count),
      state_count_(state_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("RegionRule name must be non-empty");
    }
    validate_count(input_count_, "RegionRule input_count");
    validate_count(exposure_count_, "RegionRule exposure_count");
    validate_count(state_count_, "RegionRule state_count");
    validate_count(param_count_, "RegionRule param_count");
    if (exposure_count_ == 0) {
        throw std::runtime_error("RegionRule exposure_count must be positive");
    }
}

const std::string& RegionRule::name() const noexcept {
    return name_;
}

int RegionRule::input_count() const noexcept {
    return input_count_;
}

int RegionRule::exposure_count() const noexcept {
    return exposure_count_;
}

int RegionRule::state_count() const noexcept {
    return state_count_;
}

int RegionRule::param_count() const noexcept {
    return param_count_;
}

const std::string& RegionRule::library_path() const noexcept {
    return library_->path();
}

void RegionRule::validate_state(const std::vector<double>& state) const {
    validate_vector_size(state, state_count_, "RegionRule state");
}

void RegionRule::validate_params(const std::vector<double>& params) const {
    validate_vector_size(params, param_count_, "RegionRule params");
}

void RegionRule::step_group(const std::vector<int>& roi_indices,
                            int roi_count,
                            const std::vector<float>& input_soa,
                            std::vector<float>& exposure_soa,
                            std::vector<double>& state_soa,
                            const std::vector<double>& params_soa,
                            double t,
                            double dt) const {
    const int owner_count = static_cast<int>(roi_indices.size());
    step_(owner_count,
          roi_indices.data(),
          roi_count,
          input_count_,
          exposure_count_,
          input_soa.data(),
          exposure_soa.data(),
          state_count_,
          state_soa.data(),
          param_count_,
          params_soa.data(),
          t,
          dt);
}

CouplingRule::CouplingRule(std::string name,
                           std::string library_path,
                           int input_count,
                           int exposure_count,
                           int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_coupling_rule_apply"))),
      input_count_(input_count),
      exposure_count_(exposure_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("CouplingRule name must be non-empty");
    }
    validate_count(input_count_, "CouplingRule input_count");
    validate_count(exposure_count_, "CouplingRule exposure_count");
    validate_count(param_count_, "CouplingRule param_count");
    if (exposure_count_ == 0) {
        throw std::runtime_error("CouplingRule exposure_count must be positive");
    }
}

const std::string& CouplingRule::name() const noexcept {
    return name_;
}

int CouplingRule::input_count() const noexcept {
    return input_count_;
}

int CouplingRule::exposure_count() const noexcept {
    return exposure_count_;
}

int CouplingRule::param_count() const noexcept {
    return param_count_;
}

const std::string& CouplingRule::library_path() const noexcept {
    return library_->path();
}

void CouplingRule::validate_params(const std::vector<double>& params) const {
    validate_vector_size(params, param_count_, "CouplingRule params");
}

void CouplingRule::apply_flat(int roi_count,
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
                              const std::vector<double>& params) const {
    apply_(roi_count,
           input_count,
           exposure_count_,
           history_capacity,
           step,
           static_cast<int>(target_indices.size()),
           target_indices.data(),
           target_edge_offsets.data(),
           edge_sources.data(),
           edge_weights.data(),
           edge_delay_steps.data(),
           edge_delay_offsets.data(),
           history.data(),
           inputs.data(),
           param_count_,
           params.data());
}

}  // namespace mind_sim::macro::sim
