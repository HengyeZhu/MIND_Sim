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

const mind_sim::mind_mod::AbiRuleDescriptor& descriptor_of(
    const mind_sim::utils::DynamicLibrary& library,
    mind_sim::mind_mod::AbiRuleKind expected,
    const char* what) {
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mind_mod::DescriptorFn>(library.symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error(std::string(what) + " has null descriptor");
    }
    if (descriptor->abi_version != mind_sim::mind_mod::kMindModAbiVersion) {
        throw std::runtime_error(std::string(what) + " ABI version mismatch");
    }
    if (descriptor->kind != static_cast<int>(expected)) {
        throw std::runtime_error(std::string(what) + " rule kind mismatch");
    }
    if (!descriptor->name || descriptor->name[0] == '\0') {
        throw std::runtime_error(std::string(what) + " descriptor has empty name");
    }
    return *descriptor;
}

int descriptor_name_index(int count, const char* const* names, const char* target) {
    for (int index = 0; index < count; ++index) {
        if (std::string(names[index]) == target) {
            return index;
        }
    }
    throw std::runtime_error(std::string("descriptor is missing STATE referenced by local(): ") +
                             target);
}

}  // namespace

RegionRule::RegionRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      step_(reinterpret_cast<mind_sim::mind_mod::RegionApplyFn>(
          library_->symbol("mind_region_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mind_mod::AbiRuleKind::Region, "RegionRule");
    name_ = descriptor.name;
    input_count_ = descriptor.read_count;
    exposure_count_ = descriptor.write_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
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
                            const std::vector<double>& input_soa,
                            std::vector<double>& exposure_soa,
                            std::vector<double>& state_soa,
                            const std::vector<double>& params_soa,
                            const std::vector<int>& read_input_offsets,
                            const std::vector<int>& write_exposure_offsets,
                            double t,
                            double dt) const {
    mind_sim::mind_mod::AbiRegionContext context{
        .owner_count = static_cast<int>(roi_indices.size()),
        .roi_indices = roi_indices.data(),
        .roi_count = roi_count,
        .input_count = input_count_,
        .input_soa = input_soa.data(),
        .exposure_count = exposure_count_,
        .exposure_soa = exposure_soa.data(),
        .state_count = state_count_,
        .state_soa = state_soa.data(),
        .param_count = param_count_,
        .params_soa = params_soa.data(),
        .read_input_offsets = read_input_offsets.data(),
        .write_exposure_offsets = write_exposure_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
}

NeuralFieldRule::NeuralFieldRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      step_(reinterpret_cast<mind_sim::mind_mod::NeuralFieldApplyFn>(
          library_->symbol("mind_neural_field_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mind_mod::AbiRuleKind::NeuralField, "NeuralFieldRule");
    name_ = descriptor.name;
    input_count_ = descriptor.read_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    if (state_count_ == 0) {
        throw std::runtime_error("NeuralFieldRule state_count must be positive");
    }
    local_state_indices_.reserve(static_cast<std::size_t>(descriptor.local_state_count));
    for (int index = 0; index < descriptor.local_state_count; ++index) {
        local_state_indices_.push_back(descriptor_name_index(descriptor.state_count,
                                                             descriptor.state_names,
                                                             descriptor.local_state_names[index]));
    }
}

const std::string& NeuralFieldRule::name() const noexcept {
    return name_;
}

int NeuralFieldRule::input_count() const noexcept {
    return input_count_;
}

int NeuralFieldRule::state_count() const noexcept {
    return state_count_;
}

int NeuralFieldRule::param_count() const noexcept {
    return param_count_;
}

const std::string& NeuralFieldRule::library_path() const noexcept {
    return library_->path();
}

void NeuralFieldRule::validate_state(const std::vector<double>& state, int node_count) const {
    if (node_count <= 0) {
        throw std::runtime_error("NeuralFieldRule node_count must be positive");
    }
    validate_vector_size(state, state_count_ * node_count, "NeuralFieldRule state");
}

void NeuralFieldRule::validate_params(const std::vector<double>& params) const {
    validate_vector_size(params, param_count_, "NeuralFieldRule params");
}

void NeuralFieldRule::step(int node_count,
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
                           double dt) const {
    const auto node_stride = static_cast<std::size_t>(node_count);
    for (int state: local_state_indices_) {
        const auto offset = static_cast<std::size_t>(state) * node_stride;
        std::copy_n(state_soa.begin() + static_cast<std::ptrdiff_t>(offset),
                    node_stride,
                    previous_state_soa.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    mind_sim::mind_mod::AbiNeuralFieldContext context{
        .node_count = node_count,
        .node_to_roi = node_to_roi.data(),
        .roi_count = roi_count,
        .input_count = input_count_,
        .input_soa = input_soa.data(),
        .state_count = state_count_,
        .previous_state_soa = previous_state_soa.data(),
        .state_soa = state_soa.data(),
        .param_count = param_count_,
        .params = params.data(),
        .local_indptr = local_indptr.data(),
        .local_indices = local_indices.data(),
        .local_weights = local_weights.data(),
        .read_input_offsets = read_input_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
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

CouplingRule::CouplingRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mind_mod::CouplingApplyFn>(
          library_->symbol("mind_coupling_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mind_mod::AbiRuleKind::Coupling, "CouplingRule");
    name_ = descriptor.name;
    input_count_ = descriptor.write_count;
    exposure_count_ = descriptor.read_count;
    param_count_ = descriptor.param_count;
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
                              const std::vector<int>& write_input_offsets) const {
    mind_sim::mind_mod::AbiCouplingContext context{
        .roi_count = roi_count,
        .input_count = input_count,
        .exposure_count = exposure_count,
        .history_capacity = history_capacity,
        .step = step,
        .target_count = static_cast<int>(target_indices.size()),
        .target_indices = target_indices.data(),
        .target_edge_offsets = target_edge_offsets.data(),
        .edge_sources = edge_sources.data(),
        .edge_weights = edge_weights.data(),
        .edge_delay_steps = edge_delay_steps.data(),
        .edge_delay_offsets = edge_delay_offsets.data(),
        .history = history.data(),
        .inputs = inputs.data(),
        .param_count = param_count_,
        .params = params.data(),
        .read_exposure_offsets = read_exposure_offsets.data(),
        .write_input_offsets = write_input_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::macro::sim
