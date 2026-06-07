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

const mind_sim::mod::AbiRuleDescriptor& descriptor_of(
    const mind_sim::utils::DynamicLibrary& library,
    mind_sim::mod::AbiRuleKind expected,
    const char* what) {
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mod::DescriptorFn>(library.symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error(std::string(what) + " has null descriptor");
    }
    if (descriptor->abi_version != mind_sim::mod::kModAbiVersion) {
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

std::vector<std::string> descriptor_names(int count, const char* const* names) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.emplace_back(names[index]);
    }
    return out;
}

std::vector<double> descriptor_defaults(int count, const double* values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.push_back(values[index]);
    }
    return out;
}

}  // namespace

RegionRule::RegionRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      step_(reinterpret_cast<mind_sim::mod::RegionApplyFn>(
          library_->symbol("mind_region_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mod::AbiRuleKind::Region, "RegionRule");
    name_ = descriptor.name;
    input_count_ = descriptor.target_input_count;
    output_count_ = descriptor.source_exposure_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    target_input_names_ = descriptor_names(descriptor.target_input_count, descriptor.target_input_names);
    source_exposure_names_ =
        descriptor_names(descriptor.source_exposure_count, descriptor.source_exposure_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
    if (output_count_ == 0) {
        throw std::runtime_error("RegionRule output_count must be positive");
    }
}

const std::string& RegionRule::name() const noexcept {
    return name_;
}

int RegionRule::input_count() const noexcept {
    return input_count_;
}

int RegionRule::output_count() const noexcept {
    return output_count_;
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

const std::vector<std::string>& RegionRule::target_input_names() const noexcept {
    return target_input_names_;
}

const std::vector<std::string>& RegionRule::source_exposure_names() const noexcept {
    return source_exposure_names_;
}

const std::vector<std::string>& RegionRule::state_names() const noexcept {
    return state_names_;
}

const std::vector<double>& RegionRule::state_defaults() const noexcept {
    return state_defaults_;
}

const std::vector<std::string>& RegionRule::param_names() const noexcept {
    return param_names_;
}

const std::vector<double>& RegionRule::param_defaults() const noexcept {
    return param_defaults_;
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
                            std::vector<double>& output_soa,
                            std::vector<double>& state_soa,
                            const std::vector<double>& params_soa,
                            const std::vector<int>& target_input_offsets,
                            const std::vector<int>& source_exposure_offsets,
                            double t,
                            double dt) const {
    mind_sim::mod::AbiRegionContext context{
        .owner_count = static_cast<int>(roi_indices.size()),
        .roi_indices = roi_indices.data(),
        .roi_count = roi_count,
        .input_count = input_count_,
        .input_soa = input_soa.data(),
        .exposure_count = output_count_,
        .exposure_soa = output_soa.data(),
        .state_count = state_count_,
        .state_soa = state_soa.data(),
        .param_count = param_count_,
        .params_soa = params_soa.data(),
        .target_input_offsets = target_input_offsets.data(),
        .source_exposure_offsets = source_exposure_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
}

NeuralFieldRule::NeuralFieldRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      step_(reinterpret_cast<mind_sim::mod::NeuralFieldApplyFn>(
          library_->symbol("mind_neural_field_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mod::AbiRuleKind::NeuralField, "NeuralFieldRule");
    name_ = descriptor.name;
    input_count_ = descriptor.target_input_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    target_input_names_ = descriptor_names(descriptor.target_input_count, descriptor.target_input_names);
    source_exposure_names_ =
        descriptor_names(descriptor.source_exposure_count, descriptor.source_exposure_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
    local_state_names_ = descriptor_names(descriptor.local_state_count, descriptor.local_state_names);
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

const std::vector<std::string>& NeuralFieldRule::target_input_names() const noexcept {
    return target_input_names_;
}

const std::vector<std::string>& NeuralFieldRule::source_exposure_names() const noexcept {
    return source_exposure_names_;
}

const std::vector<std::string>& NeuralFieldRule::state_names() const noexcept {
    return state_names_;
}

const std::vector<double>& NeuralFieldRule::state_defaults() const noexcept {
    return state_defaults_;
}

const std::vector<std::string>& NeuralFieldRule::param_names() const noexcept {
    return param_names_;
}

const std::vector<double>& NeuralFieldRule::param_defaults() const noexcept {
    return param_defaults_;
}

const std::vector<std::string>& NeuralFieldRule::local_state_names() const noexcept {
    return local_state_names_;
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
                           const std::vector<int>& target_input_offsets,
                           double t,
                           double dt) const {
    const auto node_stride = static_cast<std::size_t>(node_count);
    for (int state: local_state_indices_) {
        const auto offset = static_cast<std::size_t>(state) * node_stride;
        std::copy_n(state_soa.begin() + static_cast<std::ptrdiff_t>(offset),
                    node_stride,
                    previous_state_soa.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    mind_sim::mod::AbiNeuralFieldContext context{
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
        .target_input_offsets = target_input_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
}

MacroToMacroRule::MacroToMacroRule(std::string name,
                           std::string library_path,
                           int input_count,
                           int output_count,
                           int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_macro_to_macro_rule_apply"))),
      input_count_(input_count),
      output_count_(output_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("MacroToMacroRule name must be non-empty");
    }
    validate_count(input_count_, "MacroToMacroRule input_count");
    validate_count(output_count_, "MacroToMacroRule output_count");
    validate_count(param_count_, "MacroToMacroRule param_count");
    if (output_count_ == 0) {
        throw std::runtime_error("MacroToMacroRule output_count must be positive");
    }
}

MacroToMacroRule::MacroToMacroRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mod::MacroToMacroApplyFn>(
          library_->symbol("mind_macro_to_macro_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mod::AbiRuleKind::MacroToMacro, "MacroToMacroRule");
    name_ = descriptor.name;
    input_count_ = descriptor.target_input_count;
    output_count_ = descriptor.source_exposure_count;
    param_count_ = descriptor.param_count;
    source_exposure_names_ =
        descriptor_names(descriptor.source_exposure_count, descriptor.source_exposure_names);
    target_input_names_ = descriptor_names(descriptor.target_input_count, descriptor.target_input_names);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
}

const std::string& MacroToMacroRule::name() const noexcept {
    return name_;
}

int MacroToMacroRule::input_count() const noexcept {
    return input_count_;
}

int MacroToMacroRule::output_count() const noexcept {
    return output_count_;
}

int MacroToMacroRule::param_count() const noexcept {
    return param_count_;
}

const std::string& MacroToMacroRule::library_path() const noexcept {
    return library_->path();
}

const std::vector<std::string>& MacroToMacroRule::source_exposure_names() const noexcept {
    return source_exposure_names_;
}

const std::vector<std::string>& MacroToMacroRule::target_input_names() const noexcept {
    return target_input_names_;
}

const std::vector<std::string>& MacroToMacroRule::param_names() const noexcept {
    return param_names_;
}

const std::vector<double>& MacroToMacroRule::param_defaults() const noexcept {
    return param_defaults_;
}

void MacroToMacroRule::validate_params(const std::vector<double>& params) const {
    validate_vector_size(params, param_count_, "MacroToMacroRule params");
}

void MacroToMacroRule::apply_flat(int roi_count,
                              int input_count,
                              int output_count,
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
                              const std::vector<int>& source_exposure_offsets,
                              const std::vector<int>& target_input_offsets) const {
    mind_sim::mod::AbiMacroToMacroContext context{
        .roi_count = roi_count,
        .input_count = input_count,
        .exposure_count = output_count,
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
        .source_exposure_offsets = source_exposure_offsets.data(),
        .target_input_offsets = target_input_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::macro::sim
