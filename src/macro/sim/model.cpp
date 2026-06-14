#include "macro/sim/model.hpp"

#include "mod/rule_registry.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::macro::sim {

namespace {

void validate_vector_size(const std::vector<double>& values, int expected, const char* what) {
    if (values.size() != static_cast<std::size_t>(expected)) {
        throw std::runtime_error(std::string(what) + " size mismatch");
    }
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

RegionRule::RegionRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto& entry = mind_sim::mod::find_rule_entry(
        *library_, mind_sim::mod::AbiRuleKind::Region, rule_name, "RegionRule");
    if (entry.region_apply == nullptr) {
        throw std::runtime_error("RegionRule registry entry has null apply function");
    }
    step_ = entry.region_apply;
    const auto& descriptor = *entry.descriptor;
    name_ = descriptor.name;
    exposure_count_ = descriptor.exposure_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    exposure_names_ = descriptor_names(descriptor.exposure_count, descriptor.exposure_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
    if (exposure_count_ == 0) {
        throw std::runtime_error("RegionRule exposure_count must be positive");
    }
}

const std::string& RegionRule::name() const noexcept {
    return name_;
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

const std::vector<std::string>& RegionRule::exposure_names() const noexcept {
    return exposure_names_;
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
                            std::vector<double>& exposure_soa,
                            std::vector<double>& state_soa,
                            const std::vector<double>& params_soa,
                            const std::vector<int>& exposure_offsets,
                            double t,
                            double dt) const {
    mind_sim::mod::AbiRegionContext context{
        .owner_count = static_cast<int>(roi_indices.size()),
        .roi_indices = roi_indices.data(),
        .roi_count = roi_count,
        .exposure_count = exposure_count_,
        .exposure_soa = exposure_soa.data(),
        .state_count = state_count_,
        .state_soa = state_soa.data(),
        .param_count = param_count_,
        .params_soa = params_soa.data(),
        .exposure_offsets = exposure_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
}

MacroToMacroRule::MacroToMacroRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto& entry = mind_sim::mod::find_rule_entry(
        *library_, mind_sim::mod::AbiRuleKind::MacroToMacro, rule_name, "MacroToMacroRule");
    if (entry.macro_to_macro_apply == nullptr) {
        throw std::runtime_error("MacroToMacroRule registry entry has null apply function");
    }
    apply_ = entry.macro_to_macro_apply;
    const auto& descriptor = *entry.descriptor;
    name_ = descriptor.name;
    read_source_count_ = descriptor.read_source_count;
    read_target_count_ = descriptor.read_target_count;
    write_source_count_ = descriptor.write_source_count;
    write_target_count_ = descriptor.write_target_count;
    param_count_ = descriptor.param_count;
    read_source_exposure_names_ =
        descriptor_names(descriptor.read_source_count, descriptor.read_source_exposure_names);
    read_target_exposure_names_ =
        descriptor_names(descriptor.read_target_count, descriptor.read_target_exposure_names);
    write_source_exposure_names_ =
        descriptor_names(descriptor.write_source_count, descriptor.write_source_exposure_names);
    write_target_exposure_names_ =
        descriptor_names(descriptor.write_target_count, descriptor.write_target_exposure_names);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
    if (write_source_count_ + write_target_count_ == 0) {
        throw std::runtime_error("MacroToMacroRule must write at least one exposure");
    }
}

const std::string& MacroToMacroRule::name() const noexcept {
    return name_;
}

int MacroToMacroRule::read_source_count() const noexcept {
    return read_source_count_;
}

int MacroToMacroRule::read_target_count() const noexcept {
    return read_target_count_;
}

int MacroToMacroRule::write_source_count() const noexcept {
    return write_source_count_;
}

int MacroToMacroRule::write_target_count() const noexcept {
    return write_target_count_;
}

int MacroToMacroRule::param_count() const noexcept {
    return param_count_;
}

const std::string& MacroToMacroRule::library_path() const noexcept {
    return library_->path();
}

const std::vector<std::string>& MacroToMacroRule::read_source_exposure_names() const noexcept {
    return read_source_exposure_names_;
}

const std::vector<std::string>& MacroToMacroRule::read_target_exposure_names() const noexcept {
    return read_target_exposure_names_;
}

const std::vector<std::string>& MacroToMacroRule::write_source_exposure_names() const noexcept {
    return write_source_exposure_names_;
}

const std::vector<std::string>& MacroToMacroRule::write_target_exposure_names() const noexcept {
    return write_target_exposure_names_;
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
                                  const std::vector<int>& write_target_offsets) const {
    mind_sim::mod::AbiMacroToMacroContext context{
        .roi_count = roi_count,
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
        .current_exposures = current_exposures.data(),
        .exposures = exposures.data(),
        .param_count = param_count_,
        .params = params.data(),
        .read_source_offsets = read_source_offsets.data(),
        .read_target_offsets = read_target_offsets.data(),
        .write_source_offsets = write_source_offsets.data(),
        .write_target_offsets = write_target_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::macro::sim
