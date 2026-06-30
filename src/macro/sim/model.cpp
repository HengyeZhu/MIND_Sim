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

template <typename T>
int vector_count(const std::vector<T>& values) noexcept {
    return static_cast<int>(values.size());
}

}  // namespace

RegionRule::RegionRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto entry = mind_sim::mod::find_rule_entry(
        library_, mind_sim::mod::RuleKind::Region, rule_name, "RegionRule");
    if (entry->region_apply == nullptr) {
        throw std::runtime_error("RegionRule registry entry has null apply function");
    }
    library_ = entry->library;
    step_ = entry->region_apply;
    name_ = entry->name;
    exposure_names_ = entry->exposure_names;
    state_names_ = entry->state_names;
    state_defaults_ = entry->state_defaults;
    param_names_ = entry->param_names;
    param_defaults_ = entry->param_defaults;
    if (exposure_names_.empty()) {
        throw std::runtime_error("RegionRule exposure_count must be positive");
    }
}

const std::string& RegionRule::name() const noexcept {
    return name_;
}

int RegionRule::exposure_count() const noexcept {
    return vector_count(exposure_names_);
}

int RegionRule::state_count() const noexcept {
    return vector_count(state_names_);
}

int RegionRule::param_count() const noexcept {
    return vector_count(param_names_);
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
    validate_vector_size(state, state_count(), "RegionRule state");
}

void RegionRule::validate_params(const std::vector<double>& params) const {
    validate_vector_size(params, param_count(), "RegionRule params");
}

void RegionRule::step_group(const std::vector<int>& roi_indices,
                            int roi_count,
                            std::vector<double>& exposure_soa,
                            std::vector<double>& state_soa,
                            const std::vector<double>& params_soa,
                            const std::vector<int>& exposure_offsets,
                            double t,
                            double dt) const {
    mind_sim::mod::RegionContext context{
        .owner_count = static_cast<int>(roi_indices.size()),
        .roi_indices = roi_indices.data(),
        .roi_count = roi_count,
        .exposure_count = exposure_count(),
        .exposure_soa = exposure_soa.data(),
        .state_count = state_count(),
        .state_soa = state_soa.data(),
        .param_count = param_count(),
        .params_soa = params_soa.data(),
        .exposure_offsets = exposure_offsets.data(),
        .t = t,
        .dt = dt,
    };
    step_(&context);
}

MacroToMacroRule::MacroToMacroRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto entry = mind_sim::mod::find_rule_entry(
        library_, mind_sim::mod::RuleKind::MacroToMacro, rule_name, "MacroToMacroRule");
    if (entry->macro_to_macro_apply == nullptr) {
        throw std::runtime_error("MacroToMacroRule registry entry has null apply function");
    }
    library_ = entry->library;
    apply_ = entry->macro_to_macro_apply;
    name_ = entry->name;
    read_source_exposure_names_ = entry->read_source_exposure_names;
    read_target_exposure_names_ = entry->read_target_exposure_names;
    write_source_exposure_names_ = entry->write_source_exposure_names;
    write_target_exposure_names_ = entry->write_target_exposure_names;
    param_names_ = entry->param_names;
    param_defaults_ = entry->param_defaults;
    if (write_source_exposure_names_.empty() && write_target_exposure_names_.empty()) {
        throw std::runtime_error("MacroToMacroRule must write at least one exposure");
    }
}

const std::string& MacroToMacroRule::name() const noexcept {
    return name_;
}

int MacroToMacroRule::read_source_count() const noexcept {
    return vector_count(read_source_exposure_names_);
}

int MacroToMacroRule::read_target_count() const noexcept {
    return vector_count(read_target_exposure_names_);
}

int MacroToMacroRule::write_source_count() const noexcept {
    return vector_count(write_source_exposure_names_);
}

int MacroToMacroRule::write_target_count() const noexcept {
    return vector_count(write_target_exposure_names_);
}

int MacroToMacroRule::param_count() const noexcept {
    return vector_count(param_names_);
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
    validate_vector_size(params, param_count(), "MacroToMacroRule params");
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
    mind_sim::mod::MacroToMacroContext context{
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
        .param_count = param_count(),
        .params = params.data(),
        .read_source_offsets = read_source_offsets.data(),
        .read_target_offsets = read_target_offsets.data(),
        .write_source_offsets = write_source_offsets.data(),
        .write_target_offsets = write_target_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::macro::sim
