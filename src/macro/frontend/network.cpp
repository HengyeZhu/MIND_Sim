#include "macro/frontend/network.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mind_sim::macro::frontend {

namespace {

std::unordered_map<std::string, int> build_name_index(const std::vector<std::string>& names,
                                                      const char* kind,
                                                      bool require_non_empty) {
    if (require_non_empty && names.empty()) {
        throw std::runtime_error(std::string("Network requires at least one ") + kind);
    }
    std::unordered_map<std::string, int> index_by_name;
    index_by_name.reserve(names.size());
    for (int index = 0; index < static_cast<int>(names.size()); ++index) {
        const auto& name = names[static_cast<std::size_t>(index)];
        if (name.empty()) {
            throw std::runtime_error(std::string(kind) + " name must be non-empty");
        }
        const auto [_, inserted] = index_by_name.emplace(name, index);
        if (!inserted) {
            throw std::runtime_error(std::string(kind) + " names must be unique");
        }
    }
    return index_by_name;
}

void set_buffer_value(std::vector<mind_sim::macro::sim::ScalarBuffer>& buffers,
                      int roi_index,
                      int field_index,
                      double value,
                      const char* what) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + " value must be finite");
    }
    buffers[static_cast<std::size_t>(roi_index)]
        .values[static_cast<std::size_t>(field_index)] = value;
}

void validate_buffer_size(const mind_sim::macro::sim::ScalarBuffer& buffer,
                          int expected_count,
                          const char* what,
                          const char* kind) {
    if (buffer.size() != static_cast<std::size_t>(expected_count)) {
        throw std::runtime_error(std::string(what) + " " + kind + " count mismatch");
    }
}

void require_positive_finite(double value, const char* what) {
    if (value <= 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + " must be positive and finite");
    }
}

}  // namespace

Network::Network(Connectivity connectivity,
                 std::vector<std::string> inputs,
                 std::vector<std::string> outputs,
                 std::vector<int> recorded_rois,
                 std::vector<int> recorded_outputs)
    : connectivity_(std::move(connectivity)),
      inputs_(std::move(inputs)),
      input_to_index_(build_name_index(inputs_, "input", false)),
      outputs_(std::move(outputs)),
      output_to_index_(build_name_index(outputs_, "output", true)) {
    output_history_start_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(output_count())));
    dc_inputs_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(input_count())));
    owner_kind_.assign(static_cast<std::size_t>(roi_count()), OwnerKind::Empty);
    micro_circuit_by_roi_.assign(static_cast<std::size_t>(roi_count()), -1);
    micro_binding_by_roi_.assign(static_cast<std::size_t>(roi_count()), -1);
    recorded_rois_ = normalize_roi_indices(std::move(recorded_rois), "record ROI");
    recorded_outputs_ = normalize_output_indices(std::move(recorded_outputs),
                                                     "record output");
}

ROI Network::roi(int roi_index_value) const {
    validate_roi_index(roi_index_value, "ROI");
    return connectivity_.rois()[static_cast<std::size_t>(roi_index_value)];
}

ROI Network::roi(const std::string& label) const {
    return roi(roi_index(label));
}

const std::vector<ROI>& Network::rois() const noexcept {
    return connectivity_.rois();
}

int Network::roi_index(const std::string& label) const {
    return connectivity_.roi_index(label);
}

int Network::roi_count() const noexcept {
    return connectivity_.roi_count();
}

const Connectivity& Network::connectivity() const noexcept {
    return connectivity_;
}

void Network::set_dt(double dt_value) {
    require_positive_finite(dt_value, "macro dt");
    dt_ = dt_value;
}

double Network::dt() const noexcept {
    return dt_;
}

void Network::set_exchange_window(double exchange_window_value) {
    require_positive_finite(exchange_window_value, "exchange_window");
    exchange_window_ = exchange_window_value;
}

double Network::exchange_window() const noexcept {
    return exchange_window_;
}

int Network::input_index(const std::string& input_name) const {
    const auto iter = input_to_index_.find(input_name);
    if (iter == input_to_index_.end()) {
        throw std::runtime_error("unknown input: " + input_name);
    }
    return iter->second;
}

int Network::input_count() const noexcept {
    return static_cast<int>(inputs_.size());
}

const std::vector<std::string>& Network::inputs() const noexcept {
    return inputs_;
}

int Network::output_index(const std::string& output_name) const {
    const auto iter = output_to_index_.find(output_name);
    if (iter == output_to_index_.end()) {
        throw std::runtime_error("unknown output: " + output_name);
    }
    return iter->second;
}

int Network::output_count() const noexcept {
    return static_cast<int>(outputs_.size());
}

const std::vector<std::string>& Network::outputs() const noexcept {
    return outputs_;
}

const std::vector<int>& Network::recorded_rois() const noexcept {
    return recorded_rois_;
}

const std::vector<int>& Network::recorded_outputs() const noexcept {
    return recorded_outputs_;
}

void Network::set_recorded_rois(std::vector<int> recorded_rois) {
    recorded_rois_ = normalize_roi_indices(std::move(recorded_rois), "record ROI");
}

void Network::set_recorded_outputs(std::vector<int> recorded_outputs) {
    recorded_outputs_ = normalize_output_indices(std::move(recorded_outputs),
                                                     "record output");
}

void Network::record_micro(double* value, std::vector<double>* samples) {
    if (value == nullptr || samples == nullptr) {
        throw std::runtime_error("record_micro requires non-null variable and sample buffer pointers");
    }
    micro_record_targets_.push_back(MicroTraceRecorder{
        .value = value,
        .samples = samples,
    });
}

void Network::record_micro_time(std::vector<double>* samples) {
    if (samples == nullptr) {
        throw std::runtime_error("record_micro_time requires a non-null sample buffer pointer");
    }
    micro_time_record_targets_.push_back(samples);
}

const std::vector<MicroTraceRecorder>& Network::micro_record_targets() const noexcept {
    return micro_record_targets_;
}

const std::vector<std::vector<double>*>& Network::micro_time_record_targets() const noexcept {
    return micro_time_record_targets_;
}

const std::vector<mind_sim::macro::sim::ScalarBuffer>& Network::output_history_start() const noexcept {
    return output_history_start_;
}

void Network::set_dc_input(const ROI& roi_value,
                           const mind_sim::macro::sim::ScalarBuffer& input) {
    validate_roi_index(roi_value.index, "dc input ROI");
    validate_buffer_size(input, input_count(), "dc input", "input");
    dc_inputs_[static_cast<std::size_t>(roi_value.index)] = input;
}

void Network::set_dc_input_value(const ROI& roi_value,
                                 const std::string& input_name,
                                 double value) {
    validate_roi_index(roi_value.index, "dc input ROI");
    set_buffer_value(dc_inputs_,
                     roi_value.index,
                     input_index(input_name),
                     value,
                     "dc input");
}

const std::vector<mind_sim::macro::sim::ScalarBuffer>& Network::dc_inputs() const noexcept {
    return dc_inputs_;
}

void Network::macro_to_macro(const ROI& source_roi,
                     const ROI& target_roi,
                     std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule,
                     std::vector<double> params,
                     std::vector<int> source_exposure_offsets,
                     std::vector<int> target_input_offsets) {
    validate_roi_index(source_roi.index, "macro-to-macro source ROI");
    validate_roi_index(target_roi.index, "macro-to-macro target ROI");
    if (!rule) {
        throw std::runtime_error("Network macro-to-macro rule cannot be null");
    }
    rule->validate_params(params);
    if (source_exposure_offsets.size() != static_cast<std::size_t>(rule->output_count())) {
        throw std::runtime_error("MacroToMacroRule source output offset count does not match rule");
    }
    if (target_input_offsets.size() != static_cast<std::size_t>(rule->input_count())) {
        throw std::runtime_error("MacroToMacroRule target input offset count does not match rule");
    }
    macro_to_macro_projections_.push_back(MacroToMacroProjection{
        .source_roi = source_roi.index,
        .target_roi = target_roi.index,
        .rule = std::move(rule),
        .params = std::move(params),
        .source_exposure_offsets = std::move(source_exposure_offsets),
        .target_input_offsets = std::move(target_input_offsets),
    });
}

void Network::use_region_rule(const ROI& roi_value,
                              std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                              std::vector<double> state,
                              std::vector<double> params,
                              std::vector<int> target_input_offsets,
                              std::vector<int> source_exposure_offsets) {
    validate_roi_index(roi_value.index, "model ROI");
    if (!rule) {
        throw std::runtime_error("Network.use_region_rule requires a RegionRule");
    }
    if (target_input_offsets.size() != static_cast<std::size_t>(rule->input_count())) {
        throw std::runtime_error("RegionRule target input offset count does not match rule");
    }
    if (source_exposure_offsets.size() != static_cast<std::size_t>(rule->output_count())) {
        throw std::runtime_error("RegionRule source output offset count does not match rule");
    }
    rule->validate_state(state);
    rule->validate_params(params);
    claim_roi(roi_value.index, OwnerKind::NeuralMass);

    const auto& exposure_names = rule->source_exposure_names();
    const auto& state_names = rule->state_names();
    for (int exposure = 0; exposure < static_cast<int>(exposure_names.size()); ++exposure) {
        const auto& exposure_name = exposure_names[static_cast<std::size_t>(exposure)];
        const auto state_found = std::find(state_names.begin(), state_names.end(), exposure_name);
        if (state_found == state_names.end()) {
            continue;
        }
        const int state_index = static_cast<int>(state_found - state_names.begin());
        output_history_start_[static_cast<std::size_t>(roi_value.index)]
            .values[static_cast<std::size_t>(output_index(exposure_name))] =
            state[static_cast<std::size_t>(state_index)];
    }

    region_owners_.push_back(RegionOwner{
        .roi_index = roi_value.index,
        .rule = std::move(rule),
        .state = std::move(state),
        .params = std::move(params),
        .target_input_offsets = std::move(target_input_offsets),
        .source_exposure_offsets = std::move(source_exposure_offsets),
    });
}

const std::vector<MacroToMacroProjection>& Network::macro_to_macro_projections() const noexcept {
    return macro_to_macro_projections_;
}

const std::vector<RegionOwner>& Network::region_owners() const noexcept {
    return region_owners_;
}

const std::vector<NeuralFieldOwner>& Network::neural_field_owners() const noexcept {
    return neural_field_owners_;
}

const std::vector<MicroCircuitOwner>& Network::micro_circuits() const noexcept {
    return micro_circuits_;
}

void Network::validate_roi_index(int roi_index_value, const char* what) const {
    if (roi_index_value < 0 || roi_index_value >= roi_count()) {
        throw std::runtime_error(std::string(what) + " index out of range");
    }
}

void Network::claim_roi(int roi_index_value, OwnerKind kind) {
    auto& owner = owner_kind_[static_cast<std::size_t>(roi_index_value)];
    if (owner != OwnerKind::Empty) {
        throw std::runtime_error("ROI already has an owner");
    }
    owner = kind;
}

std::vector<int> Network::normalize_roi_indices(std::vector<int> roi_indices,
                                                const char* what) const {
    if (roi_indices.empty()) {
        throw std::runtime_error(std::string(what) + " selection must be non-empty");
    }
    std::unordered_set<int> seen;
    std::vector<int> normalized;
    for (int roi: roi_indices) {
        validate_roi_index(roi, what);
        if (seen.insert(roi).second) {
            normalized.push_back(roi);
        }
    }
    return normalized;
}

std::vector<int> Network::normalize_output_indices(std::vector<int> output_indices,
                                                     const char* what) const {
    if (output_indices.empty()) {
        throw std::runtime_error(std::string(what) + " selection must be non-empty");
    }
    std::unordered_set<int> seen;
    std::vector<int> normalized;
    for (int output: output_indices) {
        if (output < 0 || output >= output_count()) {
            throw std::runtime_error(std::string(what) + " index out of range");
        }
        if (seen.insert(output).second) {
            normalized.push_back(output);
        }
    }
    return normalized;
}

}  // namespace mind_sim::macro::frontend
