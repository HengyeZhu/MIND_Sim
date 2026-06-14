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

void validate_history_axis(int value, int expected, const char* name, const char* layout) {
    if (value != expected) {
        throw std::runtime_error(std::string("initial_history ") + name + " axis mismatch for " +
                                 layout + ": got " + std::to_string(value) +
                                 ", expected " + std::to_string(expected));
    }
}

void require_positive_finite(double value, const char* what) {
    if (value <= 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + " must be positive and finite");
    }
}

}  // namespace

Network::Network(Connectivity connectivity,
                 std::vector<std::string> exposures,
                 std::vector<int> recorded_rois,
                 std::vector<int> recorded_outputs)
    : connectivity_(std::move(connectivity)),
      outputs_(std::move(exposures)),
      output_to_index_(build_name_index(outputs_, "output", true)) {
    output_history_start_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(output_count())));
    region_owner_seen_.assign(static_cast<std::size_t>(roi_count()), 0);
    micro_owner_seen_.assign(static_cast<std::size_t>(roi_count()), 0);
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

void Network::set_initial_history(const std::vector<std::string>& output_names,
                                  int time_count,
                                  int axis1_count,
                                  int axis2_count,
                                  const std::vector<double>& values,
                                  InitialHistoryLayout layout) {
    if (time_count <= 0) {
        throw std::runtime_error("initial_history time axis must be positive");
    }
    if (axis1_count <= 0 || axis2_count <= 0) {
        throw std::runtime_error("initial_history spatial/output axes must be positive");
    }
    const char* layout_name =
        layout == InitialHistoryLayout::TimeOutputRoi ? "time_output_roi" : "time_roi_output";
    const int provided_output_count =
        output_names.empty() ? output_count() : static_cast<int>(output_names.size());
    if (provided_output_count <= 0) {
        throw std::runtime_error("initial_history requires at least one output");
    }
    if (layout == InitialHistoryLayout::TimeOutputRoi) {
        validate_history_axis(axis1_count, provided_output_count, "output", layout_name);
        validate_history_axis(axis2_count, roi_count(), "roi", layout_name);
    } else {
        validate_history_axis(axis1_count, roi_count(), "roi", layout_name);
        validate_history_axis(axis2_count, provided_output_count, "output", layout_name);
    }
    const auto expected_size =
        static_cast<std::size_t>(time_count) *
        static_cast<std::size_t>(axis1_count) *
        static_cast<std::size_t>(axis2_count);
    if (values.size() != expected_size) {
        throw std::runtime_error("initial_history value count does not match its shape");
    }

    std::vector<int> output_indices;
    output_indices.reserve(static_cast<std::size_t>(provided_output_count));
    if (output_names.empty()) {
        for (int output = 0; output < output_count(); ++output) {
            output_indices.push_back(output);
        }
    } else {
        std::vector<unsigned char> seen_outputs(static_cast<std::size_t>(output_count()), 0);
        for (const auto& name: output_names) {
            const int output = output_index(name);
            if (seen_outputs[static_cast<std::size_t>(output)] != 0) {
                throw std::runtime_error("initial_history output names must be unique");
            }
            seen_outputs[static_cast<std::size_t>(output)] = 1;
            output_indices.push_back(output);
        }
    }

    const auto slot_size = static_cast<std::size_t>(roi_count() * output_count());
    std::vector<double> canonical(static_cast<std::size_t>(time_count) * slot_size, 0.0);
    const auto start_soa = [&]() {
        std::vector<double> soa(slot_size, 0.0);
        for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
            const auto& buffer = output_history_start_[static_cast<std::size_t>(roi_value)];
            for (int output = 0; output < output_count(); ++output) {
                soa[static_cast<std::size_t>(output * roi_count() + roi_value)] =
                    buffer.values[static_cast<std::size_t>(output)];
            }
        }
        return soa;
    }();
    for (int time = 0; time < time_count; ++time) {
        std::copy(start_soa.begin(),
                  start_soa.end(),
                  canonical.begin() + static_cast<std::ptrdiff_t>(
                                        static_cast<std::size_t>(time) * slot_size));
    }

    for (int time = 0; time < time_count; ++time) {
        for (int provided_output = 0; provided_output < provided_output_count; ++provided_output) {
            const int output = output_indices[static_cast<std::size_t>(provided_output)];
            for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
                std::size_t source_offset = 0;
                if (layout == InitialHistoryLayout::TimeOutputRoi) {
                    source_offset =
                        (static_cast<std::size_t>(time) * static_cast<std::size_t>(axis1_count) +
                         static_cast<std::size_t>(provided_output)) *
                            static_cast<std::size_t>(axis2_count) +
                        static_cast<std::size_t>(roi_value);
                } else {
                    source_offset =
                        (static_cast<std::size_t>(time) * static_cast<std::size_t>(axis1_count) +
                         static_cast<std::size_t>(roi_value)) *
                            static_cast<std::size_t>(axis2_count) +
                        static_cast<std::size_t>(provided_output);
                }
                const double value = values[source_offset];
                if (!std::isfinite(value)) {
                    throw std::runtime_error("initial_history values must be finite");
                }
                canonical[static_cast<std::size_t>(time) * slot_size +
                          static_cast<std::size_t>(output * roi_count() + roi_value)] = value;
            }
        }
    }

    initial_history_time_count_ = time_count;
    initial_history_ = std::move(canonical);

    const auto current_offset = static_cast<std::size_t>(time_count - 1) * slot_size;
    for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
        auto& buffer = output_history_start_[static_cast<std::size_t>(roi_value)];
        for (int output = 0; output < output_count(); ++output) {
            buffer.values[static_cast<std::size_t>(output)] =
                initial_history_[current_offset +
                                 static_cast<std::size_t>(output * roi_count() + roi_value)];
        }
    }

    for (auto& owner: region_owners_) {
        const auto& exposure_names = owner.rule->exposure_names();
        const auto& state_names = owner.rule->state_names();
        for (const auto& exposure_name: exposure_names) {
            const auto state_found =
                std::find(state_names.begin(), state_names.end(), exposure_name);
            if (state_found == state_names.end()) {
                continue;
            }
            const int state_index = static_cast<int>(state_found - state_names.begin());
            const int output = output_index(exposure_name);
            owner.state[static_cast<std::size_t>(state_index)] =
                initial_history_[current_offset +
                                 static_cast<std::size_t>(output * roi_count() + owner.roi_index)];
        }
    }

}

bool Network::has_initial_history() const noexcept {
    return initial_history_time_count_ > 0;
}

int Network::initial_history_time_count() const noexcept {
    return initial_history_time_count_;
}

const std::vector<double>& Network::initial_history() const noexcept {
    return initial_history_;
}

void Network::macro_to_macro(const ROI& source_roi,
	                     const ROI& target_roi,
	                     std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule,
	                     std::vector<double> params,
	                     std::vector<int> read_source_offsets,
	                     std::vector<int> read_target_offsets,
	                     std::vector<int> write_source_offsets,
	                     std::vector<int> write_target_offsets) {
    validate_roi_index(source_roi.index, "macro-to-macro source ROI");
    validate_roi_index(target_roi.index, "macro-to-macro target ROI");
    if (!rule) {
        throw std::runtime_error("Network macro-to-macro rule cannot be null");
    }
    rule->validate_params(params);
    if (read_source_offsets.size() != static_cast<std::size_t>(rule->read_source_count())) {
        throw std::runtime_error("MacroToMacroRule READ_SOURCE offset count does not match rule");
    }
    if (read_target_offsets.size() != static_cast<std::size_t>(rule->read_target_count())) {
        throw std::runtime_error("MacroToMacroRule READ_TARGET offset count does not match rule");
    }
    if (write_source_offsets.size() != static_cast<std::size_t>(rule->write_source_count())) {
        throw std::runtime_error("MacroToMacroRule WRITE_SOURCE offset count does not match rule");
    }
    if (write_target_offsets.size() != static_cast<std::size_t>(rule->write_target_count())) {
        throw std::runtime_error("MacroToMacroRule WRITE_TARGET offset count does not match rule");
    }
    macro_to_macro_projections_.push_back(MacroToMacroProjection{
        .source_roi = source_roi.index,
        .target_roi = target_roi.index,
        .rule = std::move(rule),
        .params = std::move(params),
        .read_source_offsets = std::move(read_source_offsets),
        .read_target_offsets = std::move(read_target_offsets),
        .write_source_offsets = std::move(write_source_offsets),
        .write_target_offsets = std::move(write_target_offsets),
    });
}

void Network::use_region_rule(const ROI& roi_value,
                              std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                              std::vector<double> state,
                              std::vector<double> params,
                              std::vector<int> exposure_offsets) {
    validate_roi_index(roi_value.index, "model ROI");
    if (!rule) {
        throw std::runtime_error("Network.use_region_rule requires a RegionRule");
    }
    if (exposure_offsets.size() != static_cast<std::size_t>(rule->exposure_count())) {
        throw std::runtime_error("RegionRule exposure offset count does not match rule");
    }
    rule->validate_state(state);
    rule->validate_params(params);
    claim_roi(roi_value.index, OwnerKind::NeuralMass);

    const auto& exposure_names = rule->exposure_names();
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
        .exposure_offsets = std::move(exposure_offsets),
    });
}

const std::vector<MacroToMacroProjection>& Network::macro_to_macro_projections() const noexcept {
    return macro_to_macro_projections_;
}

const std::vector<RegionOwner>& Network::region_owners() const noexcept {
    return region_owners_;
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
    const auto index = static_cast<std::size_t>(roi_index_value);
    switch (kind) {
    case OwnerKind::NeuralMass:
        if (region_owner_seen_[index] != 0) {
            throw std::runtime_error("ROI already has a macro owner");
        }
        region_owner_seen_[index] = 1;
        return;
    case OwnerKind::Micro:
        if (micro_owner_seen_[index] != 0) {
            throw std::runtime_error("ROI already has a micro-incompatible owner");
        }
        micro_owner_seen_[index] = 1;
        return;
    case OwnerKind::Empty:
        break;
    }
    throw std::runtime_error("invalid ROI owner kind");
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
