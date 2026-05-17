#include "macro/frontend/network.hpp"

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

}  // namespace

Network::Network(Connectivity connectivity,
                 std::vector<std::string> inputs,
                 std::vector<std::string> exposures,
                 std::vector<int> recorded_rois)
    : connectivity_(std::move(connectivity)),
      inputs_(std::move(inputs)),
      input_to_index_(build_name_index(inputs_, "input", false)),
      exposures_(std::move(exposures)),
      exposure_to_index_(build_name_index(exposures_, "exposure", true)) {
    initial_exposures_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(exposure_count())));
    dc_inputs_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(input_count())));
    owner_kind_.assign(static_cast<std::size_t>(roi_count()), OwnerKind::Empty);
    micro_circuit_by_roi_.assign(static_cast<std::size_t>(roi_count()), -1);
    micro_binding_by_roi_.assign(static_cast<std::size_t>(roi_count()), -1);
    recorded_rois_ = normalize_roi_indices(std::move(recorded_rois), "record ROI");
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

int Network::exposure_index(const std::string& exposure_name) const {
    const auto iter = exposure_to_index_.find(exposure_name);
    if (iter == exposure_to_index_.end()) {
        throw std::runtime_error("unknown exposure: " + exposure_name);
    }
    return iter->second;
}

int Network::exposure_count() const noexcept {
    return static_cast<int>(exposures_.size());
}

const std::vector<std::string>& Network::exposures() const noexcept {
    return exposures_;
}

const std::vector<int>& Network::recorded_rois() const noexcept {
    return recorded_rois_;
}

void Network::set_recorded_rois(std::vector<int> recorded_rois) {
    recorded_rois_ = normalize_roi_indices(std::move(recorded_rois), "record ROI");
}

void Network::set_initial_exposure(const ROI& roi_value,
                                   const mind_sim::macro::sim::ScalarBuffer& exposure) {
    validate_roi_index(roi_value.index, "initial exposure ROI");
    validate_buffer_size(exposure, exposure_count(), "initial exposure", "exposure");
    initial_exposures_[static_cast<std::size_t>(roi_value.index)] = exposure;
}

void Network::set_initial_exposure_value(const ROI& roi_value,
                                         const std::string& exposure_name,
                                         double value) {
    validate_roi_index(roi_value.index, "initial exposure ROI");
    set_buffer_value(initial_exposures_,
                     roi_value.index,
                     exposure_index(exposure_name),
                     value,
                     "initial exposure");
}

const std::vector<mind_sim::macro::sim::ScalarBuffer>& Network::initial_exposures() const noexcept {
    return initial_exposures_;
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

void Network::couple(const ROI& source_roi,
                     const ROI& target_roi,
                     std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule,
                     std::vector<double> params,
                     std::vector<int> read_exposure_offsets,
                     std::vector<int> write_input_offsets) {
    validate_roi_index(source_roi.index, "coupling source ROI");
    validate_roi_index(target_roi.index, "coupling target ROI");
    if (!rule) {
        throw std::runtime_error("Network coupling rule cannot be null");
    }
    rule->validate_params(params);
    if (read_exposure_offsets.size() != static_cast<std::size_t>(rule->exposure_count())) {
        throw std::runtime_error("CouplingRule READ offset count does not match rule");
    }
    if (write_input_offsets.size() != static_cast<std::size_t>(rule->input_count())) {
        throw std::runtime_error("CouplingRule WRITE offset count does not match rule");
    }
    coupling_projections_.push_back(CouplingProjection{
        .source_roi = source_roi.index,
        .target_roi = target_roi.index,
        .rule = std::move(rule),
        .params = std::move(params),
        .read_exposure_offsets = std::move(read_exposure_offsets),
        .write_input_offsets = std::move(write_input_offsets),
    });
}

void Network::use_region_rule(const ROI& roi_value,
                              std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                              std::vector<double> state,
                              std::vector<double> params,
                              std::vector<int> read_input_offsets,
                              std::vector<int> write_exposure_offsets) {
    validate_roi_index(roi_value.index, "model ROI");
    if (!rule) {
        throw std::runtime_error("Network.use_region_rule requires a RegionRule");
    }
    if (read_input_offsets.size() != static_cast<std::size_t>(rule->input_count())) {
        throw std::runtime_error("RegionRule READ offset count does not match rule");
    }
    if (write_exposure_offsets.size() != static_cast<std::size_t>(rule->exposure_count())) {
        throw std::runtime_error("RegionRule WRITE offset count does not match rule");
    }
    rule->validate_state(state);
    rule->validate_params(params);
    claim_roi(roi_value.index, OwnerKind::NeuralMass);
    region_owners_.push_back(RegionOwner{
        .roi_index = roi_value.index,
        .rule = std::move(rule),
        .state = std::move(state),
        .params = std::move(params),
        .read_input_offsets = std::move(read_input_offsets),
        .write_exposure_offsets = std::move(write_exposure_offsets),
    });
}

const std::vector<CouplingProjection>& Network::coupling_projections() const noexcept {
    return coupling_projections_;
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

}  // namespace mind_sim::macro::frontend
