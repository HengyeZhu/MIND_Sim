#include "macro/frontend/network.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace mind_sim::macro::frontend {

Connectivity::Connectivity(std::vector<std::string> labels,
                           std::vector<std::vector<float>> weights,
                           std::vector<std::vector<float>> delays)
    : labels_(std::move(labels)) {
    if (labels_.empty()) {
        throw std::runtime_error("Connectivity requires at least one ROI");
    }
    rois_.reserve(labels_.size());
    for (int index = 0; index < static_cast<int>(labels_.size()); ++index) {
        auto& label = labels_[static_cast<std::size_t>(index)];
        if (label.empty()) {
            throw std::runtime_error("ROI label must be non-empty");
        }
        const auto [_, inserted] = label_to_index_.emplace(label, index);
        if (!inserted) {
            throw std::runtime_error("ROI labels must be unique");
        }
        rois_.push_back(ROI{.index = index, .label = label});
    }
    weights_ = flatten_square_matrix(weights, "weights", false);
    delays_ = flatten_square_matrix(delays, "delays", true);
}

int Connectivity::roi_count() const noexcept {
    return static_cast<int>(rois_.size());
}

const std::vector<ROI>& Connectivity::rois() const noexcept {
    return rois_;
}

const std::vector<std::string>& Connectivity::labels() const noexcept {
    return labels_;
}

const std::vector<float>& Connectivity::weights() const noexcept {
    return weights_;
}

const std::vector<float>& Connectivity::delays() const noexcept {
    return delays_;
}

float Connectivity::weight_at(int target_roi, int source_roi) const {
    return weights_[matrix_offset(target_roi, source_roi)];
}

float Connectivity::delay_at(int target_roi, int source_roi) const {
    return delays_[matrix_offset(target_roi, source_roi)];
}

float Connectivity::min_positive_delay() const noexcept {
    float value = std::numeric_limits<float>::infinity();
    for (float delay: delays_) {
        if (delay > 0.0F && delay < value) {
            value = delay;
        }
    }
    return std::isfinite(value) ? value : 0.0F;
}

int Connectivity::roi_index(const std::string& label) const {
    const auto iter = label_to_index_.find(label);
    if (iter == label_to_index_.end()) {
        throw std::runtime_error("unknown ROI: " + label);
    }
    return iter->second;
}

std::vector<float> Connectivity::flatten_square_matrix(
    const std::vector<std::vector<float>>& matrix,
    const std::string& name,
    bool require_non_negative) const {
    const auto n = labels_.size();
    if (matrix.size() != n) {
        throw std::runtime_error(name + " row count must match ROI count");
    }
    std::vector<float> out;
    out.reserve(n * n);
    for (const auto& row: matrix) {
        if (row.size() != n) {
            throw std::runtime_error(name + " must be square");
        }
        for (float value: row) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(name + " must contain finite values");
            }
            if (require_non_negative && value < 0.0F) {
                throw std::runtime_error(name + " must be non-negative");
            }
            out.push_back(value);
        }
    }
    return out;
}

std::size_t Connectivity::matrix_offset(int target_roi, int source_roi) const {
    const int n = roi_count();
    if (target_roi < 0 || target_roi >= n || source_roi < 0 || source_roi >= n) {
        throw std::runtime_error("connectivity matrix index out of range");
    }
    return static_cast<std::size_t>(target_roi * n + source_roi);
}

Network::Network(Connectivity connectivity,
                 std::vector<std::string> inputs,
                 std::vector<std::string> exposures,
                 std::vector<int> recorded_rois)
    : connectivity_(std::move(connectivity)),
      inputs_(std::move(inputs)),
      exposures_(std::move(exposures)) {
    for (int index = 0; index < static_cast<int>(inputs_.size()); ++index) {
        const auto& input = inputs_[static_cast<std::size_t>(index)];
        if (input.empty()) {
            throw std::runtime_error("input name must be non-empty");
        }
        const auto [_, inserted] = input_to_index_.emplace(input, index);
        if (!inserted) {
            throw std::runtime_error("input names must be unique");
        }
    }
    if (exposures_.empty()) {
        throw std::runtime_error("Network requires at least one exposure");
    }
    for (int index = 0; index < static_cast<int>(exposures_.size()); ++index) {
        const auto& exposure = exposures_[static_cast<std::size_t>(index)];
        if (exposure.empty()) {
            throw std::runtime_error("exposure name must be non-empty");
        }
        const auto [_, inserted] = exposure_to_index_.emplace(exposure, index);
        if (!inserted) {
            throw std::runtime_error("exposure names must be unique");
        }
    }
    initial_exposures_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(exposure_count())));
    dc_inputs_.assign(
        static_cast<std::size_t>(roi_count()),
        mind_sim::macro::sim::ScalarBuffer(static_cast<std::size_t>(input_count())));
    owner_kind_.assign(static_cast<std::size_t>(roi_count()), OwnerKind::Empty);
    recorded_rois_ = normalize_roi_indices(std::move(recorded_rois), "record ROI");
}

Network::Network(std::vector<std::string> roi_labels,
                 std::vector<std::vector<float>> weights,
                 std::vector<std::vector<float>> delays,
                 std::vector<std::string> inputs,
                 std::vector<std::string> exposures,
                 std::vector<int> recorded_rois)
    : Network(Connectivity(std::move(roi_labels), std::move(weights), std::move(delays)),
              std::move(inputs),
              std::move(exposures),
              std::move(recorded_rois)) {}

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

float Network::weight_at(int target_roi, int source_roi) const {
    return connectivity_.weight_at(target_roi, source_roi);
}

float Network::delay_at(int target_roi, int source_roi) const {
    return connectivity_.delay_at(target_roi, source_roi);
}

float Network::min_positive_delay() const noexcept {
    return connectivity_.min_positive_delay();
}

const std::vector<float>& Network::weights_flat() const noexcept {
    return connectivity_.weights();
}

const std::vector<float>& Network::delays_flat() const noexcept {
    return connectivity_.delays();
}

void Network::set_initial_exposure(const ROI& roi_value,
                                   const mind_sim::macro::sim::ScalarBuffer& exposure) {
    validate_roi_index(roi_value.index, "initial exposure ROI");
    validate_exposure_size(exposure, "initial exposure");
    initial_exposures_[static_cast<std::size_t>(roi_value.index)] = exposure;
}

void Network::set_initial_exposure_value(const ROI& roi_value,
                                         const std::string& exposure_name,
                                         double value) {
    validate_roi_index(roi_value.index, "initial exposure ROI");
    if (!std::isfinite(value)) {
        throw std::runtime_error("initial exposure value must be finite");
    }
    initial_exposures_[static_cast<std::size_t>(roi_value.index)]
        .values[static_cast<std::size_t>(exposure_index(exposure_name))] = static_cast<float>(value);
}

const std::vector<mind_sim::macro::sim::ScalarBuffer>& Network::initial_exposures() const noexcept {
    return initial_exposures_;
}

void Network::set_dc_input(const ROI& roi_value,
                           const mind_sim::macro::sim::ScalarBuffer& input) {
    validate_roi_index(roi_value.index, "dc input ROI");
    validate_input_size(input, "dc input");
    dc_inputs_[static_cast<std::size_t>(roi_value.index)] = input;
}

void Network::set_dc_input_value(const ROI& roi_value,
                                 const std::string& input_name,
                                 double value) {
    validate_roi_index(roi_value.index, "dc input ROI");
    if (!std::isfinite(value)) {
        throw std::runtime_error("dc input value must be finite");
    }
    dc_inputs_[static_cast<std::size_t>(roi_value.index)]
        .values[static_cast<std::size_t>(input_index(input_name))] = static_cast<float>(value);
}

const std::vector<mind_sim::macro::sim::ScalarBuffer>& Network::dc_inputs() const noexcept {
    return dc_inputs_;
}

void Network::couple(std::vector<int> source_rois,
                     std::vector<int> target_rois,
                     std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule,
                     std::vector<double> params) {
    if (!rule) {
        throw std::runtime_error("Network coupling rule cannot be null");
    }
    validate_coupling_rule_schema(*rule);
    rule->validate_params(params);
    coupling_projections_.push_back(CouplingProjection{
        .source_rois = normalize_roi_indices(std::move(source_rois), "coupling source ROI"),
        .target_rois = normalize_roi_indices(std::move(target_rois), "coupling target ROI"),
        .rule = std::move(rule),
        .params = std::move(params),
    });
}

void Network::couple_all(std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule,
                         std::vector<double> params) {
    couple(all_roi_indices(), all_roi_indices(), std::move(rule), std::move(params));
}

void Network::couple_from(const ROI& source_roi,
                          std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule,
                          std::vector<double> params) {
    validate_roi_index(source_roi.index, "coupling source ROI");
    couple({source_roi.index}, all_roi_indices(), std::move(rule), std::move(params));
}

void Network::use_region_rule(const ROI& roi_value,
                              std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                              std::vector<double> state,
                              std::vector<double> params) {
    validate_roi_index(roi_value.index, "model ROI");
    if (!rule) {
        throw std::runtime_error("Network.use_region_rule requires a RegionRule");
    }
    validate_region_rule_schema(*rule);
    rule->validate_state(state);
    rule->validate_params(params);
    claim_roi(roi_value.index, OwnerKind::Region);
    region_owners_.push_back(RegionOwner{
        .roi_index = roi_value.index,
        .rule = std::move(rule),
        .state = std::move(state),
        .params = std::move(params),
    });
}

int Network::use_micro(std::string name,
                       std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data) {
    if (name.empty()) {
        throw std::runtime_error("Network.use_micro requires a non-empty micro name");
    }
    if (!core_data) {
        throw std::runtime_error("Network.use_micro requires CoreNeuronData");
    }
    for (const auto& circuit: micro_circuits_) {
        if (circuit.name == name) {
            throw std::runtime_error("micro name is already used: " + name);
        }
        if (circuit.core_data.get() == core_data.get()) {
            throw std::runtime_error("CoreNeuronData is already owned by a micro circuit");
        }
    }
    const int index = static_cast<int>(micro_circuits_.size());
    micro_circuits_.push_back(MicroCircuitOwner{
        .name = std::move(name),
        .core_data = std::move(core_data),
    });
    return index;
}

void Network::bind_micro_roi(int micro_circuit_index,
                             const ROI& roi_value,
                             std::vector<GidRange> gid_ranges,
                             std::shared_ptr<mind_sim::bridge::sim::MicroInputRule> input_rule,
                             std::vector<double> input_state,
                             std::vector<double> input_params,
                             std::vector<int> input_port_bases,
                             std::shared_ptr<mind_sim::bridge::sim::MicroOutputRule> output_rule,
                             std::vector<double> output_state,
                             std::vector<double> output_params) {
    validate_roi_index(roi_value.index, "micro ROI");
    validate_gid_ranges(gid_ranges);
    if (!input_rule) {
        throw std::runtime_error("Network.bind_micro_roi requires MicroInputRule");
    }
    if (!output_rule) {
        throw std::runtime_error("Network.bind_micro_roi requires MicroOutputRule");
    }
    validate_bridge_rule_schema(*input_rule, *output_rule);
    input_rule->validate_state(input_state);
    input_rule->validate_params(input_params);
    if (input_port_bases.size() != static_cast<std::size_t>(input_rule->input_port_count())) {
        throw std::runtime_error("MicroInputRule input port count does not match binding ports");
    }
    output_rule->validate_state(output_state);
    output_rule->validate_params(output_params);
    claim_roi(roi_value.index, OwnerKind::Micro);
    auto& circuit = require_micro_circuit(micro_circuit_index);
    circuit.bindings.push_back(MicroRoiBinding{
        .roi_index = roi_value.index,
        .gid_ranges = std::move(gid_ranges),
        .input_rule = std::move(input_rule),
        .input_state = std::move(input_state),
        .input_params = std::move(input_params),
        .input_port_bases = std::move(input_port_bases),
        .output_rule = std::move(output_rule),
        .output_state = std::move(output_state),
        .output_params = std::move(output_params),
    });
    rebuild_micro_gid_index(circuit);
}

const std::vector<CouplingProjection>& Network::coupling_projections() const noexcept {
    return coupling_projections_;
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

void Network::validate_exposure_size(const mind_sim::macro::sim::ScalarBuffer& exposure,
                                     const char* what) const {
    if (exposure.size() != static_cast<std::size_t>(exposure_count())) {
        throw std::runtime_error(std::string(what) + " exposure count mismatch");
    }
}

void Network::validate_input_size(const mind_sim::macro::sim::ScalarBuffer& input,
                                  const char* what) const {
    if (input.size() != static_cast<std::size_t>(input_count())) {
        throw std::runtime_error(std::string(what) + " input count mismatch");
    }
}

void Network::claim_roi(int roi_index_value, OwnerKind kind) {
    validate_roi_index(roi_index_value, "owner ROI");
    auto& owner = owner_kind_[static_cast<std::size_t>(roi_index_value)];
    if (owner != OwnerKind::Empty) {
        throw std::runtime_error("ROI already has an owner");
    }
    owner = kind;
}

void Network::validate_region_rule_schema(const mind_sim::macro::sim::RegionRule& rule) const {
    if (rule.input_count() != input_count()) {
        throw std::runtime_error("RegionRule input count does not match Network");
    }
    if (rule.exposure_count() != exposure_count()) {
        throw std::runtime_error("RegionRule exposure count does not match Network");
    }
}

void Network::validate_coupling_rule_schema(const mind_sim::macro::sim::CouplingRule& rule) const {
    if (rule.input_count() != input_count()) {
        throw std::runtime_error("CouplingRule input count does not match Network");
    }
    if (rule.exposure_count() != exposure_count()) {
        throw std::runtime_error("CouplingRule exposure count does not match Network");
    }
}

void Network::validate_bridge_rule_schema(const mind_sim::bridge::sim::MicroInputRule& input_rule,
                                          const mind_sim::bridge::sim::MicroOutputRule& output_rule) const {
    if (input_rule.input_count() != input_count()) {
        throw std::runtime_error("MicroInputRule input count does not match Network");
    }
    if (output_rule.exposure_count() != exposure_count()) {
        throw std::runtime_error("MicroOutputRule exposure count does not match Network");
    }
}

void Network::validate_gid_ranges(const std::vector<GidRange>& gid_ranges) const {
    if (gid_ranges.empty()) {
        throw std::runtime_error("micro ROI binding requires at least one gid range");
    }
    for (const auto& range: gid_ranges) {
        if (range.begin < 0 || range.end <= range.begin) {
            throw std::runtime_error("micro ROI gid ranges must be non-empty [begin, end) intervals");
        }
    }
}

MicroCircuitOwner& Network::require_micro_circuit(int micro_circuit_index) {
    if (micro_circuit_index < 0 ||
        micro_circuit_index >= static_cast<int>(micro_circuits_.size())) {
        throw std::runtime_error("micro circuit index out of range");
    }
    return micro_circuits_[static_cast<std::size_t>(micro_circuit_index)];
}

void Network::rebuild_micro_gid_index(MicroCircuitOwner& circuit) const {
    int gid_begin = 0;
    int gid_end = 0;
    bool first = true;
    for (const auto& binding: circuit.bindings) {
        for (const auto& range: binding.gid_ranges) {
            if (first) {
                gid_begin = range.begin;
                gid_end = range.end;
                first = false;
            } else {
                gid_begin = std::min(gid_begin, range.begin);
                gid_end = std::max(gid_end, range.end);
            }
        }
    }

    circuit.gid_begin = gid_begin;
    circuit.gid_end = gid_end;
    circuit.gid_to_binding.assign(static_cast<std::size_t>(gid_end - gid_begin), -1);
    for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
        const auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
        for (const auto& range: binding.gid_ranges) {
            for (int gid = range.begin; gid < range.end; ++gid) {
                auto& slot = circuit.gid_to_binding[static_cast<std::size_t>(gid - gid_begin)];
                if (slot >= 0) {
                    throw std::runtime_error("micro ROI gid ranges overlap inside one micro circuit");
                }
                slot = binding_index;
            }
        }
    }
}

std::vector<int> Network::all_roi_indices() const {
    std::vector<int> indices(static_cast<std::size_t>(roi_count()));
    for (int roi = 0; roi < roi_count(); ++roi) {
        indices[static_cast<std::size_t>(roi)] = roi;
    }
    return indices;
}

std::vector<int> Network::normalize_roi_indices(std::vector<int> roi_indices,
                                                const char* what) const {
    if (roi_indices.empty()) {
        throw std::runtime_error(std::string(what) + " selection must be non-empty");
    }
    std::unordered_set<int> seen;
    std::vector<int> normalized;
    normalized.reserve(roi_indices.size());
    for (int roi: roi_indices) {
        validate_roi_index(roi, what);
        if (seen.insert(roi).second) {
            normalized.push_back(roi);
        }
    }
    return normalized;
}

}  // namespace mind_sim::macro::frontend
