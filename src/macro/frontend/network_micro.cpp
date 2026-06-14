#include "macro/frontend/network.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::macro::frontend {

int Network::use_micro(std::string name,
                       std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data) {
    if (name.empty()) {
        throw std::runtime_error("Network.use_micro requires a non-empty micro name");
    }
    if (!core_data) {
        throw std::runtime_error("Network.use_micro requires CoreNeuronData");
    }
    if (!micro_circuits_.empty()) {
        if (micro_circuits_.front().core_data.get() == core_data.get()) {
            return 0;
        }
        throw std::runtime_error(
            "Network supports one CoreNEURON micro circuit; bind multiple ROI ranges to it");
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
                             std::vector<GidRange> gid_ranges) {
    validate_roi_index(roi_value.index, "micro ROI");
    validate_gid_ranges(gid_ranges);
    auto& circuit = require_micro_circuit(micro_circuit_index);
    claim_roi(roi_value.index, OwnerKind::Micro);
    const int binding_index = static_cast<int>(circuit.bindings.size());
    circuit.bindings.push_back(MicroRoiBinding{
        .roi_index = roi_value.index,
        .gid_ranges = std::move(gid_ranges),
    });
    micro_circuit_by_roi_[static_cast<std::size_t>(roi_value.index)] = micro_circuit_index;
    micro_binding_by_roi_[static_cast<std::size_t>(roi_value.index)] = binding_index;
    rebuild_micro_gid_bounds(circuit);
}

void Network::configure_macro_to_micro_rule(
    const ROI& roi_value,
    std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> input_rule,
    std::vector<double> input_state,
    std::vector<double> input_params,
    std::vector<int> macro2micro_indices,
    std::vector<int> macro2micro_source_ids,
    std::vector<int> exposure_offsets) {
    if (!input_rule) {
        throw std::runtime_error("micro input connection requires MicroInputRule");
    }
    if (macro2micro_indices.empty()) {
        throw std::runtime_error("micro input connection requires at least one macro2micro id");
    }
    if (macro2micro_source_ids.size() != macro2micro_indices.size()) {
        throw std::runtime_error("macro2micro source id count must match runtime index count");
    }
    for (int index: macro2micro_indices) {
        if (index < 0) {
            throw std::runtime_error("macro2micro indices must be non-negative");
        }
    }
    for (int source_id: macro2micro_source_ids) {
        if (source_id < 0) {
            throw std::runtime_error("macro2micro source ids must be non-negative");
        }
    }
    input_rule->validate_state(input_state, static_cast<int>(macro2micro_indices.size()));
    input_rule->validate_params(input_params);
    if (exposure_offsets.size() != static_cast<std::size_t>(input_rule->exposure_count())) {
        throw std::runtime_error("MicroInputRule exposure offset count does not match rule");
    }
    auto& binding = require_micro_binding(roi_value.index);
    binding.input_rule = std::move(input_rule);
    binding.input_state = std::move(input_state);
    binding.input_params = std::move(input_params);
    binding.macro2micro_indices = std::move(macro2micro_indices);
    binding.macro2micro_source_ids = std::move(macro2micro_source_ids);
    binding.exposure_offsets = std::move(exposure_offsets);
}

void Network::configure_micro_output_rule(
    const ROI& roi_value,
    std::shared_ptr<mind_sim::cosim::transform::MicroOutputRule> output_rule,
    std::vector<double> output_state,
    std::vector<double> output_params,
    std::vector<int> output_source_sids,
    std::vector<int> exposure_offsets) {
    if (!output_rule) {
        throw std::runtime_error("micro output connection requires MicroOutputRule");
    }
    output_rule->validate_state(output_state);
    output_rule->validate_params(output_params);
    if (exposure_offsets.size() != static_cast<std::size_t>(output_rule->output_count())) {
        throw std::runtime_error("MicroOutputRule exposure offset count does not match rule");
    }
    std::sort(output_source_sids.begin(), output_source_sids.end());
    output_source_sids.erase(std::unique(output_source_sids.begin(), output_source_sids.end()),
                             output_source_sids.end());
    for (int sid: output_source_sids) {
        if (sid < 0) {
            throw std::runtime_error("micro output source sid must be non-negative");
        }
    }
    auto& binding = require_micro_binding(roi_value.index);
    binding.output_transforms.push_back(MicroOutputTransform{
        .rule = std::move(output_rule),
        .state = std::move(output_state),
        .params = std::move(output_params),
        .source_sids = std::move(output_source_sids),
        .exposure_offsets = std::move(exposure_offsets),
    });
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

MicroRoiBinding& Network::require_micro_binding(int roi_index_value) {
    validate_roi_index(roi_index_value, "micro binding ROI");
    const auto roi = static_cast<std::size_t>(roi_index_value);
    const int circuit_index = micro_circuit_by_roi_[roi];
    if (circuit_index < 0) {
        throw std::runtime_error("ROI is not bound to a micro circuit");
    }
    return micro_circuits_[static_cast<std::size_t>(circuit_index)]
        .bindings[static_cast<std::size_t>(micro_binding_by_roi_[roi])];
}

void Network::rebuild_micro_gid_bounds(MicroCircuitOwner& circuit) const {
    int gid_begin = circuit.bindings.front().gid_ranges.front().begin;
    int gid_end = circuit.bindings.front().gid_ranges.front().end;
    for (const auto& binding: circuit.bindings) {
        for (const auto& range: binding.gid_ranges) {
            gid_begin = std::min(gid_begin, range.begin);
            gid_end = std::max(gid_end, range.end);
        }
    }

    circuit.gid_begin = gid_begin;
    circuit.gid_end = gid_end;
}

}  // namespace mind_sim::macro::frontend
