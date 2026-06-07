#include "macro/frontend/network.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::macro::frontend {

void Network::use_neural_field(
    std::string name,
    std::shared_ptr<mind_sim::macro::sim::NeuralFieldRule> rule,
    NodeToRoiMap node_map,
    LocalConnectivity local_connectivity,
    std::vector<double> state_soa,
    std::vector<double> params,
    std::vector<int> target_input_offsets,
    std::vector<FieldOutputReducer> reducers) {
    if (name.empty()) {
        throw std::runtime_error("Network.use_neural_field requires a non-empty field name");
    }
    if (!rule) {
        throw std::runtime_error("Network.use_neural_field requires a NeuralFieldRule");
    }
    for (const auto& owner: neural_field_owners_) {
        if (owner.name == name) {
            throw std::runtime_error("neural field name is already used: " + name);
        }
    }
    if (target_input_offsets.size() != static_cast<std::size_t>(rule->input_count())) {
        throw std::runtime_error("NeuralFieldRule target input offset count does not match rule");
    }
    const auto& node_to_roi = node_map.node_to_roi();
    const auto& node_weights = node_map.node_weights();
    const int node_count = node_map.node_count();
    if (local_connectivity.node_count() != node_count) {
        throw std::runtime_error("neural field LocalConnectivity node_count must match node_map");
    }
    rule->validate_state(state_soa, node_count);
    rule->validate_params(params);
    auto mapping = build_field_mapping(node_to_roi, node_weights);
    validate_field_local_connectivity(local_connectivity.indptr(),
                                      local_connectivity.indices(),
                                      local_connectivity.weights(),
                                      node_count);
    auto reducer_plan = build_field_reducers(std::move(reducers), *rule, node_count);
    auto previous_state_soa = state_soa;
    for (int roi: mapping.owned_rois) {
        claim_roi(roi, OwnerKind::Field);
    }
    neural_field_owners_.push_back(NeuralFieldOwner{
        .name = std::move(name),
        .rule = std::move(rule),
        .node_count = node_count,
        .node_to_roi = node_to_roi,
        .roi_node_offsets = std::move(mapping.roi_node_offsets),
        .roi_nodes = std::move(mapping.roi_nodes),
        .roi_node_weights = std::move(mapping.roi_node_weights),
        .local_indptr = local_connectivity.indptr(),
        .local_indices = local_connectivity.indices(),
        .local_weights = local_connectivity.weights(),
        .state_soa = std::move(state_soa),
        .previous_state_soa = std::move(previous_state_soa),
        .params = std::move(params),
        .target_input_offsets = std::move(target_input_offsets),
        .reducers = std::move(reducer_plan),
        .owned_rois = std::move(mapping.owned_rois),
    });
}

Network::FieldMappingInfo Network::build_field_mapping(
    const std::vector<int>& node_to_roi,
    const std::vector<double>& node_weights) const {
    if (node_weights.size() != node_to_roi.size()) {
        throw std::runtime_error("neural field node_weights size must match node_to_roi");
    }
    const int roi_total = roi_count();
    std::vector<int> roi_node_counts(static_cast<std::size_t>(roi_total), 0);
    std::vector<double> roi_weight_sum(static_cast<std::size_t>(roi_total), 0.0);
    FieldMappingInfo info;
    for (std::size_t node = 0; node < node_to_roi.size(); ++node) {
        const int roi = node_to_roi[node];
        validate_roi_index(roi, "neural field node ROI");
        const double weight = node_weights[node];
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::runtime_error("neural field node weights must be finite and non-negative");
        }
        roi_node_counts[static_cast<std::size_t>(roi)] += 1;
        roi_weight_sum[static_cast<std::size_t>(roi)] += weight;
    }

    int owned_roi_count = 0;
    for (int roi = 0; roi < roi_total; ++roi) {
        if (roi_node_counts[static_cast<std::size_t>(roi)] == 0) {
            continue;
        }
        if (roi_weight_sum[static_cast<std::size_t>(roi)] <= 0.0) {
            throw std::runtime_error("neural field ROI has zero total node weight");
        }
        owned_roi_count += 1;
    }

    info.owned_rois.reserve(static_cast<std::size_t>(owned_roi_count));
    info.roi_node_offsets.reserve(static_cast<std::size_t>(owned_roi_count + 1));
    info.roi_node_offsets.push_back(0);
    info.roi_nodes.resize(node_to_roi.size());
    info.roi_node_weights.resize(node_to_roi.size());
    std::vector<int> owner_position_by_roi(static_cast<std::size_t>(roi_total), -1);

    for (int roi = 0; roi < roi_total; ++roi) {
        const int node_count_for_roi = roi_node_counts[static_cast<std::size_t>(roi)];
        if (node_count_for_roi == 0) {
            continue;
        }
        owner_position_by_roi[static_cast<std::size_t>(roi)] =
            static_cast<int>(info.owned_rois.size());
        info.owned_rois.push_back(roi);
        info.roi_node_offsets.push_back(info.roi_node_offsets.back() + node_count_for_roi);
    }

    auto fill_positions = info.roi_node_offsets;
    for (std::size_t node = 0; node < node_to_roi.size(); ++node) {
        const int roi = node_to_roi[node];
        const int position = owner_position_by_roi[static_cast<std::size_t>(roi)];
        const auto slot = static_cast<std::size_t>(
            fill_positions[static_cast<std::size_t>(position)]++);
        info.roi_nodes[slot] = static_cast<int>(node);
        info.roi_node_weights[slot] =
            node_weights[node] / roi_weight_sum[static_cast<std::size_t>(roi)];
    }
    return info;
}

void Network::validate_field_local_connectivity(const std::vector<int>& local_indptr,
                                                const std::vector<int>& local_indices,
                                                const std::vector<double>& local_weights,
                                                int node_count) const {
    if (local_indptr.size() != static_cast<std::size_t>(node_count + 1)) {
        throw std::runtime_error("neural field local_indptr size must be node_count + 1");
    }
    if (local_indices.size() != local_weights.size()) {
        throw std::runtime_error("neural field local_indices and local_weights size mismatch");
    }
    if (local_indptr.front() != 0 ||
        local_indptr.back() != static_cast<int>(local_indices.size())) {
        throw std::runtime_error("neural field local_indptr must start at 0 and end at nnz");
    }
    for (std::size_t row = 0; row + 1 < local_indptr.size(); ++row) {
        if (local_indptr[row] > local_indptr[row + 1]) {
            throw std::runtime_error("neural field local_indptr must be nondecreasing");
        }
    }
    for (std::size_t edge = 0; edge < local_indices.size(); ++edge) {
        const int source = local_indices[edge];
        if (source < 0 || source >= node_count) {
            throw std::runtime_error("neural field local connectivity source index out of range");
        }
    }
}

std::vector<FieldOutputPlan> Network::build_field_reducers(
    std::vector<FieldOutputReducer> reducers,
    const mind_sim::macro::sim::NeuralFieldRule& rule,
    int node_count) const {
    if (reducers.empty()) {
        throw std::runtime_error("neural field requires at least one ROI output reducer");
    }
    std::vector<unsigned char> output_seen(static_cast<std::size_t>(output_count()), 0);
    std::vector<FieldOutputPlan> plan;
    plan.reserve(reducers.size());
    const auto node_stride = static_cast<std::size_t>(node_count);
    const auto roi_stride = static_cast<std::size_t>(roi_count());
    for (const auto& reducer: reducers) {
        if (reducer.state_index < 0 || reducer.state_index >= rule.state_count()) {
            throw std::runtime_error("neural field reducer state index out of range");
        }
        if (reducer.output_index < 0 || reducer.output_index >= output_count()) {
            throw std::runtime_error("neural field reducer output index out of range");
        }
        const auto output_index = static_cast<std::size_t>(reducer.output_index);
        if (output_seen[output_index] != 0) {
            throw std::runtime_error("neural field reducers must target unique outputs");
        }
        output_seen[output_index] = 1;
        plan.push_back(FieldOutputPlan{
            .state_offset = static_cast<std::size_t>(reducer.state_index) * node_stride,
            .output_offset = output_index * roi_stride,
        });
    }
    return plan;
}

}  // namespace mind_sim::macro::frontend
