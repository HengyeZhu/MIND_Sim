#include "macro/sim/runtime_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mind_sim::macro::sim {

namespace {

std::string region_rule_signature(const mind_sim::macro::sim::RegionRule& rule) {
    return rule.library_path() + "|" + std::to_string(rule.input_count()) + "|" +
           std::to_string(rule.output_count()) + "|" +
           std::to_string(rule.state_count()) + "|" + std::to_string(rule.param_count());
}

std::string vector_signature(const std::vector<int>& values) {
    std::string out;
    for (int value: values) {
        out += "|" + std::to_string(value);
    }
    return out;
}

struct PendingMacroToMacroGroup {
    std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule{};
    std::vector<double> params{};
    std::vector<int> source_exposure_offsets{};
    std::vector<int> target_input_offsets{};
    std::vector<std::pair<int, int>> edges{};
};

std::vector<PendingMacroToMacroGroup> group_macro_to_macro_projections(
    const std::vector<mind_sim::macro::frontend::MacroToMacroProjection>& projections) {
    std::vector<PendingMacroToMacroGroup> groups;
    for (const auto& projection: projections) {
        auto found = std::find_if(
            groups.begin(),
            groups.end(),
            [&](const PendingMacroToMacroGroup& group) {
                return group.rule->library_path() == projection.rule->library_path() &&
                       group.params == projection.params &&
                       group.source_exposure_offsets == projection.source_exposure_offsets &&
                       group.target_input_offsets == projection.target_input_offsets;
            });
        if (found == groups.end()) {
            groups.push_back(PendingMacroToMacroGroup{
                .rule = projection.rule,
                .params = projection.params,
                .source_exposure_offsets = projection.source_exposure_offsets,
                .target_input_offsets = projection.target_input_offsets,
            });
            found = std::prev(groups.end());
        }
        found->edges.emplace_back(projection.target_roi, projection.source_roi);
    }
    return groups;
}

MacroToMacroGraph build_macro_to_macro_graph(const mind_sim::macro::frontend::Network& network,
                                   const PendingMacroToMacroGroup& group,
                                   double dt_macro) {
    const int roi_count = network.roi_count();
    const auto& connectivity = network.connectivity();
    const auto& weights = connectivity.weights();
    const auto& delays = connectivity.delays();
    MacroToMacroGraph graph;
    graph.rule = group.rule;
    graph.params = group.params;
    graph.source_exposure_offsets = group.source_exposure_offsets;
    graph.target_input_offsets = group.target_input_offsets;
    std::vector<int> edge_counts(static_cast<std::size_t>(roi_count), 0);
    for (const auto& [target, source]: group.edges) {
        const auto matrix_offset = static_cast<std::size_t>(target * roi_count + source);
        const double weight = weights[matrix_offset];
        if (weight == 0.0) {
            continue;
        }
        const int delay_steps = static_cast<int>(std::lrint(delays[matrix_offset] / dt_macro));
        edge_counts[static_cast<std::size_t>(target)] += 1;
        graph.max_delay_steps = std::max(graph.max_delay_steps, delay_steps);
    }
    graph.target_edge_offsets.assign(static_cast<std::size_t>(roi_count + 1), 0);
    int target_count = 0;
    for (int target = 0; target < roi_count; ++target) {
        if (edge_counts[static_cast<std::size_t>(target)] != 0) {
            ++target_count;
        }
        graph.target_edge_offsets[static_cast<std::size_t>(target + 1)] =
            graph.target_edge_offsets[static_cast<std::size_t>(target)] +
            edge_counts[static_cast<std::size_t>(target)];
    }
    graph.targets.resize(static_cast<std::size_t>(target_count));
    int target_position = 0;
    for (int target = 0; target < roi_count; ++target) {
        if (edge_counts[static_cast<std::size_t>(target)] != 0) {
            graph.targets[static_cast<std::size_t>(target_position++)] = target;
        }
    }
    const auto edge_count = static_cast<std::size_t>(graph.target_edge_offsets.back());
    graph.edge_sources.resize(edge_count);
    graph.edge_weights.resize(edge_count);
    graph.edge_delay_steps.resize(edge_count);
    std::vector<int> fill_positions = graph.target_edge_offsets;
    for (const auto& [target, source]: group.edges) {
        const auto matrix_offset = static_cast<std::size_t>(target * roi_count + source);
        const double weight = weights[matrix_offset];
        if (weight == 0.0) {
            continue;
        }
        const int delay_steps = static_cast<int>(std::lrint(delays[matrix_offset] / dt_macro));
        const auto index = static_cast<std::size_t>(
            fill_positions[static_cast<std::size_t>(target)]++);
        graph.edge_sources[index] = source;
        graph.edge_weights[index] = weight;
        graph.edge_delay_steps[index] = delay_steps;
    }
    return graph;
}

}  // namespace

RoiOwnerPartition collect_roi_owners(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& region_owners,
    const std::vector<mind_sim::macro::frontend::NeuralFieldOwner>& field_owners,
    const std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    bool require_micro_output_rule) {
    std::size_t field_roi_count = 0;
    for (const auto& owner: field_owners) {
        field_roi_count += owner.owned_rois.size();
    }
    std::size_t micro_roi_count = 0;
    for (const auto& circuit: micro_circuits) {
        micro_roi_count += circuit.bindings.size();
    }

    RoiOwnerPartition owners;
    owners.neural_mass_rois.reserve(region_owners.size());
    owners.neural_field_rois.reserve(field_roi_count);
    owners.detailed_microcircuit_rois.reserve(micro_roi_count);

    for (const auto& owner: region_owners) {
        owners.neural_mass_rois.push_back(owner.roi_index);
    }
    for (const auto& owner: field_owners) {
        for (int roi: owner.owned_rois) {
            owners.neural_field_rois.push_back(roi);
        }
    }

    for (const auto& circuit: micro_circuits) {
        for (const auto& binding: circuit.bindings) {
            if (require_micro_output_rule && binding.output_transforms.empty()) {
                throw std::runtime_error("micro ROI is missing an output mod connection");
            }
            owners.detailed_microcircuit_rois.push_back(binding.roi_index);
        }
    }
    return owners;
}

void validate_single_roi_owner(int roi_count,
                               const RoiOwnerPartition& owners,
                               const char* message) {
    std::vector<unsigned char> owner_seen(static_cast<std::size_t>(roi_count), 0);
    const auto mark_owned = [&](const std::vector<int>& rois) {
        for (int roi: rois) {
            owner_seen[static_cast<std::size_t>(roi)] += 1;
        }
    };
    mark_owned(owners.neural_mass_rois);
    mark_owned(owners.neural_field_rois);
    mark_owned(owners.detailed_microcircuit_rois);
    for (int roi = 0; roi < roi_count; ++roi) {
        if (owner_seen[static_cast<std::size_t>(roi)] != 1) {
            throw std::runtime_error(message);
        }
    }
}

std::vector<int> continuous_macro_rois(const RoiOwnerPartition& owners) {
    auto rois = owners.neural_mass_rois;
    rois.reserve(owners.neural_mass_rois.size() + owners.neural_field_rois.size());
    rois.insert(rois.end(), owners.neural_field_rois.begin(), owners.neural_field_rois.end());
    return rois;
}

MacroToMacroRuntime build_macro_to_macro_runtime(const mind_sim::macro::frontend::Network& network,
                                       double dt_macro) {
    MacroToMacroRuntime runtime;
    for (const auto& group: group_macro_to_macro_projections(network.macro_to_macro_projections())) {
        auto graph = build_macro_to_macro_graph(network, group, dt_macro);
        runtime.history_capacity = std::max(runtime.history_capacity, graph.max_delay_steps + 1);
        runtime.graphs.push_back(std::move(graph));
    }
    const int history_stride = network.roi_count() * network.output_count();
    for (auto& graph: runtime.graphs) {
        graph.edge_delay_offsets.assign(graph.edge_delay_steps.size(), 0);
        for (std::size_t index = 0; index < graph.edge_delay_steps.size(); ++index) {
            const int delay = graph.edge_delay_steps[index] % runtime.history_capacity;
            const int delay_slot = (runtime.history_capacity - delay) % runtime.history_capacity;
            graph.edge_delay_offsets[index] = delay_slot * history_stride;
        }
    }
    return runtime;
}

MacroToMacroEvaluation macro_to_macro_evaluation_for_targets(
    const MacroToMacroRuntime& macro_to_macro_runtime,
    const std::vector<int>& target_rois,
    int roi_count,
    int input_count,
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& dc_inputs) {
    std::vector<unsigned char> selected(static_cast<std::size_t>(roi_count), 0);
    for (int roi: target_rois) {
        selected[static_cast<std::size_t>(roi)] = 1;
    }

    MacroToMacroEvaluation evaluation;
    evaluation.history_capacity = macro_to_macro_runtime.history_capacity;
    evaluation.clear_offsets.reserve(
        static_cast<std::size_t>(target_rois.size()) * static_cast<std::size_t>(input_count));
    for (int roi: target_rois) {
        for (int input = 0; input < input_count; ++input) {
            evaluation.clear_offsets.push_back(input * roi_count + roi);
        }
    }
    for (int roi: target_rois) {
        const auto& input = dc_inputs[static_cast<std::size_t>(roi)];
        for (int input_index = 0; input_index < input_count; ++input_index) {
            const auto value = input.values[static_cast<std::size_t>(input_index)];
            if (value != 0.0) {
                evaluation.dc_inputs.push_back(DcInputEntry{
                    .offset = input_index * roi_count + roi,
                    .value = value,
                });
            }
        }
    }
    for (const auto& graph: macro_to_macro_runtime.graphs) {
        MacroToMacroEvaluationGraph filtered;
        filtered.graph = &graph;
        for (int target: graph.targets) {
            if (selected[static_cast<std::size_t>(target)] != 0) {
                filtered.targets.push_back(target);
            }
        }
        if (!filtered.targets.empty()) {
            evaluation.graphs.push_back(std::move(filtered));
        }
    }
    return evaluation;
}

void apply_macro_to_macro(const MacroToMacroEvaluation& evaluation,
                     int roi_count,
                     int input_count,
                     int output_count,
                     int step,
                     const std::vector<double>& history,
                     std::vector<double>& input_soa) {
    const auto input_size = static_cast<std::size_t>(roi_count * input_count);
    if (input_soa.empty()) {
        input_soa.resize(input_size);
    }
    for (int offset: evaluation.clear_offsets) {
        input_soa[static_cast<std::size_t>(offset)] = 0.0;
    }
    for (const auto& graph_view: evaluation.graphs) {
        const auto& graph = *graph_view.graph;
        graph.rule->apply_flat(roi_count,
                               input_count,
                               output_count,
                               evaluation.history_capacity,
                               step,
                               graph_view.targets,
                               graph.target_edge_offsets,
                               graph.edge_sources,
                               graph.edge_weights,
                               graph.edge_delay_steps,
                               graph.edge_delay_offsets,
                               history,
                               input_soa,
                               graph.params,
                               graph.source_exposure_offsets,
                               graph.target_input_offsets);
    }
    for (const auto& input: evaluation.dc_inputs) {
        input_soa[static_cast<std::size_t>(input.offset)] += input.value;
    }
}

std::vector<double> output_buffers_to_soa(
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& outputs,
    int roi_count,
    int output_count) {
    std::vector<double> soa(static_cast<std::size_t>(roi_count * output_count), 0.0);
    for (int roi = 0; roi < roi_count; ++roi) {
        const auto& buffer = outputs[static_cast<std::size_t>(roi)];
        for (int output = 0; output < output_count; ++output) {
            soa[static_cast<std::size_t>(output * roi_count + roi)] =
                buffer.values[static_cast<std::size_t>(output)];
        }
    }
    return soa;
}

void initialize_history(std::vector<double>& history,
                        int history_capacity,
                        int roi_count,
                        int output_count,
                        const std::vector<double>& output_soa) {
    const auto slot_size = static_cast<std::size_t>(roi_count * output_count);
    for (int slot = 0; slot < history_capacity; ++slot) {
        const auto base = static_cast<std::size_t>(slot) * slot_size;
        std::copy(output_soa.begin(), output_soa.end(), history.begin() + static_cast<std::ptrdiff_t>(base));
    }
}

void write_history_slot(std::vector<double>& history,
                        int slot,
                        int roi_count,
                        int output_count,
                        const std::vector<double>& output_soa) {
    const auto slot_size = static_cast<std::size_t>(roi_count * output_count);
    const auto base = static_cast<std::size_t>(slot) * slot_size;
    std::copy(output_soa.begin(), output_soa.end(), history.begin() + static_cast<std::ptrdiff_t>(base));
}

void append_record_table(mind_sim::macro::sim::RecordTable& record,
                            const std::vector<double>& output_soa,
                            int source_exposure_count) {
    const int roi_count = record.roi_count;
    const auto base = record.values.size();
    const auto sample_width =
        record.roi_indices.size() * static_cast<std::size_t>(record.output_count);
    record.values.resize(base + sample_width);
    auto* out = record.values.data() + base;
    for (int roi: record.roi_indices) {
        for (int output: record.output_indices) {
            if (output < 0 || output >= source_exposure_count) {
                throw std::runtime_error("record output index out of range");
            }
            *out++ = output_soa[static_cast<std::size_t>(output * roi_count + roi)];
        }
    }
}

std::vector<RegionGroup> build_region_groups(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& owners) {
    struct PendingGroup {
        std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
        std::vector<int> roi_indices{};
        std::vector<int> target_input_offsets{};
        std::vector<int> source_exposure_offsets{};
    };

    std::vector<PendingGroup> pending_groups;
    std::vector<std::size_t> owner_group_indices;
    owner_group_indices.reserve(owners.size());
    std::unordered_map<std::string, std::size_t> index_by_rule_signature;
    for (const auto& owner: owners) {
        const auto key = region_rule_signature(*owner.rule) + "|target_input" +
                         vector_signature(owner.target_input_offsets) + "|source_exposure" +
                         vector_signature(owner.source_exposure_offsets);
        auto found = index_by_rule_signature.find(key);
        if (found == index_by_rule_signature.end()) {
            const auto index = pending_groups.size();
            index_by_rule_signature.emplace(key, index);
            pending_groups.push_back(PendingGroup{
                .rule = owner.rule,
                .target_input_offsets = owner.target_input_offsets,
                .source_exposure_offsets = owner.source_exposure_offsets,
            });
            found = index_by_rule_signature.find(key);
        }
        auto& group = pending_groups[found->second];
        group.roi_indices.push_back(owner.roi_index);
        owner_group_indices.push_back(found->second);
    }

    std::vector<RegionGroup> groups;
    groups.reserve(pending_groups.size());
    for (auto& pending: pending_groups) {
        const int owner_count = static_cast<int>(pending.roi_indices.size());
        const int state_count = pending.rule->state_count();
        const int param_count = pending.rule->param_count();
        RegionGroup group;
        group.rule = std::move(pending.rule);
        group.roi_indices = std::move(pending.roi_indices);
        group.target_input_offsets = std::move(pending.target_input_offsets);
        group.source_exposure_offsets = std::move(pending.source_exposure_offsets);
        group.state_soa.assign(static_cast<std::size_t>(owner_count * state_count), 0.0);
        group.params_soa.assign(static_cast<std::size_t>(owner_count * param_count), 0.0);
        groups.push_back(std::move(group));
    }
    std::vector<int> group_fill_positions(groups.size(), 0);
    for (std::size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
        const auto group_index = owner_group_indices[owner_index];
        auto& group = groups[group_index];
        const auto& owner = owners[owner_index];
        const int unit = group_fill_positions[group_index]++;
        const int owner_count = static_cast<int>(group.roi_indices.size());
        for (int state = 0; state < group.rule->state_count(); ++state) {
            group.state_soa[static_cast<std::size_t>(state * owner_count + unit)] =
                owner.state[static_cast<std::size_t>(state)];
        }
        for (int param = 0; param < group.rule->param_count(); ++param) {
            group.params_soa[static_cast<std::size_t>(param * owner_count + unit)] =
                owner.params[static_cast<std::size_t>(param)];
        }
    }
    return groups;
}

void aggregate_field_outputs(const mind_sim::macro::frontend::NeuralFieldOwner& owner,
                               std::vector<double>& output_soa) {
    const int* roi_node_offsets = owner.roi_node_offsets.data();
    const int* roi_nodes = owner.roi_nodes.data();
    const double* roi_node_weights = owner.roi_node_weights.data();
    if (owner.reducers.size() == 1) {
        const auto& reducer = owner.reducers.front();
        double* output = output_soa.data() + reducer.output_offset;
        const double* state = owner.state_soa.data() + reducer.state_offset;
        for (std::size_t owner_roi = 0; owner_roi < owner.owned_rois.size(); ++owner_roi) {
            double total = 0.0;
            const int begin = roi_node_offsets[owner_roi];
            const int end = roi_node_offsets[owner_roi + 1];
            for (int position = begin; position < end; ++position) {
                const auto node = static_cast<std::size_t>(roi_nodes[position]);
                total += roi_node_weights[position] * state[node];
            }
            output[static_cast<std::size_t>(owner.owned_rois[owner_roi])] = total;
        }
        return;
    }

    std::vector<double> totals(owner.reducers.size(), 0.0);
    for (std::size_t owner_roi = 0; owner_roi < owner.owned_rois.size(); ++owner_roi) {
        std::fill(totals.begin(), totals.end(), 0.0);
        const int begin = roi_node_offsets[owner_roi];
        const int end = roi_node_offsets[owner_roi + 1];
        for (int position = begin; position < end; ++position) {
            const auto node = static_cast<std::size_t>(roi_nodes[position]);
            const double weight = roi_node_weights[position];
            for (std::size_t reducer_index = 0; reducer_index < owner.reducers.size(); ++reducer_index) {
                const auto& reducer = owner.reducers[reducer_index];
                totals[reducer_index] += weight * owner.state_soa[reducer.state_offset + node];
            }
        }
        const auto roi = static_cast<std::size_t>(owner.owned_rois[owner_roi]);
        for (std::size_t reducer_index = 0; reducer_index < owner.reducers.size(); ++reducer_index) {
            const auto& reducer = owner.reducers[reducer_index];
            output_soa[reducer.output_offset + roi] = totals[reducer_index];
        }
    }
}

void step_neural_field(mind_sim::macro::frontend::NeuralFieldOwner& owner,
                       int roi_count,
                       const std::vector<double>& input_soa,
                       std::vector<double>& output_soa,
                       double t,
                       double dt) {
    owner.rule->step(owner.node_count,
                     owner.node_to_roi,
                     roi_count,
                     input_soa,
                     owner.previous_state_soa,
                     owner.state_soa,
                     owner.params,
                     owner.local_indptr,
                     owner.local_indices,
                     owner.local_weights,
                     owner.target_input_offsets,
                     t,
                     dt);
    aggregate_field_outputs(owner, output_soa);
}

}  // namespace mind_sim::macro::sim
