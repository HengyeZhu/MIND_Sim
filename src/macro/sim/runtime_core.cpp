#include "macro/sim/runtime_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace mind_sim::macro::sim {

namespace {

std::string region_rule_signature(const mind_sim::macro::sim::RegionRule& rule) {
    return rule.library_path() + "|" + std::to_string(rule.input_count()) + "|" +
           std::to_string(rule.exposure_count()) + "|" +
           std::to_string(rule.state_count()) + "|" + std::to_string(rule.param_count());
}

struct PendingCouplingGroup {
    std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule{};
    std::vector<double> params{};
    std::vector<int> read_exposure_offsets{};
    std::vector<int> write_input_offsets{};
    std::vector<std::pair<int, int>> edges{};
};

bool same_params(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool same_ints(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

std::vector<PendingCouplingGroup> group_coupling_projections(
    const std::vector<mind_sim::macro::frontend::CouplingProjection>& projections) {
    std::vector<PendingCouplingGroup> groups;
    for (const auto& projection: projections) {
        auto found = std::find_if(
            groups.begin(),
            groups.end(),
            [&](const PendingCouplingGroup& group) {
                return group.rule.get() == projection.rule.get() &&
                       same_params(group.params, projection.params) &&
                       same_ints(group.read_exposure_offsets, projection.read_exposure_offsets) &&
                       same_ints(group.write_input_offsets, projection.write_input_offsets);
            });
        if (found == groups.end()) {
            groups.push_back(PendingCouplingGroup{
                .rule = projection.rule,
                .params = projection.params,
                .read_exposure_offsets = projection.read_exposure_offsets,
                .write_input_offsets = projection.write_input_offsets,
            });
            found = std::prev(groups.end());
        }
        found->edges.emplace_back(projection.target_roi, projection.source_roi);
    }
    return groups;
}

CouplingGraph build_coupling_graph(const mind_sim::macro::frontend::Network& network,
                                   const PendingCouplingGroup& group,
                                   double dt_macro) {
    const int roi_count = network.roi_count();
    const auto& weights = network.weights_flat();
    const auto& delays = network.delays_flat();
    CouplingGraph graph;
    graph.rule = group.rule;
    graph.params = group.params;
    graph.read_exposure_offsets = group.read_exposure_offsets;
    graph.write_input_offsets = group.write_input_offsets;
    std::vector<int> edge_counts(static_cast<std::size_t>(roi_count), 0);
    std::vector<unsigned char> has_target(static_cast<std::size_t>(roi_count), 0);
    for (const auto& [target, source]: group.edges) {
        const auto matrix_offset = static_cast<std::size_t>(target * roi_count + source);
        const double weight = weights[matrix_offset];
        if (weight == 0.0) {
            continue;
        }
        const int delay_steps = static_cast<int>(std::lrint(delays[matrix_offset] / dt_macro));
        edge_counts[static_cast<std::size_t>(target)] += 1;
        graph.max_delay_steps = std::max(graph.max_delay_steps, delay_steps);
        if (has_target[static_cast<std::size_t>(target)] == 0) {
            has_target[static_cast<std::size_t>(target)] = 1;
            graph.targets.push_back(target);
        }
    }
    graph.target_edge_offsets.assign(static_cast<std::size_t>(roi_count + 1), 0);
    for (int target = 0; target < roi_count; ++target) {
        graph.target_edge_offsets[static_cast<std::size_t>(target + 1)] =
            graph.target_edge_offsets[static_cast<std::size_t>(target)] +
            edge_counts[static_cast<std::size_t>(target)];
    }
    const auto edge_count = static_cast<std::size_t>(graph.target_edge_offsets.back());
    graph.edge_sources.resize(edge_count);
    graph.edge_weights.resize(edge_count);
    graph.edge_delay_steps.resize(edge_count);
    std::vector<int> write_positions = graph.target_edge_offsets;
    for (const auto& [target, source]: group.edges) {
        const auto matrix_offset = static_cast<std::size_t>(target * roi_count + source);
        const double weight = weights[matrix_offset];
        if (weight == 0.0) {
            continue;
        }
        const int delay_steps = static_cast<int>(std::lrint(delays[matrix_offset] / dt_macro));
        const auto index = static_cast<std::size_t>(
            write_positions[static_cast<std::size_t>(target)]++);
        graph.edge_sources[index] = source;
        graph.edge_weights[index] = weight;
        graph.edge_delay_steps[index] = delay_steps;
    }
    return graph;
}

}  // namespace

CouplingRuntime build_coupling_runtime(const mind_sim::macro::frontend::Network& network,
                                       double dt_macro) {
    CouplingRuntime runtime;
    for (const auto& group: group_coupling_projections(network.coupling_projections())) {
        auto graph = build_coupling_graph(network, group, dt_macro);
        runtime.history_capacity = std::max(runtime.history_capacity, graph.max_delay_steps + 1);
        runtime.graphs.push_back(std::move(graph));
    }
    const int history_stride = network.roi_count() * network.exposure_count();
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

CouplingEvaluation coupling_evaluation_for_targets(
    const CouplingRuntime& coupling_runtime,
    const std::vector<int>& target_rois,
    int roi_count,
    int input_count,
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& dc_inputs) {
    std::vector<unsigned char> selected(static_cast<std::size_t>(roi_count), 0);
    for (int roi: target_rois) {
        selected[static_cast<std::size_t>(roi)] = 1;
    }

    CouplingEvaluation evaluation;
    evaluation.history_capacity = coupling_runtime.history_capacity;
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
    for (const auto& graph: coupling_runtime.graphs) {
        CouplingEvaluationGraph filtered;
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

void apply_couplings(const CouplingEvaluation& evaluation,
                     int roi_count,
                     int input_count,
                     int exposure_count,
                     int step,
                     const std::vector<double>& history,
                     std::vector<double>& input_soa) {
    const auto input_size = static_cast<std::size_t>(roi_count * input_count);
    if (input_soa.size() != input_size) {
        input_soa.resize(input_size);
    }
    for (int offset: evaluation.clear_offsets) {
        input_soa[static_cast<std::size_t>(offset)] = 0.0;
    }
    for (const auto& graph_view: evaluation.graphs) {
        const auto& graph = *graph_view.graph;
        graph.rule->apply_flat(roi_count,
                               input_count,
                               exposure_count,
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
                               graph.read_exposure_offsets,
                               graph.write_input_offsets);
    }
    for (const auto& input: evaluation.dc_inputs) {
        input_soa[static_cast<std::size_t>(input.offset)] += input.value;
    }
}

std::vector<double> exposure_buffers_to_soa(
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& exposures,
    int roi_count,
    int exposure_count) {
    std::vector<double> soa(static_cast<std::size_t>(roi_count * exposure_count), 0.0);
    for (int roi = 0; roi < roi_count; ++roi) {
        const auto& buffer = exposures[static_cast<std::size_t>(roi)];
        for (int exposure = 0; exposure < exposure_count; ++exposure) {
            soa[static_cast<std::size_t>(exposure * roi_count + roi)] =
                buffer.values[static_cast<std::size_t>(exposure)];
        }
    }
    return soa;
}

void initialize_history(std::vector<double>& history,
                        int history_capacity,
                        int roi_count,
                        int exposure_count,
                        const std::vector<double>& exposure_soa) {
    const auto slot_size = static_cast<std::size_t>(roi_count * exposure_count);
    for (int slot = 0; slot < history_capacity; ++slot) {
        const auto base = static_cast<std::size_t>(slot) * slot_size;
        std::copy(exposure_soa.begin(),
                  exposure_soa.end(),
                  history.begin() + static_cast<std::ptrdiff_t>(base));
    }
}

void write_history_slot(std::vector<double>& history,
                        int slot,
                        int roi_count,
                        int exposure_count,
                        const std::vector<double>& exposure_soa) {
    const auto slot_size = static_cast<std::size_t>(roi_count * exposure_count);
    const auto base = static_cast<std::size_t>(slot) * slot_size;
    std::copy(exposure_soa.begin(),
              exposure_soa.end(),
              history.begin() + static_cast<std::ptrdiff_t>(base));
}

void append_exposure_record(mind_sim::macro::sim::ExposureRecord& record,
                            const std::vector<double>& exposure_soa) {
    const int roi_count = record.roi_count;
    const int exposure_count = record.exposure_count;
    const auto base = record.values.size();
    const auto sample_width =
        record.roi_indices.size() * static_cast<std::size_t>(exposure_count);
    record.values.resize(base + sample_width);
    auto* out = record.values.data() + base;
    for (int roi: record.roi_indices) {
        for (int exposure = 0; exposure < exposure_count; ++exposure) {
            *out++ = exposure_soa[static_cast<std::size_t>(exposure * roi_count + roi)];
        }
    }
}

std::vector<RegionGroup> build_region_groups(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& owners) {
    struct PendingGroup {
        std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
        std::vector<int> roi_indices{};
    };

    std::vector<PendingGroup> pending_groups;
    std::vector<std::size_t> owner_group_indices;
    owner_group_indices.reserve(owners.size());
    std::unordered_map<std::string, std::size_t> index_by_rule_signature;
    for (const auto& owner: owners) {
        const auto key = region_rule_signature(*owner.rule);
        auto found = index_by_rule_signature.find(key);
        if (found == index_by_rule_signature.end()) {
            const auto index = pending_groups.size();
            index_by_rule_signature.emplace(key, index);
            pending_groups.push_back(PendingGroup{.rule = owner.rule});
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
        group.state_soa.assign(static_cast<std::size_t>(owner_count * state_count), 0.0);
        group.params_soa.assign(static_cast<std::size_t>(owner_count * param_count), 0.0);
        groups.push_back(std::move(group));
    }
    std::vector<int> group_write_positions(groups.size(), 0);
    for (std::size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
        const auto group_index = owner_group_indices[owner_index];
        auto& group = groups[group_index];
        const auto& owner = owners[owner_index];
        const int unit = group_write_positions[group_index]++;
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

}  // namespace mind_sim::macro::sim
