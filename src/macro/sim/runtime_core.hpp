#pragma once

#include "macro/frontend/network.hpp"
#include "macro/sim/model.hpp"
#include "macro/sim/types.hpp"

#include <memory>
#include <vector>

namespace mind_sim::macro::sim {

struct MacroToMacroGraph {
    std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule{};
    std::vector<double> params{};
    std::vector<int> targets{};
    std::vector<int> target_edge_offsets{};
    std::vector<int> edge_sources{};
    std::vector<double> edge_weights{};
    std::vector<int> edge_delay_steps{};
    std::vector<int> edge_delay_offsets{};
    std::vector<int> read_source_offsets{};
    std::vector<int> read_target_offsets{};
    std::vector<int> write_source_offsets{};
    std::vector<int> write_target_offsets{};
    int max_delay_steps{0};
};

struct MacroToMacroRuntime {
    std::vector<MacroToMacroGraph> graphs{};
    int history_capacity{1};
};

struct MacroToMacroEvaluationGraph {
    const MacroToMacroGraph* graph{nullptr};
    std::vector<int> targets{};
};

struct MacroToMacroEvaluation {
    int history_capacity{1};
    bool reads_current_exposures{false};
    std::vector<MacroToMacroEvaluationGraph> graphs{};
    std::vector<int> clear_offsets{};
};

struct RegionGroup {
    std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
    std::vector<int> roi_indices{};
    std::vector<double> state_soa{};
    std::vector<double> params_soa{};
    std::vector<int> exposure_offsets{};
};

struct RoiOwnerPartition {
    std::vector<int> neural_mass_rois{};
    std::vector<int> detailed_microcircuit_rois{};
};

[[nodiscard]] RoiOwnerPartition collect_roi_owners(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& region_owners,
    const std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    bool require_micro_output_rule);

void validate_single_roi_owner(int roi_count,
                               const RoiOwnerPartition& owners,
                               const char* message);

[[nodiscard]] std::vector<int> continuous_macro_rois(const RoiOwnerPartition& owners);

[[nodiscard]] MacroToMacroRuntime build_macro_to_macro_runtime(
    const mind_sim::macro::frontend::Network& network,
    double dt_macro);

[[nodiscard]] MacroToMacroEvaluation macro_to_macro_evaluation_for_targets(
    const MacroToMacroRuntime& macro_to_macro_runtime,
    const std::vector<int>& target_rois,
    int roi_count);

void apply_macro_to_macro(const MacroToMacroEvaluation& evaluation,
                     int roi_count,
                     int exposure_count,
                     int step,
                     const std::vector<double>& history,
                     std::vector<double>& exposure_soa);

[[nodiscard]] std::vector<double> output_buffers_to_soa(
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& outputs,
    int roi_count,
    int output_count);

void initialize_history(std::vector<double>& history,
                        int history_capacity,
                        int roi_count,
                        int output_count,
                        const std::vector<double>& output_soa);
[[nodiscard]] int initialize_history(std::vector<double>& history,
                                     int history_capacity,
                                     int roi_count,
                                     int output_count,
                                     const std::vector<double>& output_soa,
                                     const std::vector<double>& initial_history,
                                     int initial_history_time_count);

void write_history_slot(std::vector<double>& history,
                        int slot,
                        int roi_count,
                        int output_count,
                        const std::vector<double>& output_soa);

void append_record_table(mind_sim::macro::sim::RecordTable& record,
                         const std::vector<double>& output_soa,
                         int exposure_count);

[[nodiscard]] std::vector<RegionGroup> build_region_groups(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& owners);

}  // namespace mind_sim::macro::sim
