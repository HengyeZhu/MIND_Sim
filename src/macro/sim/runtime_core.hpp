#pragma once

#include "macro/frontend/network.hpp"
#include "macro/sim/model.hpp"
#include "macro/sim/types.hpp"

#include <memory>
#include <vector>

namespace mind_sim::macro::sim {

struct CouplingGraph {
    std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule{};
    std::vector<double> params{};
    std::vector<int> targets{};
    std::vector<int> target_edge_offsets{};
    std::vector<int> edge_sources{};
    std::vector<double> edge_weights{};
    std::vector<int> edge_delay_steps{};
    std::vector<int> edge_delay_offsets{};
    std::vector<int> read_exposure_offsets{};
    std::vector<int> write_input_offsets{};
    int max_delay_steps{0};
};

struct CouplingRuntime {
    std::vector<CouplingGraph> graphs{};
    int history_capacity{1};
};

struct DcInputEntry {
    int offset{0};
    double value{0.0};
};

struct CouplingEvaluationGraph {
    const CouplingGraph* graph{nullptr};
    std::vector<int> targets{};
};

struct CouplingEvaluation {
    int history_capacity{1};
    std::vector<CouplingEvaluationGraph> graphs{};
    std::vector<int> clear_offsets{};
    std::vector<DcInputEntry> dc_inputs{};
};

struct RegionGroup {
    std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
    std::vector<int> roi_indices{};
    std::vector<double> state_soa{};
    std::vector<double> params_soa{};
    std::vector<int> read_input_offsets{};
    std::vector<int> write_exposure_offsets{};
};

struct RoiOwnerPartition {
    std::vector<int> neural_mass_rois{};
    std::vector<int> neural_field_rois{};
    std::vector<int> detailed_microcircuit_rois{};
};

[[nodiscard]] RoiOwnerPartition collect_roi_owners(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& region_owners,
    const std::vector<mind_sim::macro::frontend::NeuralFieldOwner>& field_owners,
    const std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    bool require_micro_output_rule);

void validate_single_roi_owner(int roi_count,
                               const RoiOwnerPartition& owners,
                               const char* message);

[[nodiscard]] std::vector<int> continuous_macro_rois(const RoiOwnerPartition& owners);

[[nodiscard]] CouplingRuntime build_coupling_runtime(
    const mind_sim::macro::frontend::Network& network,
    double dt_macro);

[[nodiscard]] CouplingEvaluation coupling_evaluation_for_targets(
    const CouplingRuntime& coupling_runtime,
    const std::vector<int>& target_rois,
    int roi_count,
    int input_count,
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& dc_inputs);

void apply_couplings(const CouplingEvaluation& evaluation,
                     int roi_count,
                     int input_count,
                     int exposure_count,
                     int step,
                     const std::vector<double>& history,
                     std::vector<double>& input_soa);

[[nodiscard]] std::vector<double> exposure_buffers_to_soa(
    const std::vector<mind_sim::macro::sim::ScalarBuffer>& exposures,
    int roi_count,
    int exposure_count);

void initialize_history(std::vector<double>& history,
                        int history_capacity,
                        int roi_count,
                        int exposure_count,
                        const std::vector<double>& exposure_soa);

void write_history_slot(std::vector<double>& history,
                        int slot,
                        int roi_count,
                        int exposure_count,
                        const std::vector<double>& exposure_soa);

void append_exposure_record(mind_sim::macro::sim::ExposureRecord& record,
                            const std::vector<double>& exposure_soa);

[[nodiscard]] std::vector<RegionGroup> build_region_groups(
    const std::vector<mind_sim::macro::frontend::RegionOwner>& owners);

void aggregate_field_exposures(const mind_sim::macro::frontend::NeuralFieldOwner& owner,
                               std::vector<double>& exposure_soa);

void step_neural_field(mind_sim::macro::frontend::NeuralFieldOwner& owner,
                       int roi_count,
                       const std::vector<double>& input_soa,
                       std::vector<double>& exposure_soa,
                       double t,
                       double dt);

}  // namespace mind_sim::macro::sim
