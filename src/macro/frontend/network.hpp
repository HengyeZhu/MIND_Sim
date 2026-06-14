#pragma once

#include "cosim/transform/interfaces.hpp"
#include "macro/frontend/connectivity.hpp"
#include "macro/sim/model.hpp"
#include "macro/sim/types.hpp"
#include "micro/sim/core_neuron_data.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_sim::macro::frontend {

struct RegionOwner {
    int roi_index{-1};
    std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
    std::vector<double> state{};
    std::vector<double> params{};
    std::vector<int> exposure_offsets{};
};

struct GidRange {
    int begin{0};
    int end{0};
};

struct MicroOutputTransform {
    std::shared_ptr<mind_sim::cosim::transform::MicroOutputRule> rule{};
    std::vector<double> state{};
    std::vector<double> params{};
    std::vector<int> source_sids{};
    std::vector<int> exposure_offsets{};
};

struct MicroRoiBinding {
    int roi_index{-1};
    std::vector<GidRange> gid_ranges{};
    std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> input_rule{};
    std::vector<double> input_state{};
    std::vector<double> input_params{};
    std::vector<int> macro2micro_indices{};
    std::vector<int> macro2micro_source_ids{};
    std::vector<int> exposure_offsets{};
    std::vector<MicroOutputTransform> output_transforms{};
};

struct MicroCircuitOwner {
    std::string name{};
    std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data{};
    std::vector<MicroRoiBinding> bindings{};
    int gid_begin{0};
    int gid_end{0};
};

struct MacroToMacroProjection {
    int source_roi{-1};
    int target_roi{-1};
    std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule{};
    std::vector<double> params{};
    std::vector<int> read_source_offsets{};
    std::vector<int> read_target_offsets{};
    std::vector<int> write_source_offsets{};
    std::vector<int> write_target_offsets{};
};

struct MicroTraceRecorder {
    double* value{nullptr};
    std::vector<double>* samples{nullptr};
};

class Network {
  public:
    enum class InitialHistoryLayout {
        TimeOutputRoi,
        TimeRoiOutput,
    };

    Network(Connectivity connectivity,
            std::vector<std::string> exposures,
            std::vector<int> recorded_rois,
            std::vector<int> recorded_outputs);

    [[nodiscard]] ROI roi(int roi_index) const;
    [[nodiscard]] ROI roi(const std::string& label) const;
    [[nodiscard]] const std::vector<ROI>& rois() const noexcept;
    [[nodiscard]] int roi_index(const std::string& label) const;
    [[nodiscard]] int roi_count() const noexcept;
    [[nodiscard]] const Connectivity& connectivity() const noexcept;
    void set_dt(double dt);
    [[nodiscard]] double dt() const noexcept;
    void set_exchange_window(double exchange_window);
    [[nodiscard]] double exchange_window() const noexcept;

    [[nodiscard]] int output_index(const std::string& output_name) const;
    [[nodiscard]] int output_count() const noexcept;
    [[nodiscard]] const std::vector<std::string>& outputs() const noexcept;
    [[nodiscard]] const std::vector<int>& recorded_rois() const noexcept;
    [[nodiscard]] const std::vector<int>& recorded_outputs() const noexcept;
    void set_recorded_rois(std::vector<int> recorded_rois);
    void set_recorded_outputs(std::vector<int> recorded_outputs);
    void record_micro(double* value, std::vector<double>* samples);
    void record_micro_time(std::vector<double>* samples);
    [[nodiscard]] const std::vector<MicroTraceRecorder>& micro_record_targets() const noexcept;
    [[nodiscard]] const std::vector<std::vector<double>*>& micro_time_record_targets() const noexcept;

    [[nodiscard]] const std::vector<mind_sim::macro::sim::ScalarBuffer>& output_history_start() const noexcept;
    void set_initial_history(const std::vector<std::string>& output_names,
                             int time_count,
                             int axis1_count,
                             int axis2_count,
                             const std::vector<double>& values,
                             InitialHistoryLayout layout);
    [[nodiscard]] bool has_initial_history() const noexcept;
    [[nodiscard]] int initial_history_time_count() const noexcept;
    [[nodiscard]] const std::vector<double>& initial_history() const noexcept;

    void macro_to_macro(const ROI& source_roi,
                const ROI& target_roi,
                std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule,
                std::vector<double> params,
                std::vector<int> read_source_offsets,
                std::vector<int> read_target_offsets,
                std::vector<int> write_source_offsets,
                std::vector<int> write_target_offsets);
    void use_region_rule(const ROI& roi,
                         std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                         std::vector<double> state,
                         std::vector<double> params,
                         std::vector<int> exposure_offsets);
    int use_micro(std::string name,
                  std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data);
    void bind_micro_roi(int micro_circuit_index,
                        const ROI& roi,
                        std::vector<GidRange> gid_ranges);
    void configure_macro_to_micro_rule(const ROI& roi,
                                       std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> input_rule,
                                       std::vector<double> input_state,
                                       std::vector<double> input_params,
                                       std::vector<int> macro2micro_indices,
                                       std::vector<int> macro2micro_source_ids,
                                       std::vector<int> exposure_offsets);
    void configure_micro_output_rule(const ROI& roi,
                                     std::shared_ptr<mind_sim::cosim::transform::MicroOutputRule> output_rule,
                                     std::vector<double> output_state,
                                     std::vector<double> output_params,
                                     std::vector<int> output_source_sids,
                                     std::vector<int> exposure_offsets);

    [[nodiscard]] const std::vector<MacroToMacroProjection>& macro_to_macro_projections() const noexcept;
    [[nodiscard]] const std::vector<RegionOwner>& region_owners() const noexcept;
    [[nodiscard]] const std::vector<MicroCircuitOwner>& micro_circuits() const noexcept;

  private:
    enum class OwnerKind {
        Empty,
        NeuralMass,
        Micro,
    };

    void validate_roi_index(int roi_index, const char* what) const;
    void claim_roi(int roi_index, OwnerKind kind);
    void validate_gid_ranges(const std::vector<GidRange>& gid_ranges) const;
    [[nodiscard]] MicroCircuitOwner& require_micro_circuit(int micro_circuit_index);
    [[nodiscard]] MicroRoiBinding& require_micro_binding(int roi_index);
    void rebuild_micro_gid_bounds(MicroCircuitOwner& circuit) const;
    [[nodiscard]] std::vector<int> normalize_roi_indices(std::vector<int> roi_indices,
                                                         const char* what) const;
    [[nodiscard]] std::vector<int> normalize_output_indices(std::vector<int> output_indices,
                                                              const char* what) const;

    Connectivity connectivity_;
    std::vector<std::string> outputs_{};
    std::unordered_map<std::string, int> output_to_index_{};
    std::vector<mind_sim::macro::sim::ScalarBuffer> output_history_start_{};
    std::vector<double> initial_history_{};
    int initial_history_time_count_{0};
    std::vector<int> recorded_rois_{};
    std::vector<int> recorded_outputs_{};
    std::vector<MicroTraceRecorder> micro_record_targets_{};
    std::vector<std::vector<double>*> micro_time_record_targets_{};
    std::vector<MacroToMacroProjection> macro_to_macro_projections_{};
    std::vector<unsigned char> region_owner_seen_{};
    std::vector<unsigned char> micro_owner_seen_{};
    std::vector<RegionOwner> region_owners_{};
    std::vector<MicroCircuitOwner> micro_circuits_{};
    std::vector<int> micro_circuit_by_roi_{};
    std::vector<int> micro_binding_by_roi_{};
    double dt_{0.0};
    double exchange_window_{0.0};
};

}  // namespace mind_sim::macro::frontend
