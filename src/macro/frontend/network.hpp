#pragma once

#include "cosim/bridge/interfaces.hpp"
#include "macro/frontend/connectivity.hpp"
#include "macro/frontend/local_connectivity.hpp"
#include "macro/frontend/node_to_roi_map.hpp"
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
    std::vector<int> target_input_offsets{};
    std::vector<int> source_exposure_offsets{};
};

struct FieldOutputReducer {
    int state_index{-1};
    int output_index{-1};
};

struct FieldOutputPlan {
    std::size_t state_offset{0};
    std::size_t output_offset{0};
};

struct NeuralFieldOwner {
    std::string name{};
    std::shared_ptr<mind_sim::macro::sim::NeuralFieldRule> rule{};
    int node_count{0};
    std::vector<int> node_to_roi{};
    std::vector<int> roi_node_offsets{};
    std::vector<int> roi_nodes{};
    std::vector<double> roi_node_weights{};
    std::vector<int> local_indptr{};
    std::vector<int> local_indices{};
    std::vector<double> local_weights{};
    std::vector<double> state_soa{};
    std::vector<double> previous_state_soa{};
    std::vector<double> params{};
    std::vector<int> target_input_offsets{};
    std::vector<FieldOutputPlan> reducers{};
    std::vector<int> owned_rois{};
};

struct GidRange {
    int begin{0};
    int end{0};
};

struct MicroRoiBinding {
    int roi_index{-1};
    std::vector<GidRange> gid_ranges{};
    std::shared_ptr<mind_sim::cosim::bridge::MicroInputRule> input_rule{};
    std::vector<double> input_state{};
    std::vector<double> input_params{};
    std::vector<int> input_source_indices{};
    std::vector<int> target_input_offsets{};
    std::shared_ptr<mind_sim::cosim::bridge::MicroOutputRule> output_rule{};
    std::vector<double> output_state{};
    std::vector<double> output_params{};
    std::vector<int> source_exposure_offsets{};
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
    std::vector<int> source_exposure_offsets{};
    std::vector<int> target_input_offsets{};
};

struct MicroTraceRecorder {
    double* value{nullptr};
    std::vector<double>* samples{nullptr};
};

class Network {
  public:
    Network(Connectivity connectivity,
            std::vector<std::string> inputs,
            std::vector<std::string> outputs,
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

    [[nodiscard]] int input_index(const std::string& input_name) const;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] const std::vector<std::string>& inputs() const noexcept;
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

    void set_dc_input(const ROI& roi, const mind_sim::macro::sim::ScalarBuffer& input);
    void set_dc_input_value(const ROI& roi, const std::string& input_name, double value);
    [[nodiscard]] const std::vector<mind_sim::macro::sim::ScalarBuffer>& dc_inputs() const noexcept;

    void macro_to_macro(const ROI& source_roi,
                const ROI& target_roi,
                std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule,
                std::vector<double> params,
                std::vector<int> source_exposure_offsets,
                std::vector<int> target_input_offsets);
    void use_region_rule(const ROI& roi,
                         std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                         std::vector<double> state,
                         std::vector<double> params,
                         std::vector<int> target_input_offsets,
                         std::vector<int> source_exposure_offsets);
    void use_neural_field(std::string name,
                          std::shared_ptr<mind_sim::macro::sim::NeuralFieldRule> rule,
                          NodeToRoiMap node_map,
                          LocalConnectivity local_connectivity,
                          std::vector<double> state_soa,
                          std::vector<double> params,
                          std::vector<int> target_input_offsets,
                          std::vector<FieldOutputReducer> reducers);
    int use_micro(std::string name,
                  std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data);
    void bind_micro_roi(int micro_circuit_index,
                        const ROI& roi,
                        std::vector<GidRange> gid_ranges);
    void configure_macro_to_micro_rule(const ROI& roi,
                                       std::shared_ptr<mind_sim::cosim::bridge::MicroInputRule> input_rule,
                                       std::vector<double> input_state,
                                       std::vector<double> input_params,
                                       std::vector<int> input_source_indices,
                                       std::vector<int> target_input_offsets);
    void configure_micro_output_rule(const ROI& roi,
                                     std::shared_ptr<mind_sim::cosim::bridge::MicroOutputRule> output_rule,
                                     std::vector<double> output_state,
                                     std::vector<double> output_params,
                                     std::vector<int> source_exposure_offsets);

    [[nodiscard]] const std::vector<MacroToMacroProjection>& macro_to_macro_projections() const noexcept;
    [[nodiscard]] const std::vector<RegionOwner>& region_owners() const noexcept;
    [[nodiscard]] const std::vector<NeuralFieldOwner>& neural_field_owners() const noexcept;
    [[nodiscard]] const std::vector<MicroCircuitOwner>& micro_circuits() const noexcept;

  private:
    enum class OwnerKind {
        Empty,
        NeuralMass,
        Field,
        Micro,
    };
    struct FieldMappingInfo {
        std::vector<int> owned_rois{};
        std::vector<int> roi_node_offsets{};
        std::vector<int> roi_nodes{};
        std::vector<double> roi_node_weights{};
    };

    void validate_roi_index(int roi_index, const char* what) const;
    void claim_roi(int roi_index, OwnerKind kind);
    [[nodiscard]] FieldMappingInfo build_field_mapping(const std::vector<int>& node_to_roi,
                                                       const std::vector<double>& node_weights) const;
    void validate_field_local_connectivity(const std::vector<int>& local_indptr,
                                           const std::vector<int>& local_indices,
                                           const std::vector<double>& local_weights,
                                           int node_count) const;
    [[nodiscard]] std::vector<FieldOutputPlan> build_field_reducers(
        std::vector<FieldOutputReducer> reducers,
        const mind_sim::macro::sim::NeuralFieldRule& rule,
        int node_count) const;
    void validate_gid_ranges(const std::vector<GidRange>& gid_ranges) const;
    [[nodiscard]] MicroCircuitOwner& require_micro_circuit(int micro_circuit_index);
    [[nodiscard]] MicroRoiBinding& require_micro_binding(int roi_index);
    void rebuild_micro_gid_bounds(MicroCircuitOwner& circuit) const;
    [[nodiscard]] std::vector<int> normalize_roi_indices(std::vector<int> roi_indices,
                                                         const char* what) const;
    [[nodiscard]] std::vector<int> normalize_output_indices(std::vector<int> output_indices,
                                                              const char* what) const;

    Connectivity connectivity_;
    std::vector<std::string> inputs_{};
    std::unordered_map<std::string, int> input_to_index_{};
    std::vector<std::string> outputs_{};
    std::unordered_map<std::string, int> output_to_index_{};
    std::vector<mind_sim::macro::sim::ScalarBuffer> output_history_start_{};
    std::vector<mind_sim::macro::sim::ScalarBuffer> dc_inputs_{};
    std::vector<int> recorded_rois_{};
    std::vector<int> recorded_outputs_{};
    std::vector<MicroTraceRecorder> micro_record_targets_{};
    std::vector<std::vector<double>*> micro_time_record_targets_{};
    std::vector<MacroToMacroProjection> macro_to_macro_projections_{};
    std::vector<OwnerKind> owner_kind_{};
    std::vector<RegionOwner> region_owners_{};
    std::vector<NeuralFieldOwner> neural_field_owners_{};
    std::vector<MicroCircuitOwner> micro_circuits_{};
    std::vector<int> micro_circuit_by_roi_{};
    std::vector<int> micro_binding_by_roi_{};
    double dt_{0.0};
    double exchange_window_{0.0};
};

}  // namespace mind_sim::macro::frontend
