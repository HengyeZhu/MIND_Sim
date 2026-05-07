#pragma once

#include "bridge/sim/interfaces.hpp"
#include "macro/sim/model.hpp"
#include "macro/sim/types.hpp"
#include "micro/sim/core_neuron_data.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_sim::macro::frontend {

struct ROI {
    int index{-1};
    std::string label{};
};

class Connectivity {
  public:
    Connectivity(std::vector<std::string> labels,
                 std::vector<std::vector<double>> weights,
                 std::vector<std::vector<double>> delays);

    [[nodiscard]] int roi_count() const noexcept;
    [[nodiscard]] const std::vector<ROI>& rois() const noexcept;
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept;
    [[nodiscard]] const std::vector<double>& weights() const noexcept;
    [[nodiscard]] const std::vector<double>& delays() const noexcept;
    [[nodiscard]] double weight_at(int target_roi, int source_roi) const;
    [[nodiscard]] double delay_at(int target_roi, int source_roi) const;
    [[nodiscard]] double min_positive_delay() const noexcept;
    [[nodiscard]] int roi_index(const std::string& label) const;

  private:
    [[nodiscard]] std::vector<double> flatten_square_matrix(
        const std::vector<std::vector<double>>& matrix,
        const std::string& name,
        bool require_non_negative) const;
    [[nodiscard]] std::size_t matrix_offset(int target_roi, int source_roi) const;

    std::vector<std::string> labels_{};
    std::vector<ROI> rois_{};
    std::unordered_map<std::string, int> label_to_index_{};
    std::vector<double> weights_{};
    std::vector<double> delays_{};
};

struct RegionOwner {
    int roi_index{-1};
    std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
    std::vector<double> state{};
    std::vector<double> params{};
};

struct GidRange {
    int begin{0};
    int end{0};
};

struct MicroRoiBinding {
    int roi_index{-1};
    std::vector<GidRange> gid_ranges{};
    std::shared_ptr<mind_sim::bridge::sim::MicroInputRule> input_rule{};
    std::vector<double> input_state{};
    std::vector<double> input_params{};
    std::vector<mind_sim::bridge::sim::RandomStreamBinding> input_random_streams{};
    std::vector<int> input_port_bases{};
    std::vector<int> input_read_offsets{};
    std::shared_ptr<mind_sim::bridge::sim::MicroOutputRule> output_rule{};
    std::vector<double> output_state{};
    std::vector<double> output_params{};
    std::vector<int> output_write_offsets{};
};

struct MicroCircuitOwner {
    std::string name{};
    std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data{};
    std::vector<MicroRoiBinding> bindings{};
    int gid_begin{0};
    int gid_end{0};
    std::vector<int> gid_to_binding{};
};

struct CouplingProjection {
    int source_roi{-1};
    int target_roi{-1};
    std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule{};
    std::vector<double> params{};
    std::vector<int> read_exposure_offsets{};
    std::vector<int> write_input_offsets{};
};

class Network {
  public:
    Network(Connectivity connectivity,
            std::vector<std::string> inputs,
            std::vector<std::string> exposures,
            std::vector<int> recorded_rois);
    Network(std::vector<std::string> roi_labels,
            std::vector<std::vector<double>> weights,
            std::vector<std::vector<double>> delays,
            std::vector<std::string> inputs,
            std::vector<std::string> exposures,
            std::vector<int> recorded_rois);

    [[nodiscard]] ROI roi(int roi_index) const;
    [[nodiscard]] ROI roi(const std::string& label) const;
    [[nodiscard]] const std::vector<ROI>& rois() const noexcept;
    [[nodiscard]] int roi_index(const std::string& label) const;
    [[nodiscard]] int roi_count() const noexcept;

    [[nodiscard]] int input_index(const std::string& input_name) const;
    [[nodiscard]] int input_count() const noexcept;
    [[nodiscard]] const std::vector<std::string>& inputs() const noexcept;
    [[nodiscard]] int exposure_index(const std::string& exposure_name) const;
    [[nodiscard]] int exposure_count() const noexcept;
    [[nodiscard]] const std::vector<std::string>& exposures() const noexcept;
    [[nodiscard]] const std::vector<int>& recorded_rois() const noexcept;
    void set_recorded_rois(std::vector<int> recorded_rois);

    [[nodiscard]] double weight_at(int target_roi, int source_roi) const;
    [[nodiscard]] double delay_at(int target_roi, int source_roi) const;
    [[nodiscard]] double min_positive_delay() const noexcept;
    [[nodiscard]] const std::vector<double>& weights_flat() const noexcept;
    [[nodiscard]] const std::vector<double>& delays_flat() const noexcept;

    void set_initial_exposure(const ROI& roi, const mind_sim::macro::sim::ScalarBuffer& exposure);
    void set_initial_exposure_value(const ROI& roi, const std::string& exposure_name, double value);
    [[nodiscard]] const std::vector<mind_sim::macro::sim::ScalarBuffer>& initial_exposures() const noexcept;

    void set_dc_input(const ROI& roi, const mind_sim::macro::sim::ScalarBuffer& input);
    void set_dc_input_value(const ROI& roi, const std::string& input_name, double value);
    [[nodiscard]] const std::vector<mind_sim::macro::sim::ScalarBuffer>& dc_inputs() const noexcept;

    void couple(const ROI& source_roi,
                const ROI& target_roi,
                std::shared_ptr<mind_sim::macro::sim::CouplingRule> rule,
                std::vector<double> params,
                std::vector<int> read_exposure_offsets,
                std::vector<int> write_input_offsets);
    void use_region_rule(const ROI& roi,
                         std::shared_ptr<mind_sim::macro::sim::RegionRule> rule,
                         std::vector<double> state,
                         std::vector<double> params);
    int use_micro(std::string name,
                  std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_data);
    void bind_micro_roi(int micro_circuit_index,
                        const ROI& roi,
                        std::vector<GidRange> gid_ranges);
    void configure_micro_input_rule(const ROI& roi,
                                    std::shared_ptr<mind_sim::bridge::sim::MicroInputRule> input_rule,
                                    std::vector<double> input_state,
                                    std::vector<double> input_params,
                                    std::vector<mind_sim::bridge::sim::RandomStreamBinding> input_random_streams,
                                    std::vector<int> input_port_bases,
                                    std::vector<int> input_read_offsets);
    void configure_micro_output_rule(const ROI& roi,
                                     std::shared_ptr<mind_sim::bridge::sim::MicroOutputRule> output_rule,
                                     std::vector<double> output_state,
                                     std::vector<double> output_params,
                                     std::vector<int> output_write_offsets);

    [[nodiscard]] const std::vector<CouplingProjection>& coupling_projections() const noexcept;
    [[nodiscard]] const std::vector<RegionOwner>& region_owners() const noexcept;
    [[nodiscard]] const std::vector<MicroCircuitOwner>& micro_circuits() const noexcept;

  private:
    enum class OwnerKind {
        Empty,
        Region,
        Micro,
    };

    void validate_roi_index(int roi_index, const char* what) const;
    void validate_exposure_size(const mind_sim::macro::sim::ScalarBuffer& exposure,
                                const char* what) const;
    void validate_input_size(const mind_sim::macro::sim::ScalarBuffer& input,
                             const char* what) const;
    void claim_roi(int roi_index, OwnerKind kind);
    void validate_region_rule_schema(const mind_sim::macro::sim::RegionRule& rule) const;
    void validate_coupling_rule_schema(const mind_sim::macro::sim::CouplingRule& rule) const;
    void validate_gid_ranges(const std::vector<GidRange>& gid_ranges) const;
    [[nodiscard]] MicroCircuitOwner& require_micro_circuit(int micro_circuit_index);
    [[nodiscard]] MicroRoiBinding& require_micro_binding(int roi_index);
    void rebuild_micro_gid_index(MicroCircuitOwner& circuit) const;
    [[nodiscard]] std::vector<int> normalize_roi_indices(std::vector<int> roi_indices,
                                                         const char* what) const;

    Connectivity connectivity_;
    std::vector<std::string> inputs_{};
    std::unordered_map<std::string, int> input_to_index_{};
    std::vector<std::string> exposures_{};
    std::unordered_map<std::string, int> exposure_to_index_{};
    std::vector<mind_sim::macro::sim::ScalarBuffer> initial_exposures_{};
    std::vector<mind_sim::macro::sim::ScalarBuffer> dc_inputs_{};
    std::vector<int> recorded_rois_{};
    std::vector<CouplingProjection> coupling_projections_{};
    std::vector<OwnerKind> owner_kind_{};
    std::vector<RegionOwner> region_owners_{};
    std::vector<MicroCircuitOwner> micro_circuits_{};
};

}  // namespace mind_sim::macro::frontend
