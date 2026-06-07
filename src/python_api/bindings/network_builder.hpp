#pragma once

#include "macro/frontend/network.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_sim::python_api::bindings {

struct Sim;
struct PointProcessView;

class NetworkBuilder {
  public:
    explicit NetworkBuilder(mind_sim::macro::frontend::Connectivity connectivity);

    [[nodiscard]] mind_sim::macro::frontend::ROI roi(int index) const;
    [[nodiscard]] mind_sim::macro::frontend::ROI roi(const std::string& label) const;
    [[nodiscard]] std::vector<mind_sim::macro::frontend::ROI> rois() const;
    [[nodiscard]] int roi_count() const;
    [[nodiscard]] double min_positive_delay() const;

    void record_rois(std::vector<int> roi_indices);
    void record_all_rois();
    void record_outputs(std::vector<std::string> output_names);
    void record_all_outputs();
    void set_dt(double dt);
    void set_exchange_window(double exchange_window);
    void load_mech(std::string directory);

    void set_dc_input(int roi_index, std::unordered_map<std::string, double> values);
    void use_region(int roi_index,
                    std::string library_path,
                    std::unordered_map<std::string, double> initial_state,
                    std::unordered_map<std::string, double> params);
    void use_neural_field(std::string name,
                          std::string library_path,
                          mind_sim::macro::frontend::NodeToRoiMap node_map,
                          mind_sim::macro::frontend::LocalConnectivity local,
                          std::unordered_map<std::string, double> initial_state,
                          std::unordered_map<std::string, double> params);
    void macro2macro(int source_roi,
                     int target_roi,
                     std::string library_path,
                     std::unordered_map<std::string, double> params);
    void use_micro(int roi_index);
    void macro2micro(int roi_index,
                     std::string library_path,
                     int gid,
                     const PointProcessView& target,
                     double weight,
                     double delay,
                     std::unordered_map<std::string, double> state,
                     std::unordered_map<std::string, double> params);
    void micro2macro(int roi_index,
                     std::string library_path,
                     std::unordered_map<std::string, double> state,
                     std::unordered_map<std::string, double> params);

    [[nodiscard]] mind_sim::macro::frontend::Network build() const;

  private:
    struct RegionConfig {
        int roi{-1};
        std::shared_ptr<mind_sim::macro::sim::RegionRule> rule{};
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    struct FieldConfig {
        std::string name{};
        std::shared_ptr<mind_sim::macro::sim::NeuralFieldRule> rule{};
        mind_sim::macro::frontend::NodeToRoiMap node_map;
        mind_sim::macro::frontend::LocalConnectivity local;
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    struct MacroToMacroConfig {
        int source{-1};
        int target{-1};
        std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> rule{};
        std::unordered_map<std::string, double> params{};
    };

    struct MicroBindingConfig {
        int roi{-1};
        Sim* micro{nullptr};
        std::vector<int> gid_range_begins{};
        std::vector<int> gid_range_ends{};
    };

    struct MicroInputConfig {
        int roi{-1};
        std::shared_ptr<mind_sim::cosim::bridge::MicroInputRule> rule{};
        int gid{-1};
        int source_id{-1};
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    struct MicroOutputConfig {
        int roi{-1};
        std::shared_ptr<mind_sim::cosim::bridge::MicroOutputRule> rule{};
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    [[nodiscard]] int roi_index(const std::string& label) const;
    [[nodiscard]] std::string resolve_rule_path(const std::string& mechanism) const;
    void register_mod_library(const std::string& library_path, const std::string& expected_name = {});
    void validate_roi_index(int roi_index, const char* what) const;

    mind_sim::macro::frontend::Connectivity connectivity_;
    std::optional<std::vector<int>> recorded_rois_{};
    std::optional<std::vector<std::string>> recorded_outputs_{};
    std::vector<std::unordered_map<std::string, double>> dc_inputs_{};
    std::vector<RegionConfig> regions_{};
    std::vector<FieldConfig> fields_{};
    std::vector<MacroToMacroConfig> macro_to_macro_{};
    std::vector<MicroBindingConfig> micro_bindings_{};
    std::vector<MicroInputConfig> micro_inputs_{};
    std::vector<MicroOutputConfig> micro_outputs_{};
    std::unordered_map<std::string, std::string> mod_libraries_{};
    double dt_{0.0};
    double exchange_window_{0.0};
};

class MacroConfig {
  public:
    void load_mech(std::string directory);
    void set_dt(double dt);
    void set_exchange_window(double exchange_window);
    void apply(NetworkBuilder& builder) const;

  private:
    std::vector<std::string> mech_dirs_{};
    double dt_{0.0};
    double exchange_window_{0.0};
};

}  // namespace mind_sim::python_api::bindings
