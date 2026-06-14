#pragma once

#include "biophysical/biophys_common.hpp"
#include "biophysical/mechanism_catalog.hpp"
#include "morph/section_spec.hpp"
#include "micro/sim/core_neuron_data.hpp"
#include "micro/sim/micro_runtime.hpp"
#include "network/network_registry.hpp"

#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mind_sim::micro::frontend {

struct MorphologyTemplateSpec {
    std::string name{};
    std::shared_ptr<const std::vector<mind_micro_frontend::SectionSpec>> sections{};
    int num_cells{0};
};

struct PopulationRange {
    std::string name{};
    int gid_begin{0};
    int gid_end{0};
};

struct VariableRef {
    enum class Kind {
        Location,
        LocationVoltage,
        Mechanism,
    };

    Kind kind{Kind::Location};
    int gid{-1};
    int section_index{-1};
    double x{0.5};
    int insert_id{-1};
    std::string mech{"global"};
    std::string var{};
    int array_index{-1};
};

enum class MechanismPlacementKind : std::uint8_t {
    SectionSet = 0,
    Location = 1,
    Artificial = 2,
};

struct CoreMechanismInsert {
    int id{-1};
    int metadata_id{-1};
    int runtime_type{-1};
    MechanismPlacementKind placement{MechanismPlacementKind::SectionSet};
    int gid{-1};
    double loc{std::numeric_limits<double>::quiet_NaN()};
    int section_index{-1};
    std::size_t param_begin{0};
    std::size_t param_end{0};
    int target_id{-1};
    std::size_t instance_begin{0};
    std::size_t instance_end{0};
};

struct CoreParamOverride {
    int field_index{-1};
    int data_offset{-1};
    mind_micro_biophysical::ParamValue value{};
};

struct CoreMechanismInstance {
    int insert_id{-1};
    std::size_t insert_index{0};
    int node_index{-1};
    int gid{-1};
    int section_index{-1};
    int segment_index{0};
    int event_target_id{-1};
};

struct CoreMechanismBlockBuilder {
    int metadata_id{-1};
    int runtime_type{-1};
    std::vector<CoreMechanismInstance> instances{};
};

struct CoreMechanismBuildState {
    std::vector<CoreMechanismInsert> inserts{};
    std::vector<CoreParamOverride> param_overrides{};
    std::vector<CoreMechanismBlockBuilder> blocks{};
    std::unordered_map<int, std::size_t> block_index_by_metadata_id{};

    void clear() {
        inserts.clear();
        param_overrides.clear();
        blocks.clear();
        block_index_by_metadata_id.clear();
    }
};

struct CoreSectionProperties {
    double celsius{6.3};
    std::vector<std::size_t> cell_section_offsets{0};
    std::vector<double> v_init{};
    std::vector<double> cm{};
    std::vector<double> ra{};

    void clear() {
        celsius = 6.3;
        cell_section_offsets.assign(1, 0);
        v_init.clear();
        cm.clear();
        ra.clear();
    }
};

struct IonRangeOverride {
    int gid{-1};
    std::vector<std::size_t> section_indices{};
    std::string ion{};
    std::string ion_mechanism{};
    std::string field{};
    mind_micro_biophysical::ParamValue value{};
};

struct IonStyleOverride {
    int gid{-1};
    std::vector<std::size_t> section_indices{};
    std::string ion{};
    std::string ion_mechanism{};
    int style{0};
};

class MicroFrontendModel {
  public:
    MicroFrontendModel();

    void set_dt(double dt);
    [[nodiscard]] double dt() const noexcept { return dt_; }
    void set_num_threads(int num_threads);
    [[nodiscard]] int num_threads() const noexcept { return requested_thread_count_; }

    void set_celsius(double celsius);
    [[nodiscard]] double celsius() const noexcept { return section_properties_.celsius; }
    void set_secondorder(int secondorder);
    [[nodiscard]] int secondorder() const noexcept { return core_neuron_data_->secondorder; }

    void set_device(const std::string& device);
    void load_mech(std::string path);
    [[nodiscard]] const std::vector<std::string>& loaded_mech_paths() const noexcept {
        return loaded_mech_paths_;
    }
    [[nodiscard]] int ion_register(std::string ion, double charge);
    [[nodiscard]] double ion_charge(const std::string& ion_mechanism) const;
    [[nodiscard]] double nernst(double ci, double co, double charge) const;
    [[nodiscard]] double ghk(double v, double ci, double co, double charge) const;
    void set_global_scalar(const std::string& name, double value);
    [[nodiscard]] double global_scalar(const std::string& name) const;

    void build_morphology(std::vector<MorphologyTemplateSpec> templates);
    [[nodiscard]] bool has_morphology() const noexcept { return has_morphology_; }
    [[nodiscard]] int population_count() const;
    [[nodiscard]] const PopulationRange& population(std::size_t index) const;
    [[nodiscard]] const PopulationRange& population(const std::string& name) const;
    [[nodiscard]] const std::vector<PopulationRange>& populations() const noexcept { return populations_; }

    [[nodiscard]] int section_count(int gid) const;
    [[nodiscard]] std::vector<std::size_t> section_indices_for_label(int gid, const std::string& label) const;
    [[nodiscard]] std::string section_label(int gid, std::size_t section_index) const;

    void set_cell_v_init(int gid, double value);
    [[nodiscard]] double cell_v_init(int gid) const;
    void set_section_group_property(int gid,
                                    const std::vector<std::size_t>& section_indices,
                                    mind_micro_biophysical::ObjectOpKind kind,
                                    double value);
    [[nodiscard]] double section_group_property(int gid,
                                                const std::vector<std::size_t>& section_indices,
                                                mind_micro_biophysical::ObjectOpKind kind) const;
    void set_section_group_ion_range(int gid,
                                     const std::vector<std::size_t>& section_indices,
                                     std::string field,
                                     mind_micro_biophysical::ParamValue value);
    [[nodiscard]] double section_group_ion_range(int gid,
                                                 const std::vector<std::size_t>& section_indices,
                                                 const std::string& field) const;
    [[nodiscard]] int section_group_ion_style(int gid,
                                              const std::vector<std::size_t>& section_indices,
                                              const std::string& ion_mechanism) const;
    int set_section_group_ion_style(int gid,
                                    const std::vector<std::size_t>& section_indices,
                                    const std::string& ion_mechanism,
                                    int c_style,
                                    int e_style,
                                    int einit,
                                    int eadvance,
                                    int cinit);
    int insert_mechanism(int gid,
                         std::vector<std::size_t> section_indices,
                         std::optional<double> loc,
                         const std::string& mech,
                         mind_micro_biophysical::ParamList params);
    [[nodiscard]] int register_artificial_cell(const std::string& mech,
                                               mind_micro_biophysical::ParamList params);
    [[nodiscard]] std::string mechanism_mech(int insert_id) const;
    [[nodiscard]] int mechanism_gid(int insert_id) const;
    [[nodiscard]] int mechanism_section_index(int insert_id) const;
    [[nodiscard]] double mechanism_loc(int insert_id) const;
    [[nodiscard]] MechanismPlacementKind mechanism_placement(int insert_id) const;
    [[nodiscard]] double mechanism_scalar(int insert_id, const std::string& key) const;
    void set_mechanism_scalar(int insert_id, const std::string& key, double value);

    [[nodiscard]] int register_spike_source(int sid,
                                            const VariableRef& source,
                                            std::optional<double> threshold);
    [[nodiscard]] int sid_connect(int sid, int post_insert_id, double weight, double delay);
    [[nodiscard]] int event_target_connect(int source_insert_id,
                                           int post_insert_id,
                                           double weight,
                                           double delay);
    [[nodiscard]] int register_spike_input_source();
    [[nodiscard]] int spike_input_connect(int spike_input_id,
                                          int post_insert_id,
                                          double weight,
                                          double delay);
    [[nodiscard]] int spike_input_runtime_index(int spike_input_id) const;
    [[nodiscard]] std::size_t netcon_weight_count(int connection_id) const;
    [[nodiscard]] double netcon_weight(int connection_id, int array_index) const;
    void set_netcon_weight(int connection_id, int array_index, double value);
    [[nodiscard]] double netcon_delay(int connection_id) const;
    void set_netcon_delay(int connection_id, double value);
    [[nodiscard]] double netcon_threshold(int connection_id) const;
    void set_netcon_threshold(int connection_id, double value);
    [[nodiscard]] int netcon_runtime_index(int connection_id) const;
    [[nodiscard]] int netcon_target_event_target_id(int connection_id) const;
    [[nodiscard]] int netcon_source_event_target_id(int connection_id) const;
    [[nodiscard]] int netcon_source_spike_input_id(int connection_id) const;
    void schedule_spike_input_event(int spike_input_id, double time);
    void schedule_netcon_event(int connection_id, double time);
    [[nodiscard]] const std::vector<double>& spike_times() const noexcept;
    [[nodiscard]] const std::vector<int>& spike_gids() const noexcept;
    void clear_spikes() noexcept;
    [[nodiscard]] double read_variable(const VariableRef& ref) const;
    [[nodiscard]] double* variable_pointer(const VariableRef& ref);
    [[nodiscard]] VariableRef location_variable_ref_from_python_attr(int gid,
                                                                     int section_index,
                                                                     double x,
                                                                     const std::string& attr);
    [[nodiscard]] VariableRef object_variable_ref_from_python_attr(int insert_id,
                                                                   const std::string& attr);

    int build_microcircuit();
    int finitialize(double v_init);
    int run(double tstop);
    int continue_run(double runtime);
    int continue_run_with_recording(double runtime,
                                    const std::vector<VariableRef>& refs,
                                    const std::vector<double*>& sample_buffers,
                                    int sample_count);
    int fadvance();
    [[nodiscard]] double time() const noexcept { return t_; }

    [[nodiscard]] const mind_micro_model::CellTemplateMorphLayout& morph_layout() const;
    [[nodiscard]] const mind_micro_network::NetworkRegistry& network_registry() const noexcept {
        return network_registry_;
    }
    [[nodiscard]] const CoreMechanismBuildState& core_mechanism_builder() const noexcept {
        return core_mechanism_builder_;
    }
    [[nodiscard]] const mind_sim::micro::sim::CoreNeuronData& core_neuron_data() const noexcept {
        return *core_neuron_data_;
    }
    [[nodiscard]] std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_neuron_data_shared() const noexcept {
        return core_neuron_data_;
    }
    [[nodiscard]] bool core_initialized() const noexcept {
        return core_initialized_;
    }

  private:
    void require_morphology() const;
    [[nodiscard]] std::size_t require_insert_index(int insert_id) const;
    [[nodiscard]] mind_micro_biophysical::ParamList params_for_insert(int insert_id) const;
    [[nodiscard]] double* resolve_variable_pointer(const VariableRef& ref, const char* action) const;
    void init_section_properties();
    [[nodiscard]] std::size_t section_property_offset(int gid, std::size_t section_index) const;
    void load_default_mechanism_metadata();
    void refresh_mechanism_runtime_cache();
    [[nodiscard]] const std::vector<mind_sim::micro::sim::CoreDParamBinding>& dparam_bindings_for_metadata(
        int metadata_id) const;
    [[nodiscard]] std::pair<std::string, std::string> resolve_ion_field(const std::string& field) const;
    [[nodiscard]] std::string resolve_ion_mechanism(const std::string& ion_mechanism) const;
    [[nodiscard]] int resolve_cached_original_node(int gid, int section_index, double loc);
    [[nodiscard]] int runtime_node_for_original(int original_node) const;
    [[nodiscard]] int thread_for_gid(int gid) const;
    [[nodiscard]] int thread_for_original_node(int original_node) const;
    [[nodiscard]] int thread_for_instance(const CoreMechanismInstance& instance) const;
    void reset_core_node_storage();
    int intern_mechanism_type(const std::string& name);
    void apply_section_axial_resistance();
    [[nodiscard]] CoreMechanismBlockBuilder& block_for_metadata(int metadata_id, int runtime_type);
    void rebuild_core_mechanisms();
    void rebuild_core_network();

    double dt_{0.025};
    int requested_thread_count_{1};
    double t_{0.0};
    bool has_morphology_{false};
    bool microcircuit_built_{false};
    bool core_initialized_{false};
    bool default_mechanism_metadata_loaded_{false};
    std::vector<std::string> loaded_mech_paths_{};

    mind_micro_model::CellTemplateMorphLayout morph_layout_{};
    std::vector<PopulationRange> populations_{};
    std::unordered_map<std::string, std::size_t> population_index_by_name_{};
    CoreSectionProperties section_properties_{};
    std::vector<IonRangeOverride> ion_range_overrides_{};
    std::vector<IonStyleOverride> ion_style_overrides_{};
    mind_micro_biophysical::MechanismCatalog mechanism_catalog_{};
    std::unordered_map<int, std::vector<mind_sim::micro::sim::CoreDParamBinding>> dparam_bindings_by_metadata_id_{};
    CoreMechanismBuildState core_mechanism_builder_{};
    mind_micro_network::NetworkRegistry network_registry_{};
    std::shared_ptr<mind_sim::micro::sim::CoreNeuronData> core_neuron_data_{
        std::make_shared<mind_sim::micro::sim::CoreNeuronData>()};
    mind_sim::micro::sim::MicroDeviceConfig device_config_{};
    std::unique_ptr<mind_sim::micro::sim::MicroRuntime> runtime_backend_{};
    mind_sim::micro::sim::MicroSpikeTable recorded_spikes_{};
    std::vector<int> morph_parent_index_{};
    std::vector<int> original_node_gid_{};
    std::vector<int> original_node_thread_{};
    std::vector<int> gid_thread_{};
    std::vector<std::vector<int>> thread_gids_{};
    std::vector<std::vector<int>> original_nodes_by_thread_{};
    std::vector<int> insert_runtime_thread_{};
    std::vector<std::size_t> insert_runtime_row_{};
    std::vector<double> morph_area_{};
    std::vector<double> morph_diam_{};
    std::vector<int> runtime_node_by_original_{};
    std::vector<double> axial_a_ra1_{};
    std::vector<double> axial_b_ra1_{};
    std::vector<double> axial_ri_ra1_{};
    std::vector<double> axial_a_scale_{};
    std::vector<double> section_location_cache_locs_{};
    std::vector<int> section_location_cache_nodes_{};
};

}  // namespace mind_sim::micro::frontend
