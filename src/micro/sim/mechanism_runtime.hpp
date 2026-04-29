#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace mind_sim::micro::sim {

enum class CoreDParamKind {
    IonVariable,
    IonStyle,
    Area,
    PointProcess,
    NetSend,
    Random,
    BbcorePointer,
    Unsupported,
};

enum class CoreMechanismFieldRole {
    Parameter,
    Assigned,
    State,
    Range,
};

struct CoreDParamBinding {
    std::string mechanism{};
    int dparam_index{-1};
    CoreDParamKind kind{CoreDParamKind::Unsupported};
    std::string dparam_field{};
    std::string mechanism_field{};
    std::string ion{};
    int ion_field{-1};
    int ion_conc_style{0};
    int ion_rev_style{0};
    bool ion_write_interior{false};
    bool ion_write_exterior{false};
};

struct CoreMechanismField {
    std::string name{};
    CoreMechanismFieldRole role{CoreMechanismFieldRole::Range};
    int array_size{1};
    int data_offset{-1};
    double default_value{0.0};
    bool is_global{false};
};

struct CoreRegisteredMechanism {
    int type{-1};
    std::string name{};
    bool is_point{false};
    bool is_artificial{false};
    bool has_net_receive{false};
    int net_receive_weight_count{0};
    int parameter_size{0};
    int dparam_size{0};
    std::vector<CoreMechanismField> fields{};
    std::vector<CoreDParamBinding> dparam_bindings{};
};

void load_core_mechanism_library(const std::filesystem::path& path);
void ensure_core_mechanisms_registered();

[[nodiscard]] int core_mechanism_type(const std::string& name);
[[nodiscard]] const char* core_mechanism_name(int type) noexcept;
[[nodiscard]] int core_mechanism_capacity() noexcept;
[[nodiscard]] int core_mechanism_order_rank(int type) noexcept;
[[nodiscard]] int core_ion_register(const std::string& ion, double charge);
[[nodiscard]] double core_ion_charge(const std::string& ion_mechanism);
[[nodiscard]] const std::vector<CoreDParamBinding>& core_dparam_bindings();
[[nodiscard]] const std::vector<CoreRegisteredMechanism>& core_registered_mechanisms();
[[nodiscard]] double core_global_scalar(const std::string& name);

void core_note_registered_mechanism(int type, const char** mechanism_info);
void core_note_writes_concentration(int type);
void core_note_point_mechanism(int type);
void core_note_artificial_cell(int type);
void core_note_net_receive(int type, int weight_count);
void core_note_prop_size(int type, int psize, int dpsize);
void core_note_dparam_semantic(int type, int index, const char* semantic);
void core_note_global_scalar(const char* name, double* value);
void core_note_global_scalar_field(int type, const char* field_name, const char* hoc_name, double* value);
void core_set_global_parameter(const std::string& mechanism, const std::string& field, double value);
void core_set_global_scalar(const std::string& name, double value);
void core_verify_ion_charges_defined();

}  // namespace mind_sim::micro::sim
