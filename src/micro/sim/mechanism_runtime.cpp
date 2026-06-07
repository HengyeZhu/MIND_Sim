#include "micro/sim/mechanism_runtime.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/mechanism/eion.hpp"
#include "coreneuron/mechanism/mech/cfile/cabvars.h"
#include "coreneuron/mechanism/membfunc.hpp"
#include "coreneuron/mechanism/neuron_registration.hpp"
#include "coreneuron/mechanism/register_mech.hpp"
#include "micro/sim/nrn_registration_mirror.hpp"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace coreneuron {
void initnrn();
void modl_reg();
void ion_reg(const char* name, double valence);
int ion_register(const char* name, double charge);
double nrn_ion_charge(int type);
void nrn_verify_ion_charge_defined();
extern std::map<std::string, int> mech2type;
}  // namespace coreneuron

namespace {

std::unordered_map<std::string, int> g_mechanism_type_by_name;
std::vector<std::string> g_mechanism_name_by_type;
std::vector<mind_sim::micro::sim::CoreRegisteredMechanism> g_registered_mechanisms;
std::vector<mind_sim::micro::sim::CoreDParamBinding> g_dparam_bindings;
std::unordered_map<std::string, std::unordered_map<std::string, double*>> g_global_scalars;
std::unordered_map<std::string, double*> g_global_scalars_by_name;
std::unordered_map<std::string, std::vector<double>> g_parameter_defaults_by_mechanism;
std::unordered_map<std::string,
                   std::unordered_map<std::string, neuron::mechanism::field_role>>
    g_registered_field_roles_by_mechanism;
std::unordered_map<std::string, std::vector<neuron::mechanism::detail::data_field_info>>
    g_neuron_side_fields_by_mechanism;
std::unordered_map<std::string, std::vector<neuron::mechanism::detail::dparam_field_info>>
    g_neuron_side_dparam_fields_by_mechanism;
std::unordered_map<std::string, std::vector<std::string>> g_dparam_semantics_by_mechanism;
std::vector<int> g_memb_order;
std::vector<int> g_memb_order_rank_by_type;
int g_memb_order_lastion = EXTRACELL + 1;
int g_recent_registered_type = -1;
bool g_core_base_registered = false;
bool g_dparam_binding_cache_dirty = true;

struct LoadedMechanismLibrary {
    std::filesystem::path library_path{};
    void* handle{nullptr};
    void (*modl_reg)(){nullptr};
    bool registered{false};
    std::vector<int> mechanism_types{};
};

std::vector<LoadedMechanismLibrary> g_loaded_libraries;
std::set<std::filesystem::path> g_loaded_library_paths;
LoadedMechanismLibrary* g_active_loaded_library = nullptr;

[[nodiscard]] std::string trim(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

[[nodiscard]] std::optional<int> fixed_core_type(std::string_view name) {
    if (name == "morphology") {
        return MORPHOLOGY;
    }
    if (name == "capacitance") {
        return CAP;
    }
    if (name == "extracellular") {
        return EXTRACELL;
    }
    return std::nullopt;
}

void resize_core_tables(std::size_t size) {
    auto& corenrn = coreneuron::corenrn;
    size = std::max(size, corenrn.get_memb_funcs().size());
    size = std::max(size, g_registered_mechanisms.size());
    if (corenrn.get_memb_funcs().size() < size) {
        corenrn.get_memb_funcs().resize(size);
    }
    if (corenrn.get_pnt_map().size() < size) {
        corenrn.get_pnt_map().resize(size);
    }
    if (corenrn.get_pnt_receive().size() < size) {
        corenrn.get_pnt_receive().resize(size);
    }
    if (corenrn.get_pnt_receive_init().size() < size) {
        corenrn.get_pnt_receive_init().resize(size);
    }
    if (corenrn.get_pnt_receive_size().size() < size) {
        corenrn.get_pnt_receive_size().resize(size);
    }
    if (corenrn.get_watch_check().size() < size) {
        corenrn.get_watch_check().resize(size);
    }
    if (corenrn.get_is_artificial().size() < size) {
        corenrn.get_is_artificial().resize(size, false);
    }
    if (corenrn.get_artcell_qindex().size() < size) {
        corenrn.get_artcell_qindex().resize(size);
    }
    if (corenrn.get_array_dims().size() < size) {
        corenrn.get_array_dims().resize(size);
    }
    if (corenrn.get_prop_param_size().size() < size) {
        corenrn.get_prop_param_size().resize(size);
    }
    if (corenrn.get_prop_dparam_size().size() < size) {
        corenrn.get_prop_dparam_size().resize(size);
    }
    if (corenrn.get_mech_data_layout().size() < size) {
        corenrn.get_mech_data_layout().resize(size, 1);
    }
    if (corenrn.get_bbcore_read().size() < size) {
        corenrn.get_bbcore_read().resize(size);
    }
    if (corenrn.get_bbcore_write().size() < size) {
        corenrn.get_bbcore_write().resize(size);
    }
    if (g_registered_mechanisms.size() < size) {
        g_registered_mechanisms.resize(size);
    }
}

void rebuild_memb_order_ranks() {
    int max_type = -1;
    for (const int type : g_memb_order) {
        max_type = std::max(max_type, type);
    }
    g_memb_order_rank_by_type.assign(static_cast<std::size_t>(max_type + 1), -1);
    for (std::size_t rank = 0; rank < g_memb_order.size(); ++rank) {
        const int type = g_memb_order[rank];
        if (type >= 0 && static_cast<std::size_t>(type) < g_memb_order_rank_by_type.size()) {
            g_memb_order_rank_by_type[static_cast<std::size_t>(type)] = static_cast<int>(rank);
        }
    }
}

void ensure_memb_order_slot(int type) {
    if (type < 0) {
        return;
    }
    if (std::find(g_memb_order.begin(), g_memb_order.end(), type) == g_memb_order.end()) {
        g_memb_order.push_back(type);
        rebuild_memb_order_ranks();
    }
}

void ensure_type_slot(int type) {
    if (type < 0) {
        return;
    }
    const auto required = static_cast<std::size_t>(type) + 1;
    if (g_mechanism_name_by_type.size() < required) {
        g_mechanism_name_by_type.resize(required);
    }
    resize_core_tables(required);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    if (info.type < 0) {
        info.type = type;
    }
    ensure_memb_order_slot(type);
}

int next_dynamic_type() {
    int type = static_cast<int>(g_mechanism_name_by_type.size());
    while (type == MORPHOLOGY || type == CAP || type == EXTRACELL) {
        ++type;
    }
    return type;
}

int intern_core_mechanism_name(const char* raw_name) {
    if (raw_name == nullptr || raw_name[0] == '\0') {
        return -1;
    }
    const std::string name(raw_name);
    if (const auto it = g_mechanism_type_by_name.find(name); it != g_mechanism_type_by_name.end()) {
        return it->second;
    }

    const int type = fixed_core_type(name).value_or(next_dynamic_type());
    ensure_type_slot(type);
    if (!g_mechanism_name_by_type[static_cast<std::size_t>(type)].empty() &&
        g_mechanism_name_by_type[static_cast<std::size_t>(type)] != name) {
        throw std::runtime_error("CoreNEURON mechanism type collision for '" + name + "'");
    }
    g_mechanism_name_by_type[static_cast<std::size_t>(type)] = name;
    g_mechanism_type_by_name.emplace(name, type);
    coreneuron::mech2type[name] = type;
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    info.type = type;
    info.name = name;
    return type;
}

void initialize_fixed_core_slots() {
    g_mechanism_type_by_name.clear();
    g_mechanism_name_by_type.assign(static_cast<std::size_t>(EXTRACELL) + 1, {});
    g_registered_mechanisms.assign(g_mechanism_name_by_type.size(), {});
    g_memb_order.clear();
    g_memb_order_rank_by_type.clear();
    g_memb_order_lastion = EXTRACELL + 1;
    resize_core_tables(g_mechanism_name_by_type.size());
    intern_core_mechanism_name("morphology");
    intern_core_mechanism_name("capacitance");
    intern_core_mechanism_name("extracellular");
    g_memb_order_lastion = static_cast<int>(g_memb_order.size());
}

[[nodiscard]] std::filesystem::path canonical_existing_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto canonical = std::filesystem::canonical(path, ec);
    if (ec) {
        throw std::runtime_error("mechanism path does not exist: " + path.string());
    }
    return canonical;
}

struct MechanismArtifactPath {
    std::filesystem::path library_path{};
};

[[nodiscard]] MechanismArtifactPath resolve_mechanism_artifact_path(
    const std::filesystem::path& requested_path) {
    const auto path = canonical_existing_path(requested_path);
    std::filesystem::path library_path;
    if (std::filesystem::is_regular_file(path) && path.filename() == "libcorenrnmech.so") {
        library_path = path;
    } else if (std::filesystem::is_directory(path) &&
               std::filesystem::is_regular_file(path / "libcorenrnmech.so")) {
        library_path = path / "libcorenrnmech.so";
    } else if (std::filesystem::is_directory(path) &&
               std::filesystem::is_regular_file(path / "x86_64" / "libcorenrnmech.so")) {
        library_path = path / "x86_64" / "libcorenrnmech.so";
    } else {
        throw std::runtime_error(
            "could not find CoreNEURON mechanism library libcorenrnmech.so under mechanism path: " +
            path.string() + "; run nrnivmodl -coreneuron and mind_nrnivmodl on the MOD directory first");
    }

    return MechanismArtifactPath{
        .library_path = canonical_existing_path(library_path),
    };
}

void promote_this_extension_symbols_to_global() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&coreneuron::nrn_get_mechtype), &info) == 0 ||
        info.dli_fname == nullptr) {
        throw std::runtime_error("failed to resolve MIND_Sim extension for mechanism loading");
    }

    dlerror();
    void* handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        dlerror();
        handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(
            std::string("failed to promote MIND_Sim extension symbols for mechanism loading: ") +
            (error != nullptr ? error : "unknown dlopen error"));
    }
}

[[nodiscard]] std::string normalize_registered_field_name(std::string name,
                                                         std::string_view mechanism) {
    name = trim(name);
    const std::string suffix = "_" + std::string(mechanism);
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

void add_or_update_field(mind_sim::micro::sim::CoreRegisteredMechanism& info,
                         std::string name,
                         mind_sim::micro::sim::CoreMechanismFieldRole role,
                         int array_size = 1,
                         int data_offset = -1,
                         std::optional<double> default_value = std::nullopt,
                         bool is_global = false) {
    name = normalize_registered_field_name(std::move(name), info.name);
    if (name.empty()) {
        return;
    }
    const auto it = std::find_if(info.fields.begin(), info.fields.end(), [&](const auto& field) {
        return field.name == name;
    });
    if (it != info.fields.end()) {
        it->role = role;
        it->array_size = array_size;
        it->data_offset = data_offset;
        if (default_value.has_value()) {
            it->default_value = *default_value;
        }
        it->is_global = it->is_global || is_global;
        return;
    }
    info.fields.push_back(mind_sim::micro::sim::CoreMechanismField{
        .name = std::move(name),
        .role = role,
        .array_size = array_size,
        .data_offset = data_offset,
        .default_value = default_value.value_or(0.0),
        .is_global = is_global,
    });
}

[[nodiscard]] std::vector<neuron::mechanism::detail::data_field_info>
data_fields_from_mechanism_info(const std::string& mechanism, const char** mechanism_info) {
    std::vector<neuron::mechanism::detail::data_field_info> fields;
    if (mechanism.empty() || mechanism_info == nullptr || mechanism_info[2] == nullptr) {
        return fields;
    }
    int section = 0;
    for (int index = 2; mechanism_info[index] != nullptr || section < 3; ++index) {
        if (mechanism_info[index] == nullptr) {
            ++section;
            if (mechanism_info[index + 1] == nullptr) {
                break;
            }
            continue;
        }
        neuron::mechanism::field_role role = neuron::mechanism::field_role::range;
        if (section == 0) {
            role = neuron::mechanism::field_role::parameter;
        } else if (section == 1) {
            role = neuron::mechanism::field_role::assigned;
        } else if (section == 2) {
            role = neuron::mechanism::field_role::state;
        }
        fields.push_back(neuron::mechanism::detail::data_field_info{
            .name = normalize_registered_field_name(mechanism_info[index], mechanism),
            .array_size = 1,
            .role = role,
        });
    }
    return fields;
}

[[nodiscard]] int data_field_scalar_count(
    const std::vector<neuron::mechanism::detail::data_field_info>& fields) {
    int count = 0;
    for (const auto& field : fields) {
        count += std::max(field.array_size, 1);
    }
    return count;
}

void pad_registered_data_fields_to_prop_size(const std::string& mechanism, int psize) {
    auto fields_it = g_neuron_side_fields_by_mechanism.find(mechanism);
    if (fields_it == g_neuron_side_fields_by_mechanism.end() || psize <= 0) {
        return;
    }
    auto& fields = fields_it->second;
    int offset = data_field_scalar_count(fields);
    while (offset < psize) {
        fields.push_back(neuron::mechanism::detail::data_field_info{
            .name = "__mind_internal_data_" + std::to_string(offset),
            .array_size = 1,
            .role = neuron::mechanism::field_role::range,
        });
        ++offset;
    }
}

[[nodiscard]] mind_sim::micro::sim::CoreMechanismFieldRole role_from_registered_field_role(
    neuron::mechanism::field_role role) {
    using Role = mind_sim::micro::sim::CoreMechanismFieldRole;
    switch (role) {
    case neuron::mechanism::field_role::parameter:
        return Role::Parameter;
    case neuron::mechanism::field_role::assigned:
        return Role::Assigned;
    case neuron::mechanism::field_role::state:
        return Role::State;
    case neuron::mechanism::field_role::range:
        return Role::Range;
    }
    return Role::Range;
}

[[nodiscard]] neuron::mechanism::field_role registered_role_for_field(
    const std::string& mechanism,
    const std::string& field,
    neuron::mechanism::field_role default_role) {
    const auto mech_it = g_registered_field_roles_by_mechanism.find(mechanism);
    if (mech_it == g_registered_field_roles_by_mechanism.end()) {
        return default_role;
    }
    const auto field_it = mech_it->second.find(field);
    if (field_it == mech_it->second.end()) {
        return default_role;
    }
    return field_it->second;
}

[[nodiscard]] mind_sim::micro::sim::CoreDParamKind dparam_kind_from_semantic(
    const std::string& semantic) {
    using Kind = mind_sim::micro::sim::CoreDParamKind;
    if (semantic == "area") {
        return Kind::Area;
    }
    if (semantic == "pntproc") {
        return Kind::PointProcess;
    }
    if (semantic == "netsend") {
        return Kind::NetSend;
    }
    if (semantic == "random") {
        return Kind::Random;
    }
    if (semantic == "bbcorepointer") {
        return Kind::BbcorePointer;
    }
    return Kind::Unsupported;
}

[[nodiscard]] const mind_sim::micro::sim::CoreRegisteredMechanism& require_registered_ion_info(
    const std::string& ion) {
    const auto type_it = g_mechanism_type_by_name.find(ion + "_ion");
    if (type_it == g_mechanism_type_by_name.end()) {
        throw std::runtime_error("CoreNEURON ion mechanism is not registered: " + ion + "_ion");
    }
    const int type = type_it->second;
    if (type < 0 || static_cast<std::size_t>(type) >= g_registered_mechanisms.size()) {
        throw std::runtime_error("CoreNEURON ion mechanism type is invalid for: " + ion +
                                 "_ion type=" + std::to_string(type) +
                                 " registered_size=" +
                                 std::to_string(g_registered_mechanisms.size()));
    }
    const auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    if (info.name != ion + "_ion" || info.fields.empty()) {
        throw std::runtime_error("CoreNEURON ion mechanism data layout is incomplete for: " + ion + "_ion");
    }
    return info;
}

[[nodiscard]] std::string infer_ion_target_from_dparam_field(const std::string& field,
                                                            const std::string& ion) {
    const std::string prefix = "_ion_";
    if (!field.starts_with(prefix)) {
        return {};
    }
    const std::string suffix = field.substr(prefix.size());
    if (suffix == "i" + ion) {
        return "i" + ion;
    }
    if (suffix == "di" + ion + "dv") {
        return "di" + ion + "_dv_";
    }
    if (suffix == "d" + ion + "dv") {
        return "di" + ion + "_dv_";
    }
    if (suffix == ion + "i" || suffix == ion + "o") {
        return suffix;
    }
    if (suffix == ion + "_erev") {
        return "e" + ion;
    }
    if (suffix == "e" + ion) {
        return suffix;
    }
    return suffix;
}

[[nodiscard]] std::pair<std::string, int> resolve_registered_ion_field(
    const std::string& ion,
    const std::string& registered_ion_field) {
    if (registered_ion_field.empty()) {
        throw std::runtime_error("missing structured registered ion field for dparam '" +
                                 ion + "_ion'");
    }
    const auto& ion_info = require_registered_ion_info(ion);
    for (std::size_t index = 0; index < ion_info.fields.size(); ++index) {
        if (ion_info.fields[index].name == registered_ion_field) {
            return {registered_ion_field, static_cast<int>(index)};
        }
    }
    throw std::runtime_error("registered ion dparam target field '" +
                             registered_ion_field + "' for '" + ion +
                             "_ion' but no matching ion mechanism field exists");
}

[[nodiscard]] mind_sim::micro::sim::CoreDParamBinding binding_from_semantic(
    const std::string& mechanism,
    int index,
    const std::string& semantic,
    const std::string& registered_field_name = {},
    const std::string& registered_semantic = {},
    const std::string& registered_target = {},
    int ion_conc_style = 0,
    int ion_rev_style = 0,
    bool ion_write_interior = false,
    bool ion_write_exterior = false) {
    using Kind = mind_sim::micro::sim::CoreDParamKind;
    const std::string effective_semantic =
        registered_semantic.empty() ? semantic : registered_semantic;
    if (effective_semantic.starts_with("#") && effective_semantic.ends_with("_ion")) {
        return mind_sim::micro::sim::CoreDParamBinding{
            .mechanism = mechanism,
            .dparam_index = index,
            .kind = Kind::IonStyle,
            .dparam_field = registered_field_name,
            .mechanism_field = registered_target,
            .ion = effective_semantic.substr(1, effective_semantic.size() - 5),
            .ion_field = -1,
            .ion_conc_style = ion_conc_style,
            .ion_rev_style = ion_rev_style,
            .ion_write_interior = ion_write_interior,
            .ion_write_exterior = ion_write_exterior,
        };
    }
    if (effective_semantic.ends_with("_ion")) {
        std::string ion = effective_semantic.substr(0, effective_semantic.size() - 4);
        std::string mechanism_field;
        int ion_field = -1;
        const std::string target =
            registered_target.empty()
                ? infer_ion_target_from_dparam_field(registered_field_name, ion)
                : registered_target;
        if (!target.empty()) {
            auto resolved = resolve_registered_ion_field(ion, target);
            mechanism_field = std::move(resolved.first);
            ion_field = resolved.second;
        }
        return mind_sim::micro::sim::CoreDParamBinding{
            .mechanism = mechanism,
            .dparam_index = index,
            .kind = Kind::IonVariable,
            .dparam_field = registered_field_name,
            .mechanism_field = std::move(mechanism_field),
            .ion = std::move(ion),
            .ion_field = ion_field,
            .ion_conc_style = ion_conc_style,
            .ion_rev_style = ion_rev_style,
            .ion_write_interior = ion_write_interior,
            .ion_write_exterior = ion_write_exterior,
        };
    }
    return mind_sim::micro::sim::CoreDParamBinding{
        .mechanism = mechanism,
        .dparam_index = index,
        .kind = dparam_kind_from_semantic(effective_semantic),
        .dparam_field = registered_field_name,
        .mechanism_field = registered_field_name,
        .ion = effective_semantic,
        .ion_field = -1,
    };
}

void set_dparam_binding(mind_sim::micro::sim::CoreRegisteredMechanism& info,
                        mind_sim::micro::sim::CoreDParamBinding binding) {
    if (binding.dparam_index < 0) {
        return;
    }
    const auto index = static_cast<std::size_t>(binding.dparam_index);
    if (info.dparam_bindings.size() <= index) {
        info.dparam_bindings.resize(index + 1);
    }
    binding.mechanism = info.name;
    info.dparam_bindings[index] = std::move(binding);
}

void rebuild_dparam_bindings_for(mind_sim::micro::sim::CoreRegisteredMechanism& info) {
    if (info.name.empty()) {
        return;
    }
    const auto semantics_it = g_dparam_semantics_by_mechanism.find(info.name);
    if (semantics_it == g_dparam_semantics_by_mechanism.end()) {
        return;
    }
    for (std::size_t index = 0; index < semantics_it->second.size(); ++index) {
        const auto& semantic = semantics_it->second[index];
        if (semantic.empty()) {
            continue;
        }
        std::string registered_field_name;
        std::string registered_semantic;
        std::string registered_target;
        int ion_conc_style = 0;
        int ion_rev_style = 0;
        bool ion_write_interior = false;
        bool ion_write_exterior = false;
        const auto dparam_fields_it = g_neuron_side_dparam_fields_by_mechanism.find(info.name);
        if (dparam_fields_it != g_neuron_side_dparam_fields_by_mechanism.end() &&
            index < dparam_fields_it->second.size()) {
            const auto& field = dparam_fields_it->second[index];
            registered_field_name = field.name;
            registered_semantic = field.semantic;
            registered_target = field.target;
            ion_conc_style = field.ion_conc_style;
            ion_rev_style = field.ion_rev_style;
            ion_write_interior = field.ion_write_interior;
            ion_write_exterior = field.ion_write_exterior;
        }
        set_dparam_binding(
            info,
            binding_from_semantic(info.name,
                                  static_cast<int>(index),
                                  semantic,
                                  registered_field_name,
                                  registered_semantic,
                                  registered_target,
                                  ion_conc_style,
                                  ion_rev_style,
                                  ion_write_interior,
                                  ion_write_exterior));
    }
}

void apply_registered_data_fields(mind_sim::micro::sim::CoreRegisteredMechanism& info) {
    if (info.type < 0 || info.name.empty()) {
        return;
    }
    const auto fields_it = g_neuron_side_fields_by_mechanism.find(info.name);
    if (fields_it == g_neuron_side_fields_by_mechanism.end()) {
        return;
    }
    std::vector<mind_sim::micro::sim::CoreMechanismField> ordered_fields;
    ordered_fields.reserve(fields_it->second.size());
    const auto defaults_it = g_parameter_defaults_by_mechanism.find(info.name);
    const auto* defaults =
        defaults_it == g_parameter_defaults_by_mechanism.end() ? nullptr : &defaults_it->second;
    std::vector<mind_sim::micro::sim::CoreMechanismField> global_fields;
    for (const auto& existing : info.fields) {
        if (existing.is_global) {
            global_fields.push_back(existing);
        }
    }
    std::size_t default_index = 0;
    int data_offset = 0;
    for (const auto& field : fields_it->second) {
        std::string name = normalize_registered_field_name(field.name, info.name);
        const auto role =
            role_from_registered_field_role(registered_role_for_field(info.name, name, field.role));
        double default_value = 0.0;
        if (role == mind_sim::micro::sim::CoreMechanismFieldRole::Parameter) {
            if (defaults != nullptr && default_index < defaults->size()) {
                default_value = (*defaults)[default_index];
            }
            ++default_index;
        }
        auto existing = std::find_if(info.fields.begin(), info.fields.end(), [&](const auto& item) {
            return item.name == name;
        });
        if (existing != info.fields.end()) {
            ordered_fields.push_back(*existing);
            ordered_fields.back().name = std::move(name);
            ordered_fields.back().role = role;
            ordered_fields.back().array_size = field.array_size;
            ordered_fields.back().data_offset = data_offset;
            if (!ordered_fields.back().is_global) {
                ordered_fields.back().default_value = default_value;
            }
            data_offset += field.array_size;
            continue;
        }
        ordered_fields.push_back(mind_sim::micro::sim::CoreMechanismField{
            .name = std::move(name),
            .role = role,
            .array_size = field.array_size,
            .data_offset = data_offset,
            .default_value = default_value,
            .is_global = false,
        });
        data_offset += field.array_size;
    }
    for (auto& field : global_fields) {
        const auto duplicate = std::find_if(ordered_fields.begin(), ordered_fields.end(), [&](const auto& item) {
            return item.name == field.name;
        });
        if (duplicate != ordered_fields.end()) {
            duplicate->is_global = true;
            duplicate->default_value = field.default_value;
            continue;
        }
        field.data_offset = -1;
        field.is_global = true;
        ordered_fields.push_back(std::move(field));
    }
    info.fields = std::move(ordered_fields);
}

void apply_registered_data_fields() {
    for (auto& info : g_registered_mechanisms) {
        apply_registered_data_fields(info);
    }
}

void rebuild_dparam_binding_cache() {
    apply_registered_data_fields();
    g_dparam_bindings.clear();
    for (const auto& info : g_registered_mechanisms) {
        if (info.type < 0 || info.name.empty()) {
            continue;
        }
        for (const auto& binding : info.dparam_bindings) {
            if (binding.dparam_index >= 0) {
                g_dparam_bindings.push_back(binding);
            }
        }
    }
}

void mark_dparam_binding_cache_dirty() noexcept {
    g_dparam_binding_cache_dirty = true;
}

void ensure_dparam_binding_cache_current() {
    if (!g_dparam_binding_cache_dirty) {
        return;
    }
    rebuild_dparam_binding_cache();
    g_dparam_binding_cache_dirty = false;
}

void register_core_base_once() {
    if (g_core_base_registered) {
        return;
    }

    initialize_fixed_core_slots();
    coreneuron::initnrn();
    coreneuron::ion_reg("na", 1.0);
    coreneuron::ion_reg("k", 1.0);
    coreneuron::ion_reg("ca", 2.0);
    for (int i = 0; coreneuron::mechanism[i] != nullptr; ++i) {
        (*coreneuron::mechanism[i])();
    }
    coreneuron::modl_reg();
    g_core_base_registered = true;
    ensure_dparam_binding_cache_current();
}

void call_loaded_modl_reg_once() {
    bool registered_any = false;
    for (auto& library : g_loaded_libraries) {
        if (library.registered) {
            continue;
        }
        if (library.modl_reg == nullptr) {
            throw std::runtime_error("loaded CoreNEURON mechanism library has no modl_reg: " +
                                     library.library_path.string());
        }
        struct ActiveLibraryScope {
            explicit ActiveLibraryScope(LoadedMechanismLibrary* library) {
                g_active_loaded_library = library;
            }
            ~ActiveLibraryScope() {
                g_active_loaded_library = nullptr;
            }
        } active_scope{&library};
        library.modl_reg();
        library.registered = true;
        registered_any = true;
    }
    if (registered_any) {
        ensure_dparam_binding_cache_current();
    }
}

[[nodiscard]] const std::string& registered_name_from_type(int type) {
    if (type < 0 || static_cast<std::size_t>(type) >= g_registered_mechanisms.size() ||
        g_registered_mechanisms[static_cast<std::size_t>(type)].name.empty()) {
        throw std::runtime_error("CoreNEURON mechanism type has no registered name: " +
                                 std::to_string(type));
    }
    return g_registered_mechanisms[static_cast<std::size_t>(type)].name;
}

void validate_generated_data_fields_registered_for(const LoadedMechanismLibrary& library) {
    for (const int type : library.mechanism_types) {
        if (type < 0 || static_cast<std::size_t>(type) >= g_registered_mechanisms.size()) {
            throw std::runtime_error("data-field validation saw invalid CoreNEURON mechanism type " +
                                     std::to_string(type));
        }
        const auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
        if (info.name.empty()) {
            throw std::runtime_error("data-field validation saw an unnamed CoreNEURON mechanism type " +
                                     std::to_string(type));
        }
        const auto fields_it = g_neuron_side_fields_by_mechanism.find(info.name);
        if (fields_it == g_neuron_side_fields_by_mechanism.end() || fields_it->second.empty()) {
            throw std::runtime_error(
                "CoreNEURON generated registration did not provide data fields for mechanism '" +
                info.name + "' from " + library.library_path.string());
        }
    }
}

void validate_loaded_library_generated_fields(LoadedMechanismLibrary& library) {
    mark_dparam_binding_cache_dirty();
    ensure_dparam_binding_cache_current();
    validate_generated_data_fields_registered_for(library);
}

}  // namespace

namespace coreneuron {

void hoc_register_parm_default(int type, const std::vector<double>* defaults) {
    g_parameter_defaults_by_mechanism[registered_name_from_type(type)] = *defaults;
}

}  // namespace coreneuron

namespace neuron::mechanism::detail {

void register_data_fields(
    int type,
    const std::vector<data_field_info>& param_info,
    const std::vector<dparam_field_info>& dparam_info) {
    const auto& mechanism = registered_name_from_type(type);
    auto& fields = g_neuron_side_fields_by_mechanism[mechanism];
    fields.clear();
    fields.reserve(param_info.size());
    std::vector<int> array_dims;
    array_dims.reserve(param_info.size());
    for (const auto& field : param_info) {
        if (!field.name.empty()) {
            fields.push_back(field);
            array_dims.push_back(field.array_size);
        }
    }
    coreneuron::corenrn.get_array_dims()[type] = std::move(array_dims);
    auto& target = g_neuron_side_dparam_fields_by_mechanism[mechanism];
    target.clear();
    target.reserve(dparam_info.size());
    target.insert(target.end(), dparam_info.begin(), dparam_info.end());
    apply_registered_data_fields(g_registered_mechanisms[static_cast<std::size_t>(type)]);
    rebuild_dparam_bindings_for(g_registered_mechanisms[static_cast<std::size_t>(type)]);
    mark_dparam_binding_cache_dirty();
}

}  // namespace neuron::mechanism::detail

namespace mind_sim::micro::sim {

void load_core_mechanism_library(const std::filesystem::path& path) {
    const auto artifact = resolve_mechanism_artifact_path(path);
    if (g_loaded_library_paths.contains(artifact.library_path)) {
        ensure_core_mechanisms_registered();
        for (auto& library : g_loaded_libraries) {
            if (library.library_path == artifact.library_path) {
                validate_loaded_library_generated_fields(library);
                break;
            }
        }
        return;
    }

    promote_this_extension_symbols_to_global();

    dlerror();
    void* handle = dlopen(artifact.library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("failed to load CoreNEURON mechanism library '" +
                                 artifact.library_path.string() + "': " +
                                 (error != nullptr ? error : "unknown dlopen error"));
    }

    dlerror();
    auto* symbol = dlsym(handle, "_ZN10coreneuron8modl_regEv");
    if (symbol == nullptr) {
        dlerror();
        symbol = dlsym(handle, "modl_reg");
    }
    if (symbol == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("loaded CoreNEURON mechanism library has no modl_reg symbol '" +
                                 artifact.library_path.string() + "': " +
                                 (error != nullptr ? error : "unknown dlsym error"));
    }

    g_loaded_library_paths.insert(artifact.library_path);
    g_loaded_libraries.push_back(LoadedMechanismLibrary{
        .library_path = artifact.library_path,
        .handle = handle,
        .modl_reg = reinterpret_cast<void (*)()>(symbol),
        .registered = false,
    });

    ensure_core_mechanisms_registered();
    validate_loaded_library_generated_fields(g_loaded_libraries.back());
}

void ensure_core_mechanisms_registered() {
    register_core_base_once();
    call_loaded_modl_reg_once();
    ensure_dparam_binding_cache_current();
}

int core_mechanism_type(const std::string& name) {
    const auto it = g_mechanism_type_by_name.find(name);
    if (it == g_mechanism_type_by_name.end()) {
        throw std::runtime_error("CoreNEURON mechanism type is not registered: " + name);
    }
    return it->second;
}

const char* core_mechanism_name(int type) noexcept {
    return coreneuron::nrn_get_mechname(type);
}

int core_mechanism_capacity() noexcept {
    return static_cast<int>(g_mechanism_name_by_type.size());
}

int core_mechanism_order_rank(int type) noexcept {
    if (type < 0 || static_cast<std::size_t>(type) >= g_memb_order_rank_by_type.size()) {
        return static_cast<int>(g_memb_order_rank_by_type.size()) + type;
    }
    const int rank = g_memb_order_rank_by_type[static_cast<std::size_t>(type)];
    return rank >= 0 ? rank : static_cast<int>(g_memb_order_rank_by_type.size()) + type;
}

const std::vector<CoreDParamBinding>& core_dparam_bindings() {
    ensure_core_mechanisms_registered();
    return g_dparam_bindings;
}

const std::vector<CoreRegisteredMechanism>& core_registered_mechanisms() {
    ensure_core_mechanisms_registered();
    return g_registered_mechanisms;
}

int core_ion_register(const std::string& ion, double charge) {
    register_core_base_once();
    const int type = coreneuron::ion_register(ion.c_str(), charge);
    ensure_dparam_binding_cache_current();
    return type;
}

double core_ion_charge(const std::string& ion_mechanism) {
    register_core_base_once();
    const int type = core_mechanism_type(ion_mechanism);
    if (coreneuron::nrn_is_ion(type) == 0) {
        throw std::runtime_error(ion_mechanism + " is not an ion mechanism");
    }
    return coreneuron::nrn_ion_charge(type);
}

double core_global_scalar(const std::string& name) {
    const auto it = g_global_scalars_by_name.find(name);
    if (it == g_global_scalars_by_name.end() || it->second == nullptr) {
        throw std::runtime_error("CoreNEURON global scalar is not registered: " + name);
    }
    return *it->second;
}

namespace nrn_registration_mirror {

void record_field_roles_from_mechanism_info(const std::string& mechanism,
                                            const char** mechanism_info) {
    if (mechanism.empty() || mechanism_info == nullptr || mechanism_info[2] == nullptr) {
        return;
    }

    auto& roles = g_registered_field_roles_by_mechanism[mechanism];
    roles.clear();
    int section = 0;
    for (int index = 2; mechanism_info[index] != nullptr || section < 3; ++index) {
        if (mechanism_info[index] == nullptr) {
            ++section;
            if (mechanism_info[index + 1] == nullptr) {
                break;
            }
            continue;
        }
        neuron::mechanism::field_role role = neuron::mechanism::field_role::range;
        if (section == 0) {
            role = neuron::mechanism::field_role::parameter;
        } else if (section == 1) {
            role = neuron::mechanism::field_role::assigned;
        } else if (section == 2) {
            role = neuron::mechanism::field_role::state;
        }
        roles[normalize_registered_field_name(mechanism_info[index], mechanism)] = role;
    }
}

void mechanism_registered(int type, const char** mechanism_info) {
    if (type < 0 || mechanism_info == nullptr || mechanism_info[1] == nullptr) {
        return;
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    info.type = type;
    info.name = mechanism_info[1];
    g_mechanism_type_by_name[info.name] = type;
    if (g_mechanism_name_by_type.size() <= static_cast<std::size_t>(type)) {
        g_mechanism_name_by_type.resize(static_cast<std::size_t>(type) + 1);
    }
    g_mechanism_name_by_type[static_cast<std::size_t>(type)] = info.name;
    g_recent_registered_type = type;
    record_field_roles_from_mechanism_info(info.name, mechanism_info);
    if (!g_neuron_side_fields_by_mechanism.contains(info.name)) {
        g_neuron_side_fields_by_mechanism[info.name] =
            data_fields_from_mechanism_info(info.name, mechanism_info);
        apply_registered_data_fields(info);
    }
    if (g_active_loaded_library != nullptr) {
        auto& types = g_active_loaded_library->mechanism_types;
        if (std::find(types.begin(), types.end(), type) == types.end()) {
            types.push_back(type);
        }
    }
}

void writes_concentration(int type) {
    if (type < 0) {
        return;
    }
    ensure_memb_order_slot(type);
    auto it = std::find(g_memb_order.begin(), g_memb_order.end(), type);
    if (it != g_memb_order.end()) {
        g_memb_order.erase(it);
    }

    const auto insert_rank = static_cast<std::size_t>(
        std::clamp(g_memb_order_lastion, 0, static_cast<int>(g_memb_order.size())));
    g_memb_order.insert(g_memb_order.begin() + insert_rank, type);
    if (coreneuron::nrn_is_ion(type) != 0) {
        ++g_memb_order_lastion;
    }
    rebuild_memb_order_ranks();
}

void point_mechanism(int type) {
    if (type < 0) {
        return;
    }
    ensure_type_slot(type);
    g_registered_mechanisms[static_cast<std::size_t>(type)].is_point = true;
}

void artificial_cell(int type) {
    if (type < 0) {
        return;
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    info.is_point = true;
    info.is_artificial = true;
}

void net_receive(int type, int weight_count) {
    if (type < 0) {
        return;
    }
    if (weight_count < 0) {
        throw std::runtime_error("CoreNEURON NET_RECEIVE weight count must be non-negative");
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    info.has_net_receive = true;
    info.net_receive_weight_count = weight_count == 0 ? 1 : weight_count;
}

void prop_size(int type, int psize, int dpsize) {
    if (type < 0) {
        return;
    }
    if (psize < 0 || dpsize < 0) {
        throw std::runtime_error("CoreNEURON mechanism property sizes must be non-negative");
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    info.parameter_size = psize;
    info.dparam_size = dpsize;
    pad_registered_data_fields_to_prop_size(info.name, psize);
    apply_registered_data_fields(info);
    if (info.dparam_bindings.size() < static_cast<std::size_t>(dpsize)) {
        info.dparam_bindings.resize(static_cast<std::size_t>(dpsize));
    }
}

void dparam_semantic(int type, int index, const char* semantic) {
    if (type < 0 || index < 0 || semantic == nullptr) {
        return;
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    auto& stored_semantics = g_dparam_semantics_by_mechanism[info.name];
    if (stored_semantics.size() <= static_cast<std::size_t>(index)) {
        stored_semantics.resize(static_cast<std::size_t>(index) + 1);
    }
    stored_semantics[static_cast<std::size_t>(index)] = semantic;
    std::string registered_field_name;
    std::string registered_semantic;
    std::string registered_target;
    int ion_conc_style = 0;
    int ion_rev_style = 0;
    bool ion_write_interior = false;
    bool ion_write_exterior = false;
    const auto dparam_fields_it = g_neuron_side_dparam_fields_by_mechanism.find(info.name);
    if (dparam_fields_it != g_neuron_side_dparam_fields_by_mechanism.end() &&
        static_cast<std::size_t>(index) < dparam_fields_it->second.size()) {
        const auto& field = dparam_fields_it->second[static_cast<std::size_t>(index)];
        registered_field_name = field.name;
        registered_semantic = field.semantic;
        registered_target = field.target;
        ion_conc_style = field.ion_conc_style;
        ion_rev_style = field.ion_rev_style;
        ion_write_interior = field.ion_write_interior;
        ion_write_exterior = field.ion_write_exterior;
    }
    set_dparam_binding(
        info,
        binding_from_semantic(info.name,
                              index,
                              semantic,
                              registered_field_name,
                              registered_semantic,
                              registered_target,
                              ion_conc_style,
                              ion_rev_style,
                              ion_write_interior,
                              ion_write_exterior));
    mark_dparam_binding_cache_dirty();
}

void global_scalar(const char* name, double* value) {
    if (name == nullptr || value == nullptr) {
        return;
    }
    g_global_scalars_by_name[name] = value;
}

void global_scalar_field(int type, const char* field_name, const char* hoc_name, double* value) {
    if (type < 0 || field_name == nullptr || hoc_name == nullptr || value == nullptr) {
        return;
    }
    ensure_type_slot(type);
    auto& info = g_registered_mechanisms[static_cast<std::size_t>(type)];
    if (info.name.empty()) {
        return;
    }
    std::string field = trim(field_name);
    if (field.empty()) {
        return;
    }
    add_or_update_field(info,
                        field,
                        CoreMechanismFieldRole::Parameter,
                        1,
                        -1,
                        *value,
                        true);
    g_global_scalars_by_name[hoc_name] = value;
    g_global_scalars[info.name][std::move(field)] = value;
}

}  // namespace nrn_registration_mirror

void core_set_global_parameter(const std::string& mechanism,
                               const std::string& field,
                               double value) {
    const auto mech_it = g_global_scalars.find(mechanism);
    if (mech_it == g_global_scalars.end()) {
        throw std::runtime_error("mechanism '" + mechanism +
                                 "' has no registered CoreNEURON global scalars");
    }
    const auto field_it = mech_it->second.find(field);
    if (field_it == mech_it->second.end() || field_it->second == nullptr) {
        throw std::runtime_error("mechanism '" + mechanism +
                                 "' has no registered CoreNEURON global scalar '" + field + "'");
    }
    *field_it->second = value;
}

void core_set_global_scalar(const std::string& name, double value) {
    const auto it = g_global_scalars_by_name.find(name);
    if (it == g_global_scalars_by_name.end() || it->second == nullptr) {
        throw std::runtime_error("CoreNEURON global scalar is not registered: " + name);
    }
    *it->second = value;
}

void core_verify_ion_charges_defined() {
    register_core_base_once();
    coreneuron::nrn_verify_ion_charge_defined();
}

}  // namespace mind_sim::micro::sim
