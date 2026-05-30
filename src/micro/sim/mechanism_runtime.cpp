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
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <sstream>
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
std::unordered_map<std::string, std::vector<neuron::mechanism::detail::data_field_info>>
    g_neuron_side_fields_by_mechanism;
std::unordered_map<std::string, std::vector<neuron::mechanism::detail::dparam_field_info>>
    g_neuron_side_dparam_fields_by_mechanism;
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
};

std::vector<LoadedMechanismLibrary> g_loaded_libraries;
std::set<std::filesystem::path> g_loaded_library_paths;

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

[[nodiscard]] bool has_mod_sources(const std::filesystem::path& dir) {
    if (!std::filesystem::is_directory(dir)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mod") {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string shell_quote(const std::filesystem::path& path) {
    std::string quoted = "'";
    for (char c : path.string()) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

[[nodiscard]] std::string shell_quote(std::string_view value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

void append_file_signature(std::ostringstream& out, const std::filesystem::path& path) {
    std::error_code ec;
    out << path.string() << '\n';
    out << std::filesystem::file_size(path, ec) << '\n';
    ec.clear();
    out << std::filesystem::last_write_time(path, ec).time_since_epoch().count() << '\n';
}

[[nodiscard]] std::string mechanism_source_signature(const std::filesystem::path& path) {
    std::ostringstream out;
    out << path.string() << '\n';
    out << MIND_SIM_CXX_COMPILER << '\n';
    out << MIND_SIM_MODCC_BACKEND << '\n';
    out << MIND_SIM_MODCC_CXX_FLAGS << '\n';
    out << MIND_SIM_MODCC_LINK_FLAGS << '\n';
    append_file_signature(out, MIND_SIM_MODCC);
    append_file_signature(out, MIND_SIM_NMODL_EXECUTABLE);
    append_file_signature(out,
                          std::filesystem::path{MIND_SIM_NMODL_SOURCE_DIR} /
                              "src/nmodl/codegen/codegen_coreneuron_cpp_visitor.cpp");
    append_file_signature(out,
                          std::filesystem::path{MIND_SIM_NMODL_SOURCE_DIR} /
                              "src/nmodl/codegen/codegen_info.hpp");
    append_file_signature(out,
                          std::filesystem::path{MIND_SIM_MECHANISM_INCLUDE_DIR} /
                              "coreneuron/mechanism/neuron_registration.hpp");
    if (std::filesystem::is_regular_file(path)) {
        out << std::filesystem::file_size(path) << '\n'
            << std::filesystem::last_write_time(path).time_since_epoch().count() << '\n';
        return out.str();
    }
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".mod") {
            continue;
        }
        out << entry.path().filename().string() << '\n'
            << entry.file_size() << '\n'
            << entry.last_write_time().time_since_epoch().count() << '\n';
    }
    return out.str();
}

[[nodiscard]] std::filesystem::path compile_mod_sources(const std::filesystem::path& path) {
    const auto signature = mechanism_source_signature(path);
    const auto hash = std::hash<std::string>{}(signature);
    std::ostringstream name;
    name << "mind_sim_mechanisms_" << std::hex << hash;
    const auto build_dir = std::filesystem::temp_directory_path() / name.str();
    const auto library = build_dir / "libmindcorenrnmech.so";
    if (std::filesystem::is_regular_file(library)) {
        return canonical_existing_path(library);
    }
    std::filesystem::create_directories(build_dir);

    std::string command = shell_quote(std::string_view{MIND_SIM_PYTHON_EXECUTABLE}) + " " +
                          shell_quote(std::string_view{MIND_SIM_MODCC}) + " --nmodl " +
                          shell_quote(std::string_view{MIND_SIM_NMODL_EXECUTABLE}) + " --cxx " +
                          shell_quote(std::string_view{MIND_SIM_CXX_COMPILER}) + " --output " +
                          shell_quote(build_dir) + " --backend " +
                          shell_quote(std::string_view{MIND_SIM_MODCC_BACKEND}) +
                          " --cxx-flag=" +
                          shell_quote(std::string_view{MIND_SIM_MODCC_CXX_FLAGS}) +
                          " --link-flag=" +
                          shell_quote(std::string_view{MIND_SIM_MODCC_LINK_FLAGS}) +
                          " --include " +
                          shell_quote(std::string_view{MIND_SIM_MECHANISM_INCLUDE_DIR}) +
                          " --define CORENEURON_BUILD"
                          " --define CORENRN_BUILD=1"
                          " --define VECTORIZE=1"
                          " --define HAVE_MALLOC_H"
                          " --define EIGEN_DONT_PARALLELIZE"
                          " --define LAYOUT=0"
                          " --define ENABLE_SPLAYTREE_QUEUING"
                          " --define NRNMPI=0"
                          " --define NRN_MULTISEND=0"
                          " --define DISABLE_HOC_EXP"
                          " --define NET_RECEIVE_BUFFERING=0"
                          " --define NRN_PRCELLSTATE=0"
#ifdef MIND_SIM_ENABLE_GPU
                          " --define CORENEURON_ENABLE_GPU"
#endif
                          " " +
                          shell_quote(path);
    const int status = std::system(command.c_str());
    if (status != 0) {
        throw std::runtime_error("MIND_Sim MOD compilation failed for: " + path.string());
    }
    return canonical_existing_path(library);
}

[[nodiscard]] MechanismArtifactPath resolve_mechanism_artifact_path(
    const std::filesystem::path& requested_path) {
    const auto path = canonical_existing_path(requested_path);
    std::filesystem::path library_path;
    if (std::filesystem::is_regular_file(path) && path.extension() == ".mod") {
        library_path = compile_mod_sources(path);
    } else if (std::filesystem::is_regular_file(path) && path.filename() == "libmindcorenrnmech.so") {
        library_path = path;
    } else if (has_mod_sources(path)) {
        library_path = compile_mod_sources(path);
    } else if (std::filesystem::is_directory(path) &&
               std::filesystem::is_regular_file(path / "libmindcorenrnmech.so")) {
        library_path = path / "libmindcorenrnmech.so";
    } else {
        throw std::runtime_error(
            "could not find libmindcorenrnmech.so or MOD sources under mechanism path: " +
            path.string());
    }

    const auto canonical_library = canonical_existing_path(library_path);
    return MechanismArtifactPath{
        .library_path = canonical_library,
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
    (void)mechanism;
    name = trim(name);
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
        throw std::runtime_error("CoreNEURON ion mechanism metadata is incomplete for: " + ion + "_ion");
    }
    return info;
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
        if (!registered_target.empty()) {
            auto resolved = resolve_registered_ion_field(ion, registered_target);
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
        const auto role = role_from_registered_field_role(field.role);
        double default_value = 0.0;
        if (role == mind_sim::micro::sim::CoreMechanismFieldRole::Parameter &&
            defaults != nullptr &&
            default_index < defaults->size()) {
            default_value = (*defaults)[default_index++];
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
    mark_dparam_binding_cache_dirty();
}

}  // namespace neuron::mechanism::detail

namespace mind_sim::micro::sim {

void load_core_mechanism_library(const std::filesystem::path& path) {
    const auto artifact = resolve_mechanism_artifact_path(path);
    if (g_loaded_library_paths.contains(artifact.library_path)) {
        ensure_core_mechanisms_registered();
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
