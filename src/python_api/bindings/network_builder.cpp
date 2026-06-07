#include "python_api/bindings/bindings.hpp"
#include "python_api/bindings/network_builder.hpp"
#include "utils/dynamic_library.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace mind_sim::python_api::bindings {

namespace {

bool is_shared_library_suffix(const std::filesystem::path& path) {
    const auto suffix = path.extension().string();
    return suffix == ".so" || suffix == ".dylib" || suffix == ".dll";
}

std::vector<std::string> library_names_for_mod(const std::string& name) {
    return {
        name + ".so",
        "lib" + name + ".so",
        "libmind_sim_mod_" + name + ".so",
    };
}

std::vector<std::filesystem::path> library_dirs_for_source(const std::filesystem::path& source_dir) {
    std::vector<std::filesystem::path> out;
    const auto x86_dir = source_dir / "x86_64";
    if (std::filesystem::exists(x86_dir)) {
        out.push_back(x86_dir);
    }
    if (source_dir.filename() == "x86_64" && std::filesystem::exists(source_dir)) {
        out.push_back(source_dir);
    }
    return out;
}

std::string canonical_string(const std::filesystem::path& path) {
    return std::filesystem::canonical(path).string();
}

const mind_sim::mod::AbiRuleDescriptor& descriptor_for_library(
    const mind_sim::utils::DynamicLibrary& library) {
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mod::DescriptorFn>(library.symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (descriptor == nullptr) {
        throw std::runtime_error("MOD library has a null descriptor");
    }
    if (descriptor->abi_version != mind_sim::mod::kModAbiVersion) {
        throw std::runtime_error("MOD library ABI version mismatch");
    }
    return *descriptor;
}

template <typename Rule>
std::vector<double> values_with_defaults(const std::vector<std::string>& names,
                                         const std::vector<double>& defaults,
                                         const std::unordered_map<std::string, double>& overrides,
                                         const std::string& what) {
    if (names.size() != defaults.size()) {
        throw std::runtime_error(what + " descriptor names/defaults size mismatch");
    }
    std::unordered_map<std::string, double> values;
    values.reserve(names.size());
    for (std::size_t index = 0; index < names.size(); ++index) {
        values.emplace(names[index], defaults[index]);
    }
    for (const auto& [name, value]: overrides) {
        if (values.find(name) == values.end()) {
            throw std::runtime_error(what + " unknown value: " + name);
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error(what + " value must be finite: " + name);
        }
        values[name] = value;
    }
    std::vector<double> out;
    out.reserve(names.size());
    for (const auto& name: names) {
        out.push_back(values.at(name));
    }
    return out;
}

std::vector<double> region_state_values(
    const mind_sim::macro::sim::RegionRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::macro::sim::RegionRule>(
        rule.state_names(),
        rule.state_defaults(),
        overrides,
        rule.name() + " state");
}

std::vector<double> region_param_values(
    const mind_sim::macro::sim::RegionRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::macro::sim::RegionRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> macro_to_macro_param_values(
    const mind_sim::macro::sim::MacroToMacroRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::macro::sim::MacroToMacroRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> micro_input_state_values(
    const mind_sim::cosim::bridge::MicroInputRule& rule,
    const std::unordered_map<std::string, double>& overrides,
    int source_count) {
    if (source_count < 0) {
        throw std::runtime_error(rule.name() + " source count must be non-negative");
    }
    const auto base = values_with_defaults<mind_sim::cosim::bridge::MicroInputRule>(
        rule.state_names(),
        rule.state_defaults(),
        overrides,
        rule.name() + " state");
    std::vector<double> out(static_cast<std::size_t>(source_count) * base.size(), 0.0);
    for (std::size_t state = 0; state < base.size(); ++state) {
        for (int source = 0; source < source_count; ++source) {
            out[(state * static_cast<std::size_t>(source_count)) + static_cast<std::size_t>(source)] =
                base[state];
        }
    }
    return out;
}

std::vector<double> micro_input_param_values(
    const mind_sim::cosim::bridge::MicroInputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::bridge::MicroInputRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> micro_output_state_values(
    const mind_sim::cosim::bridge::MicroOutputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::bridge::MicroOutputRule>(
        rule.state_names(),
        rule.state_defaults(),
        overrides,
        rule.name() + " state");
}

std::vector<double> micro_output_param_values(
    const mind_sim::cosim::bridge::MicroOutputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::bridge::MicroOutputRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> field_param_values(
    const mind_sim::macro::sim::NeuralFieldRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::macro::sim::NeuralFieldRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> node_values(double value, int node_count) {
    return std::vector<double>(static_cast<std::size_t>(node_count), value);
}

std::vector<double> field_state_values(const mind_sim::macro::sim::NeuralFieldRule& rule,
                                       const std::unordered_map<std::string, double>& overrides,
                                       int node_count) {
    auto values = values_with_defaults<mind_sim::macro::sim::NeuralFieldRule>(
        rule.state_names(),
        rule.state_defaults(),
        overrides,
        rule.name() + " state");
    std::vector<double> out;
    out.reserve(values.size() * static_cast<std::size_t>(node_count));
    for (double value: values) {
        auto expanded = node_values(value, node_count);
        out.insert(out.end(), expanded.begin(), expanded.end());
    }
    return out;
}

void add_name(std::vector<std::string>& names, const std::string& name) {
    if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
    }
}

void require_names(const std::string& context,
                   const std::vector<std::string>& names,
                   const std::unordered_set<std::string>& available) {
    std::vector<std::string> missing;
    for (const auto& name: names) {
        if (available.find(name) == available.end()) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        std::string message = context + " missing names:";
        for (const auto& name: missing) {
            message += " " + name;
        }
        throw std::runtime_error(message);
    }
}

std::vector<int> offset_map(const std::vector<std::string>& schema,
                            const std::vector<std::string>& names,
                            int width) {
    std::vector<int> offsets;
    offsets.reserve(names.size());
    for (const auto& name: names) {
        const auto found = std::find(schema.begin(), schema.end(), name);
        if (found == schema.end()) {
            throw std::runtime_error("schema is missing field: " + name);
        }
        offsets.push_back(static_cast<int>(found - schema.begin()) * width);
    }
    return offsets;
}

std::vector<int> output_indices(const std::vector<std::string>& outputs,
                                  const std::vector<std::string>& recorded) {
    std::vector<int> out;
    out.reserve(recorded.size());
    for (const auto& name: recorded) {
        const auto found = std::find(outputs.begin(), outputs.end(), name);
        if (found == outputs.end()) {
            throw std::runtime_error("record output names are not declared by the network: " + name);
        }
        out.push_back(static_cast<int>(found - outputs.begin()));
    }
    return out;
}

void validate_runtime_rule_path(const std::string& library_path) {
    if (library_path.size() >= 4 &&
        library_path.substr(library_path.size() - 4) == ".mod") {
        throw std::runtime_error(
            "rule .mod files must be compiled to a shared library before runtime loading");
    }
}

bool looks_like_library_path(const std::string& value) {
    const std::filesystem::path path(value);
    return path.has_parent_path() || is_shared_library_suffix(path);
}

void require_positive_finite(double value, const char* what) {
    if (value <= 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + " must be positive and finite");
    }
}

std::vector<int> spike_input_runtime_indices(Sim& micro, const std::vector<int>& source_ids) {
    if (source_ids.empty()) {
        throw std::runtime_error("macro2micro requires at least one source");
    }
    std::vector<int> indices;
    indices.reserve(source_ids.size());
    for (std::size_t i = 0; i < source_ids.size(); ++i) {
        const int runtime_index = micro.model.spike_input_runtime_index(source_ids[i]);
        if (runtime_index < 0) {
            throw std::runtime_error("macro2micro source has no runtime index");
        }
        indices.push_back(runtime_index);
    }
    return indices;
}

void attach_micro_recorders(mind_sim::macro::frontend::Network& network, Sim& micro) {
    micro.prune_recorders();
    for (auto& weak: micro.recorders) {
        auto recorder = weak.lock();
        if (!recorder) {
            continue;
        }
        if (recorder->records_time) {
            network.record_micro_time(&recorder->samples);
        } else if (recorder->records_var) {
            network.record_micro(
                micro.model.variable_pointer(recorder->ref),
                &recorder->samples);
        }
    }
}

}  // namespace

NetworkBuilder::NetworkBuilder(mind_sim::macro::frontend::Connectivity connectivity)
    : connectivity_(std::move(connectivity)) {
    dc_inputs_.resize(static_cast<std::size_t>(connectivity_.roi_count()));
}

mind_sim::macro::frontend::ROI NetworkBuilder::roi(int index) const {
    validate_roi_index(index, "ROI");
    return connectivity_.rois()[static_cast<std::size_t>(index)];
}

mind_sim::macro::frontend::ROI NetworkBuilder::roi(const std::string& label) const {
    return roi(roi_index(label));
}

std::vector<mind_sim::macro::frontend::ROI> NetworkBuilder::rois() const {
    return connectivity_.rois();
}

int NetworkBuilder::roi_count() const {
    return connectivity_.roi_count();
}

double NetworkBuilder::min_positive_delay() const {
    return connectivity_.min_positive_delay();
}

void NetworkBuilder::record_rois(std::vector<int> roi_indices) {
    for (int roi_value: roi_indices) {
        validate_roi_index(roi_value, "record ROI");
    }
    recorded_rois_ = std::move(roi_indices);
}

void NetworkBuilder::record_all_rois() {
    recorded_rois_.reset();
}

void NetworkBuilder::record_outputs(std::vector<std::string> output_names) {
    recorded_outputs_ = std::move(output_names);
}

void NetworkBuilder::record_all_outputs() {
    recorded_outputs_.reset();
}

void NetworkBuilder::set_dt(double dt) {
    require_positive_finite(dt, "macro dt");
    dt_ = dt;
}

void NetworkBuilder::set_exchange_window(double exchange_window) {
    require_positive_finite(exchange_window, "exchange_window");
    exchange_window_ = exchange_window;
}

void NetworkBuilder::load_mech(std::string directory) {
    const std::filesystem::path source_dir(directory);
    if (!std::filesystem::exists(source_dir)) {
        throw std::runtime_error("MOD directory does not exist: " + source_dir.string());
    }
    if (!std::filesystem::is_directory(source_dir)) {
        throw std::runtime_error("MOD path is not a directory: " + source_dir.string());
    }

    std::vector<std::string> mod_names;
    for (const auto& entry: std::filesystem::directory_iterator(source_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mod") {
            mod_names.push_back(entry.path().stem().string());
        }
    }
    std::sort(mod_names.begin(), mod_names.end());
    mod_names.erase(std::unique(mod_names.begin(), mod_names.end()), mod_names.end());

    if (mod_names.empty()) {
        std::vector<std::filesystem::path> libraries;
        for (const auto& entry: std::filesystem::directory_iterator(source_dir)) {
            if (entry.is_regular_file() && is_shared_library_suffix(entry.path())) {
                libraries.push_back(entry.path());
            }
        }
        if (libraries.empty()) {
            throw std::runtime_error("no .mod or MOD shared libraries found in " + source_dir.string());
        }
        std::sort(libraries.begin(), libraries.end());
        for (const auto& library: libraries) {
            register_mod_library(library.string());
        }
        return;
    }

    const auto library_dirs = library_dirs_for_source(source_dir);
    std::vector<std::string> missing;
    for (const auto& name: mod_names) {
        std::optional<std::filesystem::path> library;
        for (const auto& dir: library_dirs) {
            for (const auto& filename: library_names_for_mod(name)) {
                const auto candidate = dir / filename;
                if (std::filesystem::exists(candidate)) {
                    library = candidate;
                    break;
                }
            }
            if (library.has_value()) {
                break;
            }
        }
        if (!library.has_value()) {
            missing.push_back(name);
            continue;
        }
        register_mod_library(library->string(), name);
    }
    if (!missing.empty()) {
        std::string message = "MOD libraries are missing for:";
        for (const auto& name: missing) {
            message += " " + name;
        }
        message += ". Run mind_nrnivmodl " + source_dir.string() + " first.";
        throw std::runtime_error(message);
    }
}

void NetworkBuilder::set_dc_input(int roi_index_value,
                                  std::unordered_map<std::string, double> values) {
    validate_roi_index(roi_index_value, "dc input ROI");
    dc_inputs_[static_cast<std::size_t>(roi_index_value)] = std::move(values);
}

void NetworkBuilder::use_region(int roi_index_value,
                                std::string library_path,
                                std::unordered_map<std::string, double> initial_state,
                                std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "region ROI");
    library_path = resolve_rule_path(library_path);
    validate_runtime_rule_path(library_path);
    regions_.push_back(RegionConfig{
        .roi = roi_index_value,
        .rule = std::make_shared<mind_sim::macro::sim::RegionRule>(std::move(library_path)),
        .state = std::move(initial_state),
        .params = std::move(params),
    });
}

void NetworkBuilder::use_neural_field(std::string name,
                                      std::string library_path,
                                      mind_sim::macro::frontend::NodeToRoiMap node_map,
                                      mind_sim::macro::frontend::LocalConnectivity local,
                                      std::unordered_map<std::string, double> initial_state,
                                      std::unordered_map<std::string, double> params) {
    library_path = resolve_rule_path(library_path);
    validate_runtime_rule_path(library_path);
    fields_.push_back(FieldConfig{
        .name = std::move(name),
        .rule = std::make_shared<mind_sim::macro::sim::NeuralFieldRule>(std::move(library_path)),
        .node_map = std::move(node_map),
        .local = std::move(local),
        .state = std::move(initial_state),
        .params = std::move(params),
    });
}

void NetworkBuilder::macro2macro(int source_roi,
                                 int target_roi,
                                 std::string library_path,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(source_roi, "macro-to-macro source ROI");
    validate_roi_index(target_roi, "macro-to-macro target ROI");
    library_path = resolve_rule_path(library_path);
    validate_runtime_rule_path(library_path);
    macro_to_macro_.push_back(MacroToMacroConfig{
        .source = source_roi,
        .target = target_roi,
        .rule = std::make_shared<mind_sim::macro::sim::MacroToMacroRule>(std::move(library_path)),
        .params = std::move(params),
    });
}

void NetworkBuilder::use_micro(int roi_index_value) {
    validate_roi_index(roi_index_value, "micro ROI");
    Sim& micro = default_micro();
    const auto& populations = micro.model.populations();
    if (populations.empty()) {
        throw std::runtime_error("Network.use_micro requires a micro Sim with at least one population");
    }
    std::vector<int> begins;
    std::vector<int> ends;
    begins.reserve(populations.size());
    ends.reserve(populations.size());
    for (const auto& population: populations) {
        begins.push_back(population.gid_begin);
        ends.push_back(population.gid_end);
    }
    micro_bindings_.push_back(MicroBindingConfig{
        .roi = roi_index_value,
        .micro = &micro,
        .gid_range_begins = std::move(begins),
        .gid_range_ends = std::move(ends),
    });
}

void NetworkBuilder::macro2micro(int roi_index_value,
                                 std::string library_path,
                                 int gid,
                                 const PointProcessView& target,
                                 double weight,
                                 double delay,
                                 std::unordered_map<std::string, double> state,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "micro input ROI");
    library_path = resolve_rule_path(library_path);
    validate_runtime_rule_path(library_path);
    if (gid < 0) {
        throw std::runtime_error("macro2micro gid must be non-negative");
    }

    Sim* micro = nullptr;
    for (const auto& binding: micro_bindings_) {
        if (binding.roi == roi_index_value) {
            micro = binding.micro;
            break;
        }
    }
    if (micro == nullptr) {
        throw std::runtime_error("macro2micro requires Network.use_micro for the ROI first");
    }
    if (target.sim != micro) {
        throw std::runtime_error("macro2micro target point process belongs to a different micro Sim");
    }

    const int source_id = micro->model.register_spike_input_source();
    const int connection_id = micro->model.spike_input_connect(source_id, target.insert_id, weight, delay);
    static_cast<void>(connection_id);

    std::shared_ptr<mind_sim::cosim::bridge::MicroInputRule> rule;
    for (const auto& config: micro_inputs_) {
        if (config.rule && config.rule->library_path() == library_path) {
            rule = config.rule;
            break;
        }
    }
    if (!rule) {
        rule = std::make_shared<mind_sim::cosim::bridge::MicroInputRule>(std::move(library_path));
    }
    micro_inputs_.push_back(MicroInputConfig{
        .roi = roi_index_value,
        .rule = std::move(rule),
        .gid = gid,
        .source_id = source_id,
        .state = std::move(state),
        .params = std::move(params),
    });
}

void NetworkBuilder::micro2macro(int roi_index_value,
                                 std::string library_path,
                                 std::unordered_map<std::string, double> state,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "micro output ROI");
    library_path = resolve_rule_path(library_path);
    validate_runtime_rule_path(library_path);
    micro_outputs_.push_back(MicroOutputConfig{
        .roi = roi_index_value,
        .rule = std::make_shared<mind_sim::cosim::bridge::MicroOutputRule>(std::move(library_path)),
        .state = std::move(state),
        .params = std::move(params),
    });
}

mind_sim::macro::frontend::Network NetworkBuilder::build() const {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::unordered_set<std::string>> accepted_by_roi(static_cast<std::size_t>(roi_count()));
    std::vector<std::unordered_set<std::string>> outputs_by_roi(static_cast<std::size_t>(roi_count()));

    for (const auto& config: regions_) {
        for (const auto& name: config.rule->target_input_names()) {
            add_name(inputs, name);
            accepted_by_roi[static_cast<std::size_t>(config.roi)].insert(name);
        }
        for (const auto& name: config.rule->source_exposure_names()) {
            add_name(outputs, name);
            outputs_by_roi[static_cast<std::size_t>(config.roi)].insert(name);
        }
    }

    for (const auto& config: fields_) {
        const auto& node_to_roi = config.node_map.node_to_roi();
        std::unordered_set<int> owned_rois;
        for (int roi_value: node_to_roi) {
            validate_roi_index(roi_value, "neural field node ROI");
            owned_rois.insert(roi_value);
        }
        for (int roi_value: owned_rois) {
            for (const auto& name: config.rule->target_input_names()) {
                add_name(inputs, name);
                accepted_by_roi[static_cast<std::size_t>(roi_value)].insert(name);
            }
            for (const auto& name: config.rule->source_exposure_names()) {
                add_name(outputs, name);
                outputs_by_roi[static_cast<std::size_t>(roi_value)].insert(name);
            }
        }
    }

    for (const auto& config: micro_inputs_) {
        for (const auto& name: config.rule->target_input_names()) {
            add_name(inputs, name);
            accepted_by_roi[static_cast<std::size_t>(config.roi)].insert(name);
        }
    }

    for (const auto& config: micro_outputs_) {
        for (const auto& name: config.rule->source_exposure_names()) {
            add_name(outputs, name);
            outputs_by_roi[static_cast<std::size_t>(config.roi)].insert(name);
        }
    }

    for (const auto& config: macro_to_macro_) {
        for (const auto& name: config.rule->source_exposure_names()) {
            add_name(outputs, name);
        }
        for (const auto& name: config.rule->target_input_names()) {
            add_name(inputs, name);
        }
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " source output",
                      config.rule->source_exposure_names(),
                      outputs_by_roi[static_cast<std::size_t>(config.source)]);
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " target input",
                      config.rule->target_input_names(),
                      accepted_by_roi[static_cast<std::size_t>(config.target)]);
    }

    for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
        const auto& dc = dc_inputs_[static_cast<std::size_t>(roi_value)];
        std::vector<std::string> dc_names;
        dc_names.reserve(dc.size());
        for (const auto& [name, _]: dc) {
            dc_names.push_back(name);
        }
        require_names(connectivity_.rois()[static_cast<std::size_t>(roi_value)].label + " dc_input",
                      dc_names,
                      accepted_by_roi[static_cast<std::size_t>(roi_value)]);
        for (const auto& name: dc_names) {
            add_name(inputs, name);
        }
    }

    std::vector<int> record_rois;
    if (recorded_rois_.has_value()) {
        record_rois = *recorded_rois_;
    } else {
        record_rois.reserve(static_cast<std::size_t>(roi_count()));
        for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
            record_rois.push_back(roi_value);
        }
    }

    std::vector<int> record_output_indices;
    if (recorded_outputs_.has_value()) {
        record_output_indices = output_indices(outputs, *recorded_outputs_);
    } else {
        record_output_indices.reserve(outputs.size());
        for (int index = 0; index < static_cast<int>(outputs.size()); ++index) {
            record_output_indices.push_back(index);
        }
    }

    mind_sim::macro::frontend::Network network(
        connectivity_,
        inputs,
        outputs,
        std::move(record_rois),
        std::move(record_output_indices));
    if (dt_ > 0.0) {
        network.set_dt(dt_);
    }
    if (exchange_window_ > 0.0) {
        network.set_exchange_window(exchange_window_);
    }
    const int roi_width = roi_count();

    for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
        const auto native_roi = network.roi(roi_value);
        for (const auto& [name, value]: dc_inputs_[static_cast<std::size_t>(roi_value)]) {
            network.set_dc_input_value(native_roi, name, value);
        }
    }

    for (const auto& config: regions_) {
        network.use_region_rule(network.roi(config.roi),
                                config.rule,
                                region_state_values(*config.rule, config.state),
                                region_param_values(*config.rule, config.params),
                                offset_map(inputs, config.rule->target_input_names(), roi_width),
                                offset_map(outputs, config.rule->source_exposure_names(), roi_width));
    }

    for (const auto& config: fields_) {
        std::vector<mind_sim::macro::frontend::FieldOutputReducer> reducers;
        reducers.reserve(config.rule->source_exposure_names().size());
        for (const auto& name: config.rule->source_exposure_names()) {
            const auto state_found = std::find(config.rule->state_names().begin(),
                                               config.rule->state_names().end(),
                                               name);
            if (state_found == config.rule->state_names().end()) {
                throw std::runtime_error("neural field source output must name a STATE: " + name);
            }
            const auto output_found = std::find(outputs.begin(), outputs.end(), name);
            reducers.push_back(mind_sim::macro::frontend::FieldOutputReducer{
                .state_index = static_cast<int>(state_found - config.rule->state_names().begin()),
                .output_index = static_cast<int>(output_found - outputs.begin()),
            });
        }
        network.use_neural_field(config.name,
                                 config.rule,
                                 config.node_map,
                                 config.local,
                                 field_state_values(*config.rule, config.state, config.node_map.node_count()),
                                 field_param_values(*config.rule, config.params),
                                 offset_map(inputs, config.rule->target_input_names(), roi_width),
                                 std::move(reducers));
    }

    std::vector<Sim*> micro_by_roi(static_cast<std::size_t>(roi_count()), nullptr);
    std::unordered_set<Sim*> recorder_attached;
    for (const auto& binding: micro_bindings_) {
        if (!binding.micro->model.core_initialized()) {
            throw std::runtime_error("Network.use_micro requires an initialized micro Sim");
        }
        const int circuit_index = network.use_micro("micro", binding.micro->model.core_neuron_data_shared());
        network.bind_micro_roi(circuit_index,
                               network.roi(binding.roi),
                               make_gid_ranges(binding.gid_range_begins, binding.gid_range_ends));
        micro_by_roi[static_cast<std::size_t>(binding.roi)] = binding.micro;
        if (recorder_attached.insert(binding.micro).second) {
            attach_micro_recorders(network, *binding.micro);
        }
    }

    struct MicroInputBatch {
        int roi{-1};
        std::shared_ptr<mind_sim::cosim::bridge::MicroInputRule> rule{};
        std::vector<std::pair<int, int>> entries{};
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    std::vector<MicroInputBatch> micro_input_batches;
    for (const auto& config: micro_inputs_) {
        bool merged = false;
        for (auto& batch: micro_input_batches) {
            if (batch.roi == config.roi &&
                batch.rule->library_path() == config.rule->library_path() &&
                batch.state == config.state &&
                batch.params == config.params) {
                batch.entries.emplace_back(config.gid, config.source_id);
                merged = true;
                break;
            }
        }
        if (!merged) {
            micro_input_batches.push_back(MicroInputBatch{
                .roi = config.roi,
                .rule = config.rule,
                .entries = {{config.gid, config.source_id}},
                .state = config.state,
                .params = config.params,
            });
        }
    }

    for (auto& batch: micro_input_batches) {
        std::stable_sort(
            batch.entries.begin(),
            batch.entries.end(),
            [](const auto& left, const auto& right) {
                if (left.first != right.first) {
                    return left.first < right.first;
                }
                return left.second < right.second;
            });
        std::vector<int> source_ids;
        source_ids.reserve(batch.entries.size());
        for (const auto& [_, source_id]: batch.entries) {
            source_ids.push_back(source_id);
        }
        auto* micro = micro_by_roi[static_cast<std::size_t>(batch.roi)];
        if (!micro) {
            throw std::runtime_error("micro input rule requires Network.use_micro for the ROI");
        }
        const int source_count = static_cast<int>(source_ids.size());
        auto source_indices = spike_input_runtime_indices(*micro, source_ids);
        network.configure_macro_to_micro_rule(network.roi(batch.roi),
                                              batch.rule,
                                              micro_input_state_values(*batch.rule, batch.state, source_count),
                                              micro_input_param_values(*batch.rule, batch.params),
                                              std::move(source_indices),
                                              offset_map(inputs, batch.rule->target_input_names(), roi_width));
    }

    for (const auto& config: micro_outputs_) {
        network.configure_micro_output_rule(network.roi(config.roi),
                                            config.rule,
                                            micro_output_state_values(*config.rule, config.state),
                                            micro_output_param_values(*config.rule, config.params),
                                            offset_map(outputs, config.rule->source_exposure_names(), roi_width));
    }

    for (const auto& config: macro_to_macro_) {
        network.macro_to_macro(network.roi(config.source),
                       network.roi(config.target),
                       config.rule,
                       macro_to_macro_param_values(*config.rule, config.params),
                       offset_map(outputs, config.rule->source_exposure_names(), roi_width),
                       offset_map(inputs, config.rule->target_input_names(), roi_width));
    }
    return network;
}

int NetworkBuilder::roi_index(const std::string& label) const {
    return connectivity_.roi_index(label);
}

std::string NetworkBuilder::resolve_rule_path(const std::string& mechanism) const {
    const auto found = mod_libraries_.find(mechanism);
    if (found != mod_libraries_.end()) {
        return found->second;
    }
    if (looks_like_library_path(mechanism)) {
        return mechanism;
    }
    std::string known;
    for (const auto& [name, _]: mod_libraries_) {
        if (!known.empty()) {
            known += ", ";
        }
        known += name;
    }
    if (known.empty()) {
        known = "<none>";
    }
    throw std::runtime_error("unknown MOD mechanism '" + mechanism +
                             "'; call ms.macro.load_mech(directory) first. Loaded mechanisms: " +
                             known);
}

void NetworkBuilder::register_mod_library(const std::string& library_path,
                                          const std::string& expected_name) {
    const auto canonical_path = canonical_string(library_path);
    auto library = mind_sim::utils::load_dynamic_library(canonical_path);
    const auto& descriptor = descriptor_for_library(*library);
    std::vector<std::string> names{descriptor.name};
    if (!expected_name.empty() &&
        std::find(names.begin(), names.end(), expected_name) == names.end()) {
        names.push_back(expected_name);
    }
    for (const auto& name: names) {
        const auto found = mod_libraries_.find(name);
        if (found != mod_libraries_.end() && found->second != canonical_path) {
            throw std::runtime_error("duplicate MOD mechanism '" + name + "': " +
                                     found->second + " and " + canonical_path);
        }
        mod_libraries_[name] = canonical_path;
    }
}

void NetworkBuilder::validate_roi_index(int roi_index_value, const char* what) const {
    if (roi_index_value < 0 || roi_index_value >= roi_count()) {
        throw std::runtime_error(std::string(what) + " index out of range");
    }
}

void MacroConfig::load_mech(std::string directory) {
    mech_dirs_.push_back(std::move(directory));
}

void MacroConfig::set_dt(double dt) {
    require_positive_finite(dt, "macro dt");
    dt_ = dt;
}

void MacroConfig::set_exchange_window(double exchange_window) {
    require_positive_finite(exchange_window, "exchange_window");
    exchange_window_ = exchange_window;
}

void MacroConfig::apply(NetworkBuilder& builder) const {
    for (const auto& directory: mech_dirs_) {
        builder.load_mech(directory);
    }
    if (dt_ > 0.0) {
        builder.set_dt(dt_);
    }
    if (exchange_window_ > 0.0) {
        builder.set_exchange_window(exchange_window_);
    }
}

}  // namespace mind_sim::python_api::bindings
