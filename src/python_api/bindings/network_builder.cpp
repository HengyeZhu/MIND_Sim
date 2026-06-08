#include "python_api/bindings/bindings.hpp"
#include "python_api/bindings/network_builder.hpp"
#include "mod/rule_registry.hpp"
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

std::string canonical_string(const std::filesystem::path& path) {
    return std::filesystem::canonical(path).string();
}

std::filesystem::path unified_mod_library_for(const std::filesystem::path& source_dir) {
    const auto direct = source_dir / "libcorenrnmech.so";
    if (std::filesystem::is_regular_file(direct)) {
        return direct;
    }
    const auto x86 = source_dir / "x86_64" / "libcorenrnmech.so";
    if (std::filesystem::is_regular_file(x86)) {
        return x86;
    }
    throw std::runtime_error("could not find libcorenrnmech.so under " + source_dir.string() +
                             "; run mind_nrnivmodl " + source_dir.string() + " first");
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
    const mind_sim::cosim::transform::MicroInputRule& rule,
    const std::unordered_map<std::string, double>& overrides,
    int source_count) {
    if (source_count < 0) {
        throw std::runtime_error(rule.name() + " source count must be non-negative");
    }
    const auto base = values_with_defaults<mind_sim::cosim::transform::MicroInputRule>(
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
    const mind_sim::cosim::transform::MicroInputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::transform::MicroInputRule>(
        rule.param_names(),
        rule.param_defaults(),
        overrides,
        rule.name() + " params");
}

std::vector<double> micro_output_state_values(
    const mind_sim::cosim::transform::MicroOutputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::transform::MicroOutputRule>(
        rule.state_names(),
        rule.state_defaults(),
        overrides,
        rule.name() + " state");
}

std::vector<double> micro_output_param_values(
    const mind_sim::cosim::transform::MicroOutputRule& rule,
    const std::unordered_map<std::string, double>& overrides) {
    return values_with_defaults<mind_sim::cosim::transform::MicroOutputRule>(
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

mind_sim::macro::frontend::LocalConnectivity empty_local_connectivity(int node_count) {
    return mind_sim::macro::frontend::LocalConnectivity(
        node_count,
        std::vector<int>(static_cast<std::size_t>(node_count + 1), 0),
        {},
        {});
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

void require_positive_finite(double value, const char* what) {
    if (value <= 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + " must be positive and finite");
    }
}

std::vector<int> spike_input_runtime_indices(Sim& micro, const std::vector<int>& macro2micro_ids) {
    if (macro2micro_ids.empty()) {
        throw std::runtime_error("macro2micro requires at least one source");
    }
    std::vector<int> indices;
    indices.reserve(macro2micro_ids.size());
    for (std::size_t i = 0; i < macro2micro_ids.size(); ++i) {
        const int runtime_index = micro.model.spike_input_runtime_index(macro2micro_ids[i]);
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

void NetworkBuilder::record(int roi_index_value, std::string output_name) {
    validate_roi_index(roi_index_value, "record ROI");
    if (output_name.empty()) {
        throw std::runtime_error("record output name must be non-empty");
    }
    if (!recorded_rois_.has_value()) {
        recorded_rois_ = std::vector<int>{};
    }
    if (std::find(recorded_rois_->begin(), recorded_rois_->end(), roi_index_value) == recorded_rois_->end()) {
        recorded_rois_->push_back(roi_index_value);
    }
    if (!recorded_outputs_.has_value()) {
        recorded_outputs_ = std::vector<std::string>{};
    }
    if (std::find(recorded_outputs_->begin(), recorded_outputs_->end(), output_name) == recorded_outputs_->end()) {
        recorded_outputs_->push_back(std::move(output_name));
    }
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
    register_mod_library(unified_mod_library_for(source_dir).string());
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
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    regions_.push_back(RegionConfig{
        .roi = roi_index_value,
        .rule = std::make_shared<mind_sim::macro::sim::RegionRule>(rule_ref.library_path,
                                                                    rule_ref.rule_name),
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
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    fields_.push_back(FieldConfig{
        .name = std::move(name),
        .rule = std::make_shared<mind_sim::macro::sim::NeuralFieldRule>(rule_ref.library_path,
                                                                        rule_ref.rule_name),
        .node_map = std::move(node_map),
        .local = std::move(local),
        .state = std::move(initial_state),
        .params = std::move(params),
    });
}

void NetworkBuilder::use_neural_field(std::string name,
                                      std::string library_path,
                                      mind_sim::macro::frontend::NodeToRoiMap node_map,
                                      std::unordered_map<std::string, double> initial_state,
                                      std::unordered_map<std::string, double> params) {
    const int node_count = node_map.node_count();
    use_neural_field(std::move(name),
                     std::move(library_path),
                     std::move(node_map),
                     empty_local_connectivity(node_count),
                     std::move(initial_state),
                     std::move(params));
}

void NetworkBuilder::macro2macro(int source_roi,
                                 int target_roi,
                                 std::string library_path,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(source_roi, "macro-to-macro source ROI");
    validate_roi_index(target_roi, "macro-to-macro target ROI");
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    macro_to_macro_.push_back(MacroToMacroConfig{
        .source = source_roi,
        .target = target_roi,
        .rule = std::make_shared<mind_sim::macro::sim::MacroToMacroRule>(rule_ref.library_path,
                                                                         rule_ref.rule_name),
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
                                 const PointProcessView& target,
                                 double weight,
                                 double delay,
                                 std::unordered_map<std::string, double> state,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "micro input ROI");
    const RuleRef rule_ref = resolve_rule_ref(library_path);

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

    const int macro2micro_id = micro->model.register_spike_input_source();
    const int connection_id = micro->model.spike_input_connect(macro2micro_id, target.insert_id, weight, delay);
    static_cast<void>(connection_id);

    std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> rule;
    for (const auto& config: micro_inputs_) {
        if (config.rule && config.rule->library_path() == rule_ref.library_path &&
            config.rule->name() == rule_ref.rule_name) {
            rule = config.rule;
            break;
        }
    }
    if (!rule) {
        rule = std::make_shared<mind_sim::cosim::transform::MicroInputRule>(rule_ref.library_path,
                                                                            rule_ref.rule_name);
    }
    micro_inputs_.push_back(MicroInputConfig{
        .roi = roi_index_value,
        .rule = std::move(rule),
        .macro2micro_id = macro2micro_id,
        .state = std::move(state),
        .params = std::move(params),
    });
}

void NetworkBuilder::micro2macro(int roi_index_value,
                                 std::string library_path,
                                 int sid,
                                 std::unordered_map<std::string, double> state,
                                 std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "micro output ROI");
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    if (sid < 0) {
        throw std::runtime_error("micro2macro source sid must be non-negative");
    }
    micro_outputs_.push_back(MicroOutputConfig{
        .roi = roi_index_value,
        .rule = std::make_shared<mind_sim::cosim::transform::MicroOutputRule>(rule_ref.library_path,
                                                                              rule_ref.rule_name),
        .sid = sid,
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

    for (const auto& config: micro_inputs_) {
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.roi)].label +
                          " macro2micro source exposure",
                      config.rule->source_exposure_names(),
                      outputs_by_roi[static_cast<std::size_t>(config.roi)]);
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
        std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> rule{};
        std::vector<int> macro2micro_ids{};
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
                batch.macro2micro_ids.push_back(config.macro2micro_id);
                merged = true;
                break;
            }
        }
        if (!merged) {
            micro_input_batches.push_back(MicroInputBatch{
                .roi = config.roi,
                .rule = config.rule,
                .macro2micro_ids = {config.macro2micro_id},
                .state = config.state,
                .params = config.params,
            });
        }
    }

    for (auto& batch: micro_input_batches) {
        auto* micro = micro_by_roi[static_cast<std::size_t>(batch.roi)];
        if (!micro) {
            throw std::runtime_error("micro input rule requires Network.use_micro for the ROI");
        }
        const int source_count = static_cast<int>(batch.macro2micro_ids.size());
        auto macro2micro_indices = spike_input_runtime_indices(*micro, batch.macro2micro_ids);
        network.configure_macro_to_micro_rule(network.roi(batch.roi),
                                              batch.rule,
                                              micro_input_state_values(*batch.rule, batch.state, source_count),
                                              micro_input_param_values(*batch.rule, batch.params),
                                              std::move(macro2micro_indices),
                                              batch.macro2micro_ids,
                                              offset_map(inputs, batch.rule->target_input_names(), roi_width),
                                              offset_map(outputs, batch.rule->source_exposure_names(), roi_width));
    }

    struct MicroOutputBatch {
        int roi{-1};
        std::shared_ptr<mind_sim::cosim::transform::MicroOutputRule> rule{};
        std::vector<int> sids{};
        std::unordered_map<std::string, double> state{};
        std::unordered_map<std::string, double> params{};
    };

    std::vector<MicroOutputBatch> micro_output_batches;
    for (const auto& config: micro_outputs_) {
        bool merged = false;
        for (auto& batch: micro_output_batches) {
            if (batch.roi == config.roi &&
                batch.rule->library_path() == config.rule->library_path() &&
                batch.state == config.state &&
                batch.params == config.params) {
                batch.sids.push_back(config.sid);
                merged = true;
                break;
            }
        }
        if (!merged) {
            micro_output_batches.push_back(MicroOutputBatch{
                .roi = config.roi,
                .rule = config.rule,
                .sids = {config.sid},
                .state = config.state,
                .params = config.params,
            });
        }
    }

    for (auto& batch: micro_output_batches) {
        std::sort(batch.sids.begin(), batch.sids.end());
        batch.sids.erase(std::unique(batch.sids.begin(), batch.sids.end()), batch.sids.end());
        network.configure_micro_output_rule(network.roi(batch.roi),
                                            batch.rule,
                                            micro_output_state_values(*batch.rule, batch.state),
                                            micro_output_param_values(*batch.rule, batch.params),
                                            std::move(batch.sids),
                                            offset_map(outputs, batch.rule->source_exposure_names(), roi_width));
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

NetworkBuilder::RuleRef NetworkBuilder::resolve_rule_ref(const std::string& mechanism) const {
    const auto found = mod_rules_.find(mechanism);
    if (found != mod_rules_.end()) {
        return found->second;
    }
    std::string known;
    for (const auto& [name, _]: mod_rules_) {
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

void NetworkBuilder::register_mod_library(const std::string& library_path) {
    const auto canonical_path = canonical_string(library_path);
    auto library = mind_sim::utils::load_dynamic_library(canonical_path);
    const auto descriptors = mind_sim::mod::rule_descriptors(*library, "MOD library");
    if (descriptors.empty()) {
        throw std::runtime_error("MOD library has no MIND rules: " + canonical_path);
    }
    for (const auto* descriptor: descriptors) {
        const std::string name = descriptor->name;
        const auto found = mod_rules_.find(name);
        if (found != mod_rules_.end() && found->second.library_path != canonical_path) {
            throw std::runtime_error("duplicate MOD mechanism '" + name + "': " +
                                     found->second.library_path + " and " + canonical_path);
        }
        mod_rules_[name] = RuleRef{
            .library_path = canonical_path,
            .rule_name = name,
        };
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
