#include "python_api/bindings/bindings.hpp"
#include "python_api/bindings/network_builder.hpp"
#include "mod/rule_registry.hpp"
#include "utils/dynamic_library.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
                             "; run mind-nrnivmodl " + source_dir.string() + " first");
}

template <typename Rule>
std::vector<double> values_with_defaults(const std::vector<std::string>& names,
                                         const std::vector<double>& defaults,
                                         const std::unordered_map<std::string, double>& overrides,
                                         const std::string& what) {
    if (names.size() != defaults.size()) {
        throw std::runtime_error(what + " rule metadata names/defaults size mismatch");
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
    : connectivity_(std::move(connectivity)) {}

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

void NetworkBuilder::set_roi_initial_history(int roi_index_value,
                                             std::vector<std::string> output_names,
                                             int axis1_count,
                                             int axis2_count,
                                             std::vector<double> values,
                                             RoiInitialHistoryLayout layout) {
    validate_roi_index(roi_index_value, "initial_history ROI");
    if (output_names.empty()) {
        throw std::runtime_error("ROI initial_history requires at least one output");
    }
    if (axis1_count <= 0 || axis2_count <= 0) {
        throw std::runtime_error("ROI initial_history axes must be positive");
    }
    const int output_axis_count =
        layout == RoiInitialHistoryLayout::TimeOutput ? axis2_count : axis1_count;
    if (output_axis_count != static_cast<int>(output_names.size())) {
        throw std::runtime_error("ROI initial_history output axis does not match outputs");
    }
    const auto expected_size =
        static_cast<std::size_t>(axis1_count) * static_cast<std::size_t>(axis2_count);
    if (values.size() != expected_size) {
        throw std::runtime_error("ROI initial_history value count does not match its shape");
    }
    std::unordered_set<std::string> seen_outputs;
    for (const auto& output_name: output_names) {
        if (!seen_outputs.insert(output_name).second) {
            throw std::runtime_error("ROI initial_history output names must be unique");
        }
    }

    roi_initial_histories_.push_back(RoiInitialHistoryConfig{
        .roi = roi_index_value,
        .output_names = std::move(output_names),
        .axis1_count = axis1_count,
        .axis2_count = axis2_count,
        .values = std::move(values),
        .layout = layout,
    });
}

void NetworkBuilder::use_region(int roi_index_value,
                                std::string library_path,
                                std::unordered_map<std::string, double> initial_state,
                                std::unordered_map<std::string, double> params) {
    validate_roi_index(roi_index_value, "region ROI");
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    regions_.push_back(RegionConfig{
        .roi = roi_index_value,
        .rule = region_rule(rule_ref),
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
    const RuleRef rule_ref = resolve_rule_ref(library_path);
    macro_to_macro_.push_back(MacroToMacroConfig{
        .source = source_roi,
        .target = target_roi,
        .rule = macro_to_macro_rule(rule_ref),
        .params = std::move(params),
    });
}

void NetworkBuilder::use_micro(int roi_index_value, Sim& micro, std::vector<std::string> exposures) {
    validate_roi_index(roi_index_value, "micro ROI");
    if (exposures.empty()) {
        throw std::runtime_error("ROI.use_micro requires at least one exposure");
    }
    if (micro_ == nullptr) {
        micro_ = &micro;
    } else if (micro_ != &micro) {
        throw std::runtime_error("a Network can bind only one micro sim");
    }
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
        .exposures = std::move(exposures),
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

    micro_inputs_.push_back(MicroInputConfig{
        .roi = roi_index_value,
        .rule = micro_input_rule(rule_ref),
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
        .rule = micro_output_rule(rule_ref),
        .sid = sid,
        .state = std::move(state),
        .params = std::move(params),
    });
}

mind_sim::macro::frontend::Network NetworkBuilder::build() const {
    std::vector<std::string> outputs;
    std::vector<std::unordered_set<std::string>> exposures_by_roi(static_cast<std::size_t>(roi_count()));

    for (const auto& config: regions_) {
        for (const auto& name: config.rule->exposure_names()) {
            add_name(outputs, name);
            exposures_by_roi[static_cast<std::size_t>(config.roi)].insert(name);
        }
    }

    for (const auto& binding: micro_bindings_) {
        for (const auto& name: binding.exposures) {
            add_name(outputs, name);
            exposures_by_roi[static_cast<std::size_t>(binding.roi)].insert(name);
        }
    }

    for (const auto& config: macro_to_macro_) {
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " source exposure",
                      config.rule->read_source_exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.source)]);
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " target exposure",
                      config.rule->read_target_exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.target)]);
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " source write exposure",
                      config.rule->write_source_exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.source)]);
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.source)].label +
                          " -> " + connectivity_.rois()[static_cast<std::size_t>(config.target)].label +
                          " target write exposure",
                      config.rule->write_target_exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.target)]);
    }

    for (const auto& config: micro_inputs_) {
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.roi)].label +
                          " macro2micro exposure",
                      config.rule->exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.roi)]);
    }

    for (const auto& config: micro_outputs_) {
        require_names(connectivity_.rois()[static_cast<std::size_t>(config.roi)].label +
                          " micro2macro exposure",
                      config.rule->exposure_names(),
                      exposures_by_roi[static_cast<std::size_t>(config.roi)]);
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

    for (const auto& config: regions_) {
        network.use_region_rule(network.roi(config.roi),
                                config.rule,
                                region_state_values(*config.rule, config.state),
                                region_param_values(*config.rule, config.params),
                                offset_map(outputs, config.rule->exposure_names(), roi_width));
    }

    std::vector<Sim*> micro_by_roi(static_cast<std::size_t>(roi_count()), nullptr);
    int circuit_index = -1;
    for (const auto& binding: micro_bindings_) {
        if (!binding.micro->model.core_initialized()) {
            throw std::runtime_error("Network.use_micro requires an initialized micro Sim");
        }
        if (circuit_index < 0) {
            circuit_index = network.use_micro(binding.micro->name,
                                              binding.micro->model.core_neuron_data_shared());
            attach_micro_recorders(network, *binding.micro);
        }
        network.bind_micro_roi(circuit_index,
                               network.roi(binding.roi),
                               make_gid_ranges(binding.gid_range_begins, binding.gid_range_ends));
        micro_by_roi[static_cast<std::size_t>(binding.roi)] = binding.micro;
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
                batch.rule->name() == config.rule->name() &&
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

    std::uint64_t rng_stream_id = 0;
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
                                              offset_map(outputs, batch.rule->exposure_names(), roi_width),
                                              rng_stream_id++);
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
                batch.rule->name() == config.rule->name() &&
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
                                            offset_map(outputs, batch.rule->exposure_names(), roi_width));
    }

    for (const auto& config: macro_to_macro_) {
        network.macro_to_macro(network.roi(config.source),
	                       network.roi(config.target),
	                       config.rule,
	                       macro_to_macro_param_values(*config.rule, config.params),
	                       offset_map(outputs, config.rule->read_source_exposure_names(), roi_width),
	                       offset_map(outputs, config.rule->read_target_exposure_names(), roi_width),
	                       offset_map(outputs, config.rule->write_source_exposure_names(), roi_width),
	                       offset_map(outputs, config.rule->write_target_exposure_names(), roi_width));
    }
    if (!roi_initial_histories_.empty()) {
        int time_count = -1;
        std::vector<std::string> history_outputs;
        std::unordered_map<std::string, int> history_output_slots;
        for (const auto& config: roi_initial_histories_) {
            const int config_time_count =
                config.layout == RoiInitialHistoryLayout::TimeOutput ? config.axis1_count : config.axis2_count;
            if (time_count < 0) {
                time_count = config_time_count;
            } else if (time_count != config_time_count) {
                throw std::runtime_error("all ROI initial_history calls must use the same time count");
            }
            for (const auto& output_name: config.output_names) {
                const int output = network.output_index(output_name);
                static_cast<void>(output);
                if (history_output_slots.find(output_name) == history_output_slots.end()) {
                    const int slot = static_cast<int>(history_outputs.size());
                    history_output_slots.emplace(output_name, slot);
                    history_outputs.push_back(output_name);
                }
            }
        }

        const auto history_output_count = static_cast<int>(history_outputs.size());
        const auto slot_size = static_cast<std::size_t>(history_output_count * roi_count());
        std::vector<double> history(static_cast<std::size_t>(time_count) * slot_size, 0.0);
        for (int time = 0; time < time_count; ++time) {
            for (int history_output = 0; history_output < history_output_count; ++history_output) {
                const int output = network.output_index(history_outputs[static_cast<std::size_t>(history_output)]);
                for (int roi_value = 0; roi_value < roi_count(); ++roi_value) {
                    history[static_cast<std::size_t>(time) * slot_size +
                            static_cast<std::size_t>(history_output * roi_count() + roi_value)] =
                        network.output_history_start()[static_cast<std::size_t>(roi_value)]
                            .values[static_cast<std::size_t>(output)];
                }
            }
        }

        std::vector<unsigned char> seen(
            static_cast<std::size_t>(roi_count() * history_output_count), 0);
        for (const auto& config: roi_initial_histories_) {
            for (int provided_output = 0; provided_output < static_cast<int>(config.output_names.size());
                 ++provided_output) {
                const auto slot_found =
                    history_output_slots.find(config.output_names[static_cast<std::size_t>(provided_output)]);
                const int history_output = slot_found->second;
                const auto seen_offset =
                    static_cast<std::size_t>(config.roi * history_output_count + history_output);
                if (seen[seen_offset] != 0) {
                    throw std::runtime_error("ROI initial_history already has output " +
                                             config.output_names[static_cast<std::size_t>(provided_output)]);
                }
                seen[seen_offset] = 1;

                for (int time = 0; time < time_count; ++time) {
                    std::size_t source_offset = 0;
                    if (config.layout == RoiInitialHistoryLayout::TimeOutput) {
                        source_offset =
                            static_cast<std::size_t>(time) * static_cast<std::size_t>(config.axis2_count) +
                            static_cast<std::size_t>(provided_output);
                    } else {
                        source_offset =
                            static_cast<std::size_t>(provided_output) *
                                static_cast<std::size_t>(config.axis2_count) +
                            static_cast<std::size_t>(time);
                    }
                    const double value = config.values[source_offset];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("ROI initial_history values must be finite");
                    }
                    history[static_cast<std::size_t>(time) * slot_size +
                            static_cast<std::size_t>(history_output * roi_count() + config.roi)] = value;
                }
            }
        }

        network.set_initial_history(history_outputs,
                                    time_count,
                                    history_output_count,
                                    roi_count(),
                                    history,
                                    mind_sim::macro::frontend::Network::InitialHistoryLayout::TimeOutputRoi);
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
                             "'; call mind_sim.load_mech(directory) before creating the network. Loaded mechanisms: " +
                             known);
}

std::shared_ptr<mind_sim::macro::sim::RegionRule> NetworkBuilder::region_rule(
    const RuleRef& rule_ref) {
    const auto key = rule_ref.library_path + '\n' + rule_ref.rule_name;
    auto found = region_rule_cache_.find(key);
    if (found == region_rule_cache_.end()) {
        found = region_rule_cache_
                    .emplace(key,
                             std::make_shared<mind_sim::macro::sim::RegionRule>(
                                 rule_ref.library_path,
                                 rule_ref.rule_name))
                    .first;
    }
    return found->second;
}

std::shared_ptr<mind_sim::macro::sim::MacroToMacroRule> NetworkBuilder::macro_to_macro_rule(
    const RuleRef& rule_ref) {
    const auto key = rule_ref.library_path + '\n' + rule_ref.rule_name;
    auto found = macro_to_macro_rule_cache_.find(key);
    if (found == macro_to_macro_rule_cache_.end()) {
        found = macro_to_macro_rule_cache_
                    .emplace(key,
                             std::make_shared<mind_sim::macro::sim::MacroToMacroRule>(
                                 rule_ref.library_path,
                                 rule_ref.rule_name))
                    .first;
    }
    return found->second;
}

std::shared_ptr<mind_sim::cosim::transform::MicroInputRule> NetworkBuilder::micro_input_rule(
    const RuleRef& rule_ref) {
    const auto key = rule_ref.library_path + '\n' + rule_ref.rule_name;
    auto found = micro_input_rule_cache_.find(key);
    if (found == micro_input_rule_cache_.end()) {
        found = micro_input_rule_cache_
                    .emplace(key,
                             std::make_shared<mind_sim::cosim::transform::MicroInputRule>(
                                 rule_ref.library_path,
                                 rule_ref.rule_name))
                    .first;
    }
    return found->second;
}

std::shared_ptr<mind_sim::cosim::transform::MicroOutputRule> NetworkBuilder::micro_output_rule(
    const RuleRef& rule_ref) {
    const auto key = rule_ref.library_path + '\n' + rule_ref.rule_name;
    auto found = micro_output_rule_cache_.find(key);
    if (found == micro_output_rule_cache_.end()) {
        found = micro_output_rule_cache_
                    .emplace(key,
                             std::make_shared<mind_sim::cosim::transform::MicroOutputRule>(
                                 rule_ref.library_path,
                                 rule_ref.rule_name))
                    .first;
    }
    return found->second;
}

void NetworkBuilder::register_mod_library(const std::string& library_path) {
    const auto canonical_path = canonical_string(library_path);
    auto library = mind_sim::utils::load_dynamic_library(canonical_path);
    if (!library->has_symbol("mind_rule_reg")) {
        return;
    }
    const auto rules = mind_sim::mod::register_rules_from_library(library, "MOD library");
    if (rules.empty()) {
        return;
    }
    for (const auto& rule: rules) {
        const std::string& name = rule->name;
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
    if (std::find(mech_dirs_.begin(), mech_dirs_.end(), directory) == mech_dirs_.end()) {
        mech_dirs_.push_back(std::move(directory));
    }
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
