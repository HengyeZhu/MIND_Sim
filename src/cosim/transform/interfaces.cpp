#include "cosim/transform/interfaces.hpp"

#include "mod/rule_registry.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mind_sim::cosim::transform {

namespace {

template <typename T>
void validate_params_size(const std::vector<T>& params,
                          int expected,
                          const char* what) {
    if (params.size() != static_cast<std::size_t>(expected)) {
        throw std::runtime_error(std::string(what) + " params size mismatch");
    }
}

template <typename T>
void validate_state_size(const std::vector<T>& state,
                         int expected,
                         const char* what) {
    if (state.size() != static_cast<std::size_t>(expected)) {
        throw std::runtime_error(std::string(what) + " state size mismatch");
    }
}

std::vector<std::string> descriptor_names(int count, const char* const* names) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.emplace_back(names[index]);
    }
    return out;
}

std::vector<double> descriptor_defaults(int count, const double* values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.push_back(values[index]);
    }
    return out;
}

void append_micro_event(void* user_data, double time, int source_index) {
    if (user_data == nullptr) {
        throw std::runtime_error("MicroInputRule event sink is null");
    }
    auto& events = *static_cast<mind_sim::micro::sim::MicroEventTable*>(user_data);
    events.time.push_back(time);
    events.index.push_back(source_index);
}

}  // namespace

MicroInputRule::MicroInputRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto& entry = mind_sim::mod::find_rule_entry(
        *library_, mind_sim::mod::AbiRuleKind::MicroInput, rule_name, "MicroInputRule");
    if (entry.micro_input_apply == nullptr) {
        throw std::runtime_error("MicroInputRule registry entry has null apply function");
    }
    apply_ = entry.micro_input_apply;
    const auto& descriptor = *entry.descriptor;
    name_ = descriptor.name;
    exposure_count_ = descriptor.read_source_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    exposure_names_ = descriptor_names(descriptor.read_source_count,
                                       descriptor.read_source_exposure_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
}

const std::string& MicroInputRule::name() const noexcept {
    return name_;
}

int MicroInputRule::exposure_count() const noexcept {
    return exposure_count_;
}

int MicroInputRule::state_count() const noexcept {
    return state_count_;
}

int MicroInputRule::param_count() const noexcept {
    return param_count_;
}

const std::string& MicroInputRule::library_path() const noexcept {
    return library_->path();
}

const std::vector<std::string>& MicroInputRule::exposure_names() const noexcept {
    return exposure_names_;
}

const std::vector<std::string>& MicroInputRule::state_names() const noexcept {
    return state_names_;
}

const std::vector<double>& MicroInputRule::state_defaults() const noexcept {
    return state_defaults_;
}

const std::vector<std::string>& MicroInputRule::param_names() const noexcept {
    return param_names_;
}

const std::vector<double>& MicroInputRule::param_defaults() const noexcept {
    return param_defaults_;
}

void MicroInputRule::validate_state(const std::vector<double>& state, int source_count) const {
    if (source_count < 0) {
        throw std::runtime_error("MicroInputRule source_count must be non-negative");
    }
    validate_state_size(state, state_count_ * source_count, "MicroInputRule");
}

void MicroInputRule::validate_params(const std::vector<double>& params) const {
    validate_params_size(params, param_count_, "MicroInputRule");
}

void MicroInputRule::apply(const std::vector<double>& exposure_trace_soa,
                           int sample_count,
                           double sample_dt,
                           int network_exposure_count,
                           int roi_count,
                           int roi,
                           std::vector<double>& state,
                           const std::vector<double>& params,
                           double start_time,
                           double stop_time,
                           std::uint64_t rng_seed,
                           int exchange_start_step,
                           const std::vector<int>& source_indices,
                           const std::vector<int>& source_ids,
                           mind_sim::micro::sim::MicroEventTable& events,
                           const std::vector<int>& exposure_offsets) const {
    const int source_count = static_cast<int>(source_indices.size());
    if (sample_count <= 0) {
        throw std::runtime_error("MicroInputRule sample_count must be positive");
    }
    if (sample_dt <= 0.0) {
        throw std::runtime_error("MicroInputRule sample_dt must be positive");
    }
    if (network_exposure_count <= 0) {
        throw std::runtime_error("MicroInputRule network_exposure_count must be positive");
    }
    const auto expected_exposure_trace_size =
        static_cast<std::size_t>(sample_count) *
        static_cast<std::size_t>(network_exposure_count) *
        static_cast<std::size_t>(roi_count);
    if (exposure_trace_soa.size() != expected_exposure_trace_size) {
        throw std::runtime_error("MicroInputRule exposure trace size mismatch");
    }
    if (source_count < 0) {
        throw std::runtime_error("MicroInputRule source_count must be non-negative");
    }
    if (source_ids.size() != source_indices.size()) {
        throw std::runtime_error("MicroInputRule source id count must match source index count");
    }
    for (int index: source_indices) {
        if (index < 0) {
            throw std::runtime_error("MicroInputRule source indices must be non-negative");
        }
    }
    for (int source_id: source_ids) {
        if (source_id < 0) {
            throw std::runtime_error("MicroInputRule source ids must be non-negative");
        }
    }
    mind_sim::mod::AbiMicroInputContext context{
        .exposure_count = network_exposure_count,
        .roi_count = roi_count,
        .target_roi = roi,
        .exposure_trace_soa = exposure_trace_soa.data(),
        .sample_count = sample_count,
        .sample_dt = sample_dt,
        .source_count = source_count,
        .source_indices = source_indices.data(),
        .source_ids = source_ids.data(),
        .state_count = state_count_,
        .state = state.data(),
        .param_count = param_count_,
        .params = params.data(),
        .start_time = start_time,
        .stop_time = stop_time,
        .rng_seed = rng_seed,
        .exchange_start_step = exchange_start_step,
        .exposure_offsets = exposure_offsets.data(),
        .event_user_data = &events,
        .emit_event = append_micro_event,
    };
    apply_(&context);
}

MicroOutputRule::MicroOutputRule(std::string library_path, std::string rule_name)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))) {
    const auto& entry = mind_sim::mod::find_rule_entry(
        *library_, mind_sim::mod::AbiRuleKind::MicroOutput, rule_name, "MicroOutputRule");
    if (entry.micro_output_apply == nullptr) {
        throw std::runtime_error("MicroOutputRule registry entry has null apply function");
    }
    apply_ = entry.micro_output_apply;
    const auto& descriptor = *entry.descriptor;
    name_ = descriptor.name;
    output_count_ = descriptor.write_target_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    exposure_names_ = descriptor_names(descriptor.write_target_count,
                                       descriptor.write_target_exposure_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
}

const std::string& MicroOutputRule::name() const noexcept {
    return name_;
}

int MicroOutputRule::output_count() const noexcept {
    return output_count_;
}

int MicroOutputRule::state_count() const noexcept {
    return state_count_;
}

int MicroOutputRule::param_count() const noexcept {
    return param_count_;
}

const std::string& MicroOutputRule::library_path() const noexcept {
    return library_->path();
}

const std::vector<std::string>& MicroOutputRule::exposure_names() const noexcept {
    return exposure_names_;
}

const std::vector<std::string>& MicroOutputRule::state_names() const noexcept {
    return state_names_;
}

const std::vector<double>& MicroOutputRule::state_defaults() const noexcept {
    return state_defaults_;
}

const std::vector<std::string>& MicroOutputRule::param_names() const noexcept {
    return param_names_;
}

const std::vector<double>& MicroOutputRule::param_defaults() const noexcept {
    return param_defaults_;
}

void MicroOutputRule::validate_state(const std::vector<double>& state) const {
    validate_state_size(state, state_count_, "MicroOutputRule");
}

void MicroOutputRule::validate_params(const std::vector<double>& params) const {
    validate_params_size(params, param_count_, "MicroOutputRule");
}

void MicroOutputRule::apply(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
                            std::vector<double>& output_soa,
                            std::vector<double>& output_trace_soa,
                            int roi_count,
                            int output_count,
                            int roi,
                            std::vector<double>& state,
                            const std::vector<double>& params,
                            double start_time,
                            double stop_time,
                            int sample_count,
                            double sample_dt,
                            const std::vector<int>& exposure_offsets) const {
    mind_sim::mod::AbiSpikeTable spike_table{
        .time = spikes.time,
        .size = static_cast<int>(spikes.size()),
    };
    mind_sim::mod::AbiMicroOutputContext context{
        .exposure_count = output_count,
        .spikes = &spike_table,
        .roi_count = roi_count,
        .target_roi = roi,
        .exposure_soa = output_soa.data(),
        .sample_count = sample_count,
        .sample_dt = sample_dt,
        .exposure_trace_soa = output_trace_soa.data(),
        .state_count = state_count_,
        .state = state.data(),
        .param_count = param_count_,
        .params = params.data(),
        .start_time = start_time,
        .stop_time = stop_time,
        .exposure_offsets = exposure_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::cosim::transform
