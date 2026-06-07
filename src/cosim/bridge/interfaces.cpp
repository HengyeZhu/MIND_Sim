#include "cosim/bridge/interfaces.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::cosim::bridge {

namespace {

void validate_count(int count, const char* what) {
    if (count < 0) {
        throw std::runtime_error(std::string(what) + " must be non-negative");
    }
}

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

const mind_sim::mod::AbiRuleDescriptor& descriptor_of(
    const mind_sim::utils::DynamicLibrary& library,
    mind_sim::mod::AbiRuleKind expected,
    const char* what) {
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mod::DescriptorFn>(library.symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error(std::string(what) + " has null descriptor");
    }
    if (descriptor->abi_version != mind_sim::mod::kModAbiVersion) {
        throw std::runtime_error(std::string(what) + " ABI version mismatch");
    }
    if (descriptor->kind != static_cast<int>(expected)) {
        throw std::runtime_error(std::string(what) + " rule kind mismatch");
    }
    if (!descriptor->name || descriptor->name[0] == '\0') {
        throw std::runtime_error(std::string(what) + " descriptor has empty name");
    }
    return *descriptor;
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

MicroInputRule::MicroInputRule(std::string name,
                               std::string library_path,
                               int input_count,
                               int state_count,
                               int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_micro_input_rule_apply"))),
      input_count_(input_count),
      state_count_(state_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("MicroInputRule name must be non-empty");
    }
    validate_count(input_count_, "MicroInputRule input_count");
    validate_count(state_count_, "MicroInputRule state_count");
    validate_count(param_count_, "MicroInputRule param_count");
}

MicroInputRule::MicroInputRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mod::MicroInputApplyFn>(
          library_->symbol("mind_micro_input_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mod::AbiRuleKind::MicroInput, "MicroInputRule");
    name_ = descriptor.name;
    input_count_ = descriptor.target_input_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    target_input_names_ = descriptor_names(descriptor.target_input_count, descriptor.target_input_names);
    state_names_ = descriptor_names(descriptor.state_count, descriptor.state_names);
    state_defaults_ = descriptor_defaults(descriptor.state_count, descriptor.state_defaults);
    param_names_ = descriptor_names(descriptor.param_count, descriptor.param_names);
    param_defaults_ = descriptor_defaults(descriptor.param_count, descriptor.param_defaults);
}

const std::string& MicroInputRule::name() const noexcept {
    return name_;
}

int MicroInputRule::input_count() const noexcept {
    return input_count_;
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

const std::vector<std::string>& MicroInputRule::target_input_names() const noexcept {
    return target_input_names_;
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

void MicroInputRule::apply(const std::vector<double>& input_trace_soa,
                           int sample_count,
                           double sample_dt,
                           int network_input_count,
                           int roi_count,
                           int roi,
                           std::vector<double>& state,
                           const std::vector<double>& params,
                           double start_time,
                           double stop_time,
                           std::uint64_t rng_seed,
                           int exchange_start_step,
                           const std::vector<int>& source_indices,
                           mind_sim::micro::sim::MicroEventTable& events,
                           const std::vector<int>& target_input_offsets) const {
    const int source_count = static_cast<int>(source_indices.size());
    if (sample_count <= 0) {
        throw std::runtime_error("MicroInputRule sample_count must be positive");
    }
    if (sample_dt <= 0.0) {
        throw std::runtime_error("MicroInputRule sample_dt must be positive");
    }
    if (network_input_count <= 0) {
        throw std::runtime_error("MicroInputRule network_input_count must be positive");
    }
    const auto expected_trace_size =
        static_cast<std::size_t>(sample_count) *
        static_cast<std::size_t>(network_input_count) *
        static_cast<std::size_t>(roi_count);
    if (input_trace_soa.size() != expected_trace_size) {
        throw std::runtime_error("MicroInputRule input trace size mismatch");
    }
    if (source_count < 0) {
        throw std::runtime_error("MicroInputRule source_count must be non-negative");
    }
    for (int index: source_indices) {
        if (index < 0) {
            throw std::runtime_error("MicroInputRule source indices must be non-negative");
        }
    }
    mind_sim::mod::AbiMicroInputContext context{
        .input_count = network_input_count,
        .roi_count = roi_count,
        .target_roi = roi,
        .input_trace_soa = input_trace_soa.data(),
        .sample_count = sample_count,
        .sample_dt = sample_dt,
        .source_count = source_count,
        .source_indices = source_indices.data(),
        .state_count = state_count_,
        .state = state.data(),
        .param_count = param_count_,
        .params = params.data(),
        .start_time = start_time,
        .stop_time = stop_time,
        .rng_seed = rng_seed,
        .exchange_start_step = exchange_start_step,
        .target_input_offsets = target_input_offsets.data(),
        .event_user_data = &events,
        .emit_event = append_micro_event,
    };
    apply_(&context);
}

MicroOutputRule::MicroOutputRule(std::string name,
                                 std::string library_path,
                                 int output_count,
                                 int state_count,
                                 int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_micro_output_rule_apply"))),
      output_count_(output_count),
      state_count_(state_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("MicroOutputRule name must be non-empty");
    }
    validate_count(output_count_, "MicroOutputRule output_count");
    validate_count(state_count_, "MicroOutputRule state_count");
    validate_count(param_count_, "MicroOutputRule param_count");
    if (output_count_ == 0) {
        throw std::runtime_error("MicroOutputRule output_count must be positive");
    }
}

MicroOutputRule::MicroOutputRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mod::MicroOutputApplyFn>(
          library_->symbol("mind_micro_output_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mod::AbiRuleKind::MicroOutput, "MicroOutputRule");
    name_ = descriptor.name;
    output_count_ = descriptor.source_exposure_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    source_exposure_names_ =
        descriptor_names(descriptor.source_exposure_count, descriptor.source_exposure_names);
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

const std::vector<std::string>& MicroOutputRule::source_exposure_names() const noexcept {
    return source_exposure_names_;
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
                            const std::vector<int>& source_exposure_offsets) const {
    mind_sim::mod::AbiSpikeTable spike_table{
        .time = spikes.time,
        .gid = spikes.gid,
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
        .source_exposure_offsets = source_exposure_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::cosim::bridge
