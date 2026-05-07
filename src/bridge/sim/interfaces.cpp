#include "bridge/sim/interfaces.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mind_sim::bridge::sim {

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

void write_event(void* user, double time, int index) {
    auto* events = static_cast<mind_sim::micro::sim::MicroEventTable*>(user);
    events->time.push_back(time);
    events->index.push_back(index);
}

const mind_sim::mind_mod::AbiRuleDescriptor& descriptor_of(
    const mind_sim::utils::DynamicLibrary& library,
    mind_sim::mind_mod::AbiRuleKind expected,
    const char* what) {
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mind_mod::DescriptorFn>(library.symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error(std::string(what) + " has null descriptor");
    }
    if (descriptor->abi_version != mind_sim::mind_mod::kMindModAbiVersion) {
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

struct RandomRuntimeContext {
    const RandomStreamRule* rule{nullptr};
    std::vector<double>* state{nullptr};
};

double random_stream_uniform(void* user, int index, int draw) {
    auto* context = static_cast<RandomRuntimeContext*>(user);
    return context->rule->uniform(context->state->data(), index, draw);
}

}  // namespace

RandomStreamRule::RandomStreamRule(std::string library_path, int state_count)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      uniform_(reinterpret_cast<decltype(uniform_)>(library_->symbol("mind_random_uniform"))),
      state_count_(state_count) {
    validate_count(state_count_, "RandomStreamRule state_count");
}

int RandomStreamRule::state_count() const noexcept {
    return state_count_;
}

const std::string& RandomStreamRule::library_path() const noexcept {
    return library_->path();
}

double RandomStreamRule::uniform(double* state, int index, int draw) const {
    return uniform_(state, state_count_, index, draw);
}

MicroInputRule::MicroInputRule(std::string name,
                               std::string library_path,
                               int input_count,
                               int state_count,
                               int param_count,
                               int input_port_count,
                               int random_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_micro_input_rule_apply"))),
      input_count_(input_count),
      state_count_(state_count),
      param_count_(param_count),
      input_port_count_(input_port_count),
      random_count_(random_count) {
    if (name_.empty()) {
        throw std::runtime_error("MicroInputRule name must be non-empty");
    }
    validate_count(input_count_, "MicroInputRule input_count");
    validate_count(state_count_, "MicroInputRule state_count");
    validate_count(param_count_, "MicroInputRule param_count");
    validate_count(input_port_count_, "MicroInputRule input_port_count");
    validate_count(random_count_, "MicroInputRule random_count");
}

MicroInputRule::MicroInputRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mind_mod::MicroInputApplyFn>(
          library_->symbol("mind_micro_input_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mind_mod::AbiRuleKind::MicroInput, "MicroInputRule");
    name_ = descriptor.name;
    input_count_ = descriptor.read_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
    input_port_count_ = descriptor.emit_count;
    random_count_ = descriptor.random_count;
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

int MicroInputRule::input_port_count() const noexcept {
    return input_port_count_;
}

int MicroInputRule::random_count() const noexcept {
    return random_count_;
}

const std::string& MicroInputRule::library_path() const noexcept {
    return library_->path();
}

void MicroInputRule::validate_state(const std::vector<double>& state) const {
    validate_state_size(state, state_count_, "MicroInputRule");
}

void MicroInputRule::validate_params(const std::vector<double>& params) const {
    validate_params_size(params, param_count_, "MicroInputRule");
}

void MicroInputRule::apply(const std::vector<double>& input_soa,
                           int roi_count,
                           int roi,
                           std::vector<double>& state,
                           const std::vector<double>& params,
                           std::vector<RandomStreamBinding>& random_streams,
                           double start_time,
                           double stop_time,
                           const std::vector<int>& input_port_bases,
                           mind_sim::micro::sim::MicroEventTable& events,
                           const std::vector<int>& read_input_offsets) const {
    mind_sim::mind_mod::AbiEventWriter event_writer{
        .user = &events,
        .write = write_event,
    };
    if (random_streams.size() != static_cast<std::size_t>(random_count_)) {
        throw std::runtime_error("MicroInputRule random stream count mismatch");
    }
    std::vector<RandomRuntimeContext> random_contexts;
    random_contexts.reserve(static_cast<std::size_t>(random_count_));
    std::vector<mind_sim::mind_mod::AbiRandomStream> abi_random_streams;
    abi_random_streams.reserve(static_cast<std::size_t>(random_count_));
    for (int index = 0; index < random_count_; ++index) {
        auto& binding = random_streams[static_cast<std::size_t>(index)];
        if (!binding.rule) {
            throw std::runtime_error("MicroInputRule random stream is missing provider");
        }
        if (binding.state.size() != static_cast<std::size_t>(binding.rule->state_count())) {
            throw std::runtime_error("MicroInputRule random stream state size mismatch");
        }
        random_contexts.push_back(RandomRuntimeContext{
            .rule = binding.rule.get(),
            .state = &binding.state,
        });
        abi_random_streams.push_back(mind_sim::mind_mod::AbiRandomStream{
            .user = &random_contexts.back(),
            .uniform = random_stream_uniform,
        });
    }
    mind_sim::mind_mod::AbiMicroInputContext context{
        .input_count = input_count_,
        .roi_count = roi_count,
        .roi = roi,
        .input_soa = input_soa.data(),
        .state_count = state_count_,
        .state = state.data(),
        .param_count = param_count_,
        .params = params.data(),
        .start_time = start_time,
        .stop_time = stop_time,
        .input_port_count = input_port_count_,
        .input_port_bases = input_port_bases.data(),
        .event_writer = &event_writer,
        .read_input_offsets = read_input_offsets.data(),
        .random_count = random_count_,
        .random_streams = abi_random_streams.data(),
    };
    apply_(&context);
}

MicroOutputRule::MicroOutputRule(std::string name,
                                 std::string library_path,
                                 int exposure_count,
                                 int state_count,
                                 int param_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_micro_output_rule_apply"))),
      exposure_count_(exposure_count),
      state_count_(state_count),
      param_count_(param_count) {
    if (name_.empty()) {
        throw std::runtime_error("MicroOutputRule name must be non-empty");
    }
    validate_count(exposure_count_, "MicroOutputRule exposure_count");
    validate_count(state_count_, "MicroOutputRule state_count");
    validate_count(param_count_, "MicroOutputRule param_count");
    if (exposure_count_ == 0) {
        throw std::runtime_error("MicroOutputRule exposure_count must be positive");
    }
}

MicroOutputRule::MicroOutputRule(std::string library_path)
    : library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<mind_sim::mind_mod::MicroOutputApplyFn>(
          library_->symbol("mind_micro_output_rule_apply"))) {
    const auto& descriptor =
        descriptor_of(*library_, mind_sim::mind_mod::AbiRuleKind::MicroOutput, "MicroOutputRule");
    name_ = descriptor.name;
    exposure_count_ = descriptor.write_count;
    state_count_ = descriptor.state_count;
    param_count_ = descriptor.param_count;
}

const std::string& MicroOutputRule::name() const noexcept {
    return name_;
}

int MicroOutputRule::exposure_count() const noexcept {
    return exposure_count_;
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

void MicroOutputRule::validate_state(const std::vector<double>& state) const {
    validate_state_size(state, state_count_, "MicroOutputRule");
}

void MicroOutputRule::validate_params(const std::vector<double>& params) const {
    validate_params_size(params, param_count_, "MicroOutputRule");
}

void MicroOutputRule::apply(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
                            std::vector<double>& exposure_soa,
                            int roi_count,
                            int roi,
                            std::vector<double>& state,
                            const std::vector<double>& params,
                            double start_time,
                            double stop_time,
                            const std::vector<int>& write_exposure_offsets) const {
    mind_sim::mind_mod::AbiSpikeTable spike_table{
        .time = spikes.time,
        .gid = spikes.gid,
        .size = static_cast<int>(spikes.size()),
    };
    mind_sim::mind_mod::AbiMicroOutputContext context{
        .exposure_count = exposure_count_,
        .spikes = &spike_table,
        .roi_count = roi_count,
        .roi = roi,
        .exposure_soa = exposure_soa.data(),
        .state_count = state_count_,
        .state = state.data(),
        .param_count = param_count_,
        .params = params.data(),
        .start_time = start_time,
        .stop_time = stop_time,
        .write_exposure_offsets = write_exposure_offsets.data(),
    };
    apply_(&context);
}

}  // namespace mind_sim::bridge::sim
