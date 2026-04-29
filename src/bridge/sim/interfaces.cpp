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

}  // namespace

MicroInputRule::MicroInputRule(std::string name,
                               std::string library_path,
                               int input_count,
                               int state_count,
                               int param_count,
                               int input_port_count)
    : name_(std::move(name)),
      library_(mind_sim::utils::load_dynamic_library(std::move(library_path))),
      apply_(reinterpret_cast<decltype(apply_)>(library_->symbol("mind_micro_input_rule_apply"))),
      input_count_(input_count),
      state_count_(state_count),
      param_count_(param_count),
      input_port_count_(input_port_count) {
    if (name_.empty()) {
        throw std::runtime_error("MicroInputRule name must be non-empty");
    }
    validate_count(input_count_, "MicroInputRule input_count");
    validate_count(state_count_, "MicroInputRule state_count");
    validate_count(param_count_, "MicroInputRule param_count");
    validate_count(input_port_count_, "MicroInputRule input_port_count");
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

const std::string& MicroInputRule::library_path() const noexcept {
    return library_->path();
}

void MicroInputRule::validate_state(const std::vector<double>& state) const {
    validate_state_size(state, state_count_, "MicroInputRule");
}

void MicroInputRule::validate_params(const std::vector<double>& params) const {
    validate_params_size(params, param_count_, "MicroInputRule");
}

void MicroInputRule::apply(const std::vector<float>& input_soa,
                           int roi_count,
                           int roi,
                           std::vector<double>& state,
                           const std::vector<double>& params,
                           double start_time,
                           double stop_time,
                           const std::vector<int>& input_port_bases,
                           mind_sim::micro::sim::MicroEventTable& events) const {
    AbiEventWriter event_writer{
        .user = &events,
        .write = write_event,
    };
    apply_(input_count_,
           roi_count,
           roi,
           input_soa.data(),
           state_count_,
           state.data(),
           param_count_,
           params.data(),
           start_time,
           stop_time,
           input_port_count_,
           input_port_bases.data(),
           &event_writer);
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
                            std::vector<float>& exposure_soa,
                            int roi_count,
                            int roi,
                            std::vector<double>& state,
                            const std::vector<double>& params,
                            double start_time,
                            double stop_time) const {
    AbiSpikeTable spike_table{
        .time = spikes.time,
        .gid = spikes.gid,
        .size = static_cast<int>(spikes.size()),
    };
    apply_(exposure_count_,
           &spike_table,
           roi_count,
           roi,
           exposure_soa.data(),
           state_count_,
           state.data(),
           param_count_,
           params.data(),
           start_time,
           stop_time);
}

}  // namespace mind_sim::bridge::sim
