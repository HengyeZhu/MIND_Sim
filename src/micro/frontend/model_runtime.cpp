#include "micro/frontend/model.hpp"

#include "micro/sim/recording.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mind_sim::micro::frontend {

int MicroFrontendModel::build_microcircuit() {
    require_morphology();
    rebuild_core_mechanisms();
    rebuild_core_network();
    microcircuit_built_ = true;
    core_initialized_ = false;
    runtime_backend_.reset();
    return 0;
}

int MicroFrontendModel::finitialize(double v_init) {
    require_morphology();
    if (!microcircuit_built_) {
        throw std::runtime_error("finitialize requires build_microcircuit first");
    }
    mind_micro_biophysical::validate_section_property_value(
        mind_micro_biophysical::ObjectOpKind::VInit,
        v_init);
    t_ = 0.0;
    if (!core_neuron_data_->threads.empty()) {
        for (auto& nt: core_neuron_data_->threads) {
            auto values = nt.actual_v();
            std::fill(values.begin(), values.end(), v_init);
            nt._t = 0.0;
        }
        core_neuron_data_->bind();
    }
    if (!runtime_backend_) {
        runtime_backend_ = std::make_unique<mind_sim::micro::sim::MicroRuntime>(*core_neuron_data_);
    }
    runtime_backend_->finitialize(v_init);
    recorded_spikes_.clear();
    core_initialized_ = true;
    return 0;
}

int MicroFrontendModel::run(double tstop) {
    if (!std::isfinite(tstop) || tstop < t_) {
        throw std::runtime_error("run target time must be finite and >= current time");
    }
    return continue_run(tstop - t_);
}

int MicroFrontendModel::continue_run(double runtime) {
    if (!std::isfinite(runtime) || runtime < 0.0) {
        throw std::runtime_error("continue_run duration must be finite and non-negative");
    }
    if (!core_initialized_) {
        throw std::runtime_error("continue_run requires finitialize first");
    }
    if (!runtime_backend_) {
        runtime_backend_ = std::make_unique<mind_sim::micro::sim::MicroRuntime>(*core_neuron_data_);
    }
    const auto spikes = runtime_backend_->advance_window(t_, t_ + runtime);
    recorded_spikes_.append_view(spikes);
    t_ += runtime;
    return 0;
}

const std::vector<double>& MicroFrontendModel::spike_times() const noexcept {
    return recorded_spikes_.time;
}

const std::vector<int>& MicroFrontendModel::spike_gids() const noexcept {
    return recorded_spikes_.gid;
}

void MicroFrontendModel::clear_spikes() noexcept {
    recorded_spikes_.clear();
}

void MicroFrontendModel::schedule_spike_input_event(int spike_input_id, double time) {
    if (!std::isfinite(time)) {
        throw std::runtime_error("spike input event time must be finite");
    }
    if (!core_initialized_) {
        throw std::runtime_error("spike input event requires finitialize first");
    }
    const int runtime_index = network_registry_.spike_input_runtime_index(spike_input_id);
    if (runtime_index < 0) {
        throw std::runtime_error("spike input event requires build_microcircuit first");
    }
    if (!runtime_backend_) {
        runtime_backend_ = std::make_unique<mind_sim::micro::sim::MicroRuntime>(*core_neuron_data_);
    }
    auto& events = runtime_backend_->scheduled_events();
    events.time.push_back(time);
    events.index.push_back(runtime_index);
}

void MicroFrontendModel::schedule_netcon_event(int connection_id, double time) {
    const int spike_input_id = network_registry_.get_netcon_source_spike_input_id(connection_id);
    if (spike_input_id < 0) {
        throw std::runtime_error("NetCon.event is currently supported only for spike-input sources");
    }
    schedule_spike_input_event(spike_input_id, time);
}

int MicroFrontendModel::continue_run_with_recording(double runtime,
                                                    const std::vector<VariableRef>& refs,
                                                    const std::vector<double*>& sample_buffers,
                                                    int sample_count) {
    if (!std::isfinite(runtime) || runtime < 0.0) {
        throw std::runtime_error("recorded continue_run duration must be finite and non-negative");
    }
    if (sample_count < 0) {
        throw std::runtime_error("recorded continue_run sample count must be non-negative");
    }
    if (refs.size() != sample_buffers.size()) {
        throw std::runtime_error("recorded continue_run refs and buffers size mismatch");
    }
    if (!core_initialized_) {
        throw std::runtime_error("recorded continue_run requires finitialize first");
    }
    if (core_neuron_data_->threads.empty()) {
        throw std::runtime_error("recorded continue_run requires built CoreNEURON thread data");
    }
    if (refs.empty() || sample_count == 0) {
        return continue_run(runtime);
    }

    mind_sim::micro::sim::CoreRecordingPlan recording_plan(*core_neuron_data_);
    for (std::size_t i = 0; i < refs.size(); ++i) {
        if (sample_buffers[i] == nullptr) {
            throw std::runtime_error("recorded continue_run received null sample buffer");
        }
        recording_plan.add_target(variable_pointer(refs[i]), sample_buffers[i]);
    }
    recording_plan.prepare(sample_count);

    try {
        recording_plan.activate();
        continue_run(runtime);
        recording_plan.update_host();
    } catch (...) {
        recording_plan.deactivate();
        throw;
    }
    const int recorded = recording_plan.recorded_sample_count();
    recording_plan.deactivate();
    return recorded;
}

int MicroFrontendModel::fadvance() {
    return continue_run(dt_);
}

}  // namespace mind_sim::micro::frontend
