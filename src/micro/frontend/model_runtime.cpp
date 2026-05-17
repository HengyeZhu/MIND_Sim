#include "micro/frontend/model.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"

#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
#include "coreneuron/gpu/nrn_acc_manager.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
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
        auto& nt = core_neuron_data_->threads.front();
        auto values = nt.actual_v();
        std::fill(values.begin(), values.end(), v_init);
        nt._t = 0.0;
        core_neuron_data_->bind();
    }
    if (!runtime_backend_) {
        runtime_backend_ = std::make_unique<mind_sim::micro::sim::MicroRuntime>(*core_neuron_data_);
    }
    runtime_backend_->finitialize(v_init);
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
    (void) runtime_backend_->advance_window(t_, t_ + runtime);
    t_ += runtime;
    return 0;
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

    std::vector<void*> vpr(refs.size(), nullptr);
    std::vector<double*> gather;
    std::vector<double*> varrays;
    gather.reserve(refs.size());
    varrays.reserve(sample_buffers.size());
    for (std::size_t i = 0; i < refs.size(); ++i) {
        if (sample_buffers[i] == nullptr) {
            throw std::runtime_error("recorded continue_run received null sample buffer");
        }
        gather.push_back(variable_pointer(refs[i]));
        varrays.push_back(sample_buffers[i]);
    }

    auto& nt = core_neuron_data_->threads.front();
    if (nt.trajec_requests != nullptr) {
        throw std::runtime_error("CoreNEURON trajectory recording is already active");
    }
    coreneuron::TrajectoryRequests trajectory{};
    trajectory.vpr = vpr.data();
    trajectory.scatter = nullptr;
    trajectory.varrays = varrays.data();
    trajectory.gather = gather.data();
    trajectory.n_pr = static_cast<int>(refs.size());
    trajectory.n_trajec = static_cast<int>(refs.size());
    trajectory.bsize = sample_count;
    trajectory.vsize = 0;

    const bool use_gpu =
        core_neuron_data_->device_config.kind == mind_sim::micro::sim::MicroDeviceKind::Gpu;
#if !defined(MIND_SIM_ENABLE_GPU) || !defined(CORENEURON_ENABLE_GPU)
    if (use_gpu) {
        throw std::runtime_error(
            "MIND_Sim was built without CoreNEURON GPU support; rebuild with -DMIND_SIM_ENABLE_GPU=ON");
    }
#endif

    nt.trajec_requests = &trajectory;
    try {
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (use_gpu && core_neuron_data_->gpu_device_runtime_active) {
            core_neuron_data_->bind();
            coreneuron::nrn_nthread = static_cast<int>(core_neuron_data_->threads.size());
            coreneuron::nrn_threads = core_neuron_data_->threads.data();
            coreneuron::setup_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                            coreneuron::nrn_nthread);
        }
#endif
        continue_run(runtime);
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (use_gpu && core_neuron_data_->gpu_device_runtime_active) {
            core_neuron_data_->bind();
            coreneuron::nrn_nthread = static_cast<int>(core_neuron_data_->threads.size());
            coreneuron::nrn_threads = core_neuron_data_->threads.data();
            coreneuron::update_trajectory_requests_on_host(coreneuron::nrn_threads,
                                                           coreneuron::nrn_nthread);
            coreneuron::delete_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                             coreneuron::nrn_nthread);
        }
#endif
    } catch (...) {
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (use_gpu && core_neuron_data_->gpu_device_runtime_active) {
            try {
                core_neuron_data_->bind();
                coreneuron::nrn_nthread = static_cast<int>(core_neuron_data_->threads.size());
                coreneuron::nrn_threads = core_neuron_data_->threads.data();
                coreneuron::delete_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                                 coreneuron::nrn_nthread);
            } catch (...) {
            }
        }
#endif
        nt.trajec_requests = nullptr;
        throw;
    }
    nt.trajec_requests = nullptr;
    return trajectory.vsize;
}

int MicroFrontendModel::fadvance() {
    return continue_run(dt_);
}

}  // namespace mind_sim::micro::frontend
