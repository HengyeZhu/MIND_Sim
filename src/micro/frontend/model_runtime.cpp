#include "micro/frontend/model.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"

#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
#include "coreneuron/gpu/nrn_acc_manager.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mind_sim::micro::frontend {

namespace {

int thread_index_for_pointer(const mind_sim::micro::sim::CoreNeuronData& core_data,
                             const double* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    for (std::size_t tid = 0; tid < core_data.threads.size(); ++tid) {
        const auto& storage = core_data.threads[tid].data_storage;
        if (storage.empty()) {
            continue;
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(storage.data());
        const auto end = begin + storage.size() * sizeof(double);
        if (address >= begin && address < end) {
            return static_cast<int>(tid);
        }
    }
    return 0;
}

struct ThreadTrajectoryRecording {
    std::vector<void*> vpr{};
    std::vector<double*> gather{};
    std::vector<double*> varrays{};
    coreneuron::TrajectoryRequests request{};

    [[nodiscard]] bool active() const noexcept {
        return !gather.empty();
    }
};

}  // namespace

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

    std::vector<ThreadTrajectoryRecording> recordings(core_neuron_data_->threads.size());
    for (std::size_t i = 0; i < refs.size(); ++i) {
        if (sample_buffers[i] == nullptr) {
            throw std::runtime_error("recorded continue_run received null sample buffer");
        }
        double* const pointer = variable_pointer(refs[i]);
        const int tid = thread_index_for_pointer(*core_neuron_data_, pointer);
        auto& recording = recordings[static_cast<std::size_t>(tid)];
        recording.vpr.push_back(nullptr);
        recording.gather.push_back(pointer);
        recording.varrays.push_back(sample_buffers[i]);
    }

    for (std::size_t tid = 0; tid < recordings.size(); ++tid) {
        if (!recordings[tid].active()) {
            continue;
        }
        auto& nt = core_neuron_data_->threads[tid];
        if (nt.trajec_requests != nullptr) {
            throw std::runtime_error("CoreNEURON trajectory recording is already active");
        }
        auto& trajectory = recordings[tid].request;
        trajectory.vpr = recordings[tid].vpr.data();
        trajectory.scatter = nullptr;
        trajectory.varrays = recordings[tid].varrays.data();
        trajectory.gather = recordings[tid].gather.data();
        trajectory.n_pr = static_cast<int>(recordings[tid].gather.size());
        trajectory.n_trajec = static_cast<int>(recordings[tid].gather.size());
        trajectory.bsize = sample_count;
        trajectory.vsize = 0;
        nt.trajec_requests = &trajectory;
        if (tid < core_neuron_data_->runtime_threads.size()) {
            core_neuron_data_->runtime_threads[tid].trajec_requests = &trajectory;
        }
    }

    const auto clear_recording_requests = [&]() {
        for (std::size_t tid = 0; tid < recordings.size(); ++tid) {
        if (recordings[tid].active()) {
            core_neuron_data_->threads[tid].trajec_requests = nullptr;
            if (tid < core_neuron_data_->runtime_threads.size()) {
                core_neuron_data_->runtime_threads[tid].trajec_requests = nullptr;
            }
        }
        }
    };

    const bool use_gpu =
        core_neuron_data_->device_config.kind == mind_sim::micro::sim::MicroDeviceKind::Gpu;
#if !defined(MIND_SIM_ENABLE_GPU) || !defined(CORENEURON_ENABLE_GPU)
    if (use_gpu) {
        throw std::runtime_error(
            "MIND_Sim was built without CoreNEURON GPU support; rebuild with -DMIND_SIM_ENABLE_GPU=ON");
    }
#endif

    try {
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (use_gpu && core_neuron_data_->gpu_device_runtime_active) {
            core_neuron_data_->bind();
            coreneuron::nrn_nthread = static_cast<int>(core_neuron_data_->threads.size());
            coreneuron::nrn_threads = core_neuron_data_->nrn_threads();
            coreneuron::setup_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                            coreneuron::nrn_nthread);
        }
#endif
        continue_run(runtime);
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (use_gpu && core_neuron_data_->gpu_device_runtime_active) {
            core_neuron_data_->bind();
            coreneuron::nrn_nthread = static_cast<int>(core_neuron_data_->threads.size());
            coreneuron::nrn_threads = core_neuron_data_->nrn_threads();
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
                coreneuron::nrn_threads = core_neuron_data_->nrn_threads();
                coreneuron::delete_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                                 coreneuron::nrn_nthread);
            } catch (...) {
            }
        }
#endif
        clear_recording_requests();
        throw;
    }
    clear_recording_requests();

    int recorded = -1;
    for (const auto& recording: recordings) {
        if (!recording.active()) {
            continue;
        }
        if (recorded < 0) {
            recorded = recording.request.vsize;
        } else if (recorded != recording.request.vsize) {
            throw std::runtime_error("CoreNEURON trajectory recording count differs across threads");
        }
    }
    return recorded < 0 ? 0 : recorded;
}

int MicroFrontendModel::fadvance() {
    return continue_run(dt_);
}

}  // namespace mind_sim::micro::frontend
