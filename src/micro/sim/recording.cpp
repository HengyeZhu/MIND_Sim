#include "micro/sim/recording.hpp"

#include "coreneuron/coreneuron.hpp"

#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
#include "coreneuron/gpu/nrn_acc_manager.hpp"
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mind_sim::micro::sim {

int thread_index_for_data_pointer(const CoreNeuronData& core_data, const double* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("recording target has null value pointer");
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
    throw std::runtime_error("recording target pointer does not belong to CoreNEURON data storage");
}

CoreRecordingPlan::CoreRecordingPlan(CoreNeuronData& core_data): core_data_(&core_data) {
    threads_.resize(core_data.threads.size());
}

CoreRecordingPlan::~CoreRecordingPlan() {
    deactivate();
}

void CoreRecordingPlan::add_target(double* value, double* samples) {
    if (prepared_ || active_) {
        throw std::runtime_error("recording targets cannot be added after the plan is prepared");
    }
    if (core_data_ == nullptr) {
        throw std::runtime_error("recording plan has no CoreNeuronData");
    }
    if (samples == nullptr) {
        throw std::runtime_error("recording target has null sample buffer");
    }
    const int tid = thread_index_for_data_pointer(*core_data_, value);
    auto& recording = threads_[static_cast<std::size_t>(tid)];
    recording.vpr.push_back(nullptr);
    recording.gather.push_back(value);
    recording.varrays.push_back(samples);
}

void CoreRecordingPlan::prepare(int sample_count) {
    if (active_) {
        throw std::runtime_error("active recording plan cannot be prepared again");
    }
    if (sample_count < 0) {
        throw std::runtime_error("recording sample count must be non-negative");
    }
    for (auto& recording: threads_) {
        if (!recording.active()) {
            continue;
        }
        auto& request = recording.request;
        request.vpr = recording.vpr.data();
        request.scatter = nullptr;
        request.varrays = recording.varrays.data();
        request.gather = recording.gather.data();
        request.n_pr = static_cast<int>(recording.gather.size());
        request.n_trajec = static_cast<int>(recording.gather.size());
        request.bsize = sample_count;
        request.vsize = 0;
    }
    prepared_ = true;
}

bool CoreRecordingPlan::has_targets() const noexcept {
    for (const auto& recording: threads_) {
        if (recording.active()) {
            return true;
        }
    }
    return false;
}

bool CoreRecordingPlan::active() const noexcept {
    return active_;
}

void CoreRecordingPlan::activate() {
    if (active_) {
        return;
    }
    if (!prepared_) {
        throw std::runtime_error("recording plan must be prepared before activation");
    }
    if (core_data_ == nullptr) {
        throw std::runtime_error("recording plan has no CoreNeuronData");
    }
    if (!has_targets()) {
        return;
    }

    try {
        for (std::size_t tid = 0; tid < threads_.size(); ++tid) {
            auto& recording = threads_[tid];
            if (!recording.active()) {
                continue;
            }
            if (core_data_->threads[tid].trajec_requests != nullptr) {
                throw std::runtime_error("CoreNEURON trajectory recording is already active");
            }
            core_data_->threads[tid].trajec_requests = &recording.request;
        }
        active_ = true;

        core_data_->bind();
        coreneuron::nrn_nthread = static_cast<int>(core_data_->threads.size());
        coreneuron::nrn_threads = core_data_->nrn_threads();

#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (core_data_->device_config.kind == MicroDeviceKind::Gpu) {
            if (core_data_->gpu_device_runtime_active) {
                coreneuron::setup_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                                coreneuron::nrn_nthread);
            }
            device_request_active_ = true;
        }
#endif
    } catch (...) {
        deactivate();
        throw;
    }
}

void CoreRecordingPlan::update_host() {
    if (!active_ || core_data_ == nullptr) {
        return;
    }
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
    if (core_data_->device_config.kind == MicroDeviceKind::Gpu &&
        core_data_->gpu_device_runtime_active) {
        core_data_->bind();
        coreneuron::nrn_nthread = static_cast<int>(core_data_->threads.size());
        coreneuron::nrn_threads = core_data_->nrn_threads();
        coreneuron::update_trajectory_requests_on_host(coreneuron::nrn_threads,
                                                       coreneuron::nrn_nthread);
    }
#endif
}

void CoreRecordingPlan::deactivate() noexcept {
    if (!active_ || core_data_ == nullptr) {
        active_ = false;
        device_request_active_ = false;
        return;
    }
    try {
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
        if (device_request_active_ &&
            core_data_->device_config.kind == MicroDeviceKind::Gpu &&
            core_data_->gpu_device_runtime_active) {
            core_data_->bind();
            coreneuron::nrn_nthread = static_cast<int>(core_data_->threads.size());
            coreneuron::nrn_threads = core_data_->nrn_threads();
            coreneuron::delete_trajectory_requests_on_device(coreneuron::nrn_threads,
                                                             coreneuron::nrn_nthread);
        }
#endif
        for (std::size_t tid = 0; tid < threads_.size() && tid < core_data_->threads.size(); ++tid) {
            if (!threads_[tid].active()) {
                continue;
            }
            core_data_->threads[tid].trajec_requests = nullptr;
            if (tid < core_data_->runtime_threads.size()) {
                core_data_->runtime_threads[tid].trajec_requests = nullptr;
            }
        }
    } catch (...) {
    }
    active_ = false;
    device_request_active_ = false;
}

int CoreRecordingPlan::recorded_sample_count() const {
    int recorded = -1;
    for (const auto& recording: threads_) {
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

void CoreRecordingPlan::validate_recorded_sample_count(int expected) const {
    const int recorded = recorded_sample_count();
    if (recorded != expected) {
        throw std::runtime_error("CoreNEURON trajectory sample count mismatch: expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(recorded));
    }
}

}  // namespace mind_sim::micro::sim
