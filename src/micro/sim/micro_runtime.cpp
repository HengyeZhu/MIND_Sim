#include "micro/sim/micro_runtime.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/io/nrn_setup.hpp"
#include "coreneuron/io/output_spikes.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/nrnconf.h"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/sim/multicore.hpp"
#include "micro/sim/device.hpp"

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>

#if defined(MIND_SIM_ENABLE_GPU)
#include "coreneuron/gpu/nrn_acc_manager.hpp"
#include "coreneuron/utils/offload.hpp"
#include "coreneuron/utils/randoms/nrnran123.h"
#endif

namespace coreneuron {
void ncs2nrn_integrate(double tstop);
}

namespace mind_sim::micro::sim {

namespace {

mind_sim::micro::sim::CoreNeuronData* bound_core_data = nullptr;

std::mutex& core_runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

void enqueue_event(const mind_sim::micro::sim::MicroEventTable& events,
                   std::size_t i,
                   mind_sim::micro::sim::CoreNeuronThread& thread) {
    auto* const nt = &thread;
    const double event_time = events.time[i];
    const int event_index = events.index[i];
    thread.input_presyns[static_cast<std::size_t>(event_index)].send(
        event_time,
        coreneuron::net_cvode_instance,
        nt);
}

void enqueue_due_events(mind_sim::micro::sim::MicroEventTable& events,
                        double stop_time,
                        mind_sim::micro::sim::CoreNeuronThread& thread) {
    std::size_t write = 0;
    for (std::size_t read = 0; read < events.size(); ++read) {
        if (events.time[read] < stop_time) {
            enqueue_event(events, read, thread);
            continue;
        }
        if (write != read) {
            events.time[write] = events.time[read];
            events.index[write] = events.index[read];
        }
        ++write;
    }
    events.resize(write);
}

}  // namespace

MicroRuntime::MicroRuntime(CoreNeuronData& core_data): core_data_(&core_data) {}

MicroRuntime::~MicroRuntime() {
    stop_gpu_worker();
    release_runtime_ref();
}

mind_sim::micro::sim::MicroEventTable& MicroRuntime::scheduled_events() noexcept {
    return scheduled_events_;
}

bool MicroRuntime::uses_gpu() const noexcept {
    return core_data_ != nullptr && core_data_->device_config.kind == MicroDeviceKind::Gpu;
}

void MicroRuntime::bind_core_globals() {
    if (core_data_->threads.empty()) {
        throw std::runtime_error("MicroRuntime requires non-empty CoreNeuronData");
    }
    if (core_data_->threads.size() != 1) {
        throw std::runtime_error("MicroRuntime currently supports one CoreNEURON thread");
    }

    core_data_->bind();
    auto& thread = core_data_->threads.front();

    if (uses_gpu()) {
#ifdef MIND_SIM_ENABLE_GPU
        coreneuron::nrn_configure_embedded_gpu_runtime(
            1,
            cell_permute_for_device(core_data_->device_config.kind));
#else
        throw std::runtime_error(
            "MIND_Sim was built without GPU support; rebuild with -DMIND_SIM_ENABLE_GPU=ON");
#endif
    } else {
        coreneuron::nrn_configure_embedded_cpu_runtime();
    }
    coreneuron::nrn_nthread = 1;
    coreneuron::nrn_threads = &thread;
    coreneuron::dt = core_data_->dt;
    coreneuron::celsius = core_data_->celsius;
    coreneuron::rev_dt = static_cast<int>(std::llround(1.0 / core_data_->dt));
    coreneuron::t = thread._t;
    coreneuron::mk_netcvode();
    coreneuron::nrn_p_construct();
    coreneuron::nrn_mk_table_check();

    auto& pnttype2presyn = coreneuron::corenrn.get_pnttype2presyn();
    pnttype2presyn.assign(static_cast<std::size_t>(core_data_->mechanism_capacity()), -1);
    for (std::size_t type = 0; type < thread.pnt2presyn_ix_ptrs.size(); ++type) {
        if (thread.pnt2presyn_ix_ptrs[type] != nullptr) {
            pnttype2presyn[type] = static_cast<int>(type);
        }
    }

    coreneuron::netcon_in_presyn_order_.clear();
    coreneuron::netcon_in_presyn_order_.reserve(thread.netcon_presyn_order.size());
    for (int netcon_index: thread.netcon_presyn_order) {
        coreneuron::netcon_in_presyn_order_.push_back(
            thread.netcon_storage.data() + netcon_index);
    }
    bound_core_data = core_data_;
    core_globals_bound_ = true;
}

void MicroRuntime::ensure_core_globals_bound() {
    if (!core_globals_bound_ || bound_core_data != core_data_) {
        bind_core_globals();
        return;
    }
    auto& thread = core_data_->threads.front();
    coreneuron::nrn_nthread = 1;
    coreneuron::nrn_threads = &thread;
    coreneuron::dt = core_data_->dt;
    coreneuron::celsius = core_data_->celsius;
    coreneuron::rev_dt = static_cast<int>(std::llround(1.0 / core_data_->dt));
    coreneuron::t = thread._t;
}

void MicroRuntime::require_registered_mechanisms() const {
    auto& funcs = coreneuron::corenrn.get_memb_funcs();
    for (const auto& thread: core_data_->threads) {
        for (const auto& tml: thread.tml_storage) {
            const int type = tml.index;
            if (type < 0 || static_cast<std::size_t>(type) >= funcs.size()) {
                throw std::runtime_error(
                    "CoreNEURON mechanism table is not registered for MIND_Sim runtime type " +
                    std::to_string(type));
            }
            const auto& fn = funcs[static_cast<std::size_t>(type)];
            if (fn.current == nullptr && fn.state == nullptr && fn.initialize == nullptr) {
                throw std::runtime_error(
                    "CoreNEURON mechanism functions are missing for MIND_Sim runtime type " +
                    std::to_string(type));
            }
        }
    }
}

void MicroRuntime::ensure_registered_mechanisms() {
    if (mechanisms_checked_) {
        return;
    }
    require_registered_mechanisms();
    mechanisms_checked_ = true;
}

void MicroRuntime::initialize_private_mechanism_storage() {
    auto& funcs = coreneuron::corenrn.get_memb_funcs();
    for (auto& thread: core_data_->threads) {
        for (auto& memb_list: thread.memb_lists) {
            auto& func = funcs[static_cast<std::size_t>(memb_list.type)];
            if (func.private_constructor != nullptr && memb_list.ml.instance == nullptr) {
                func.private_constructor(&thread, &memb_list.ml, memb_list.type);
            }
        }
    }
}

void MicroRuntime::ensure_private_mechanism_storage_initialized() {
    if (private_mechanism_storage_checked_) {
        return;
    }
    require_private_mechanism_storage_initialized();
    private_mechanism_storage_checked_ = true;
}

void MicroRuntime::require_private_mechanism_storage_initialized() const {
    auto& funcs = coreneuron::corenrn.get_memb_funcs();
    for (const auto& thread: core_data_->threads) {
        for (const auto& memb_list: thread.memb_lists) {
            const auto& func = funcs[static_cast<std::size_t>(memb_list.type)];
            if (func.private_constructor != nullptr && memb_list.ml.instance == nullptr) {
                throw std::runtime_error("MicroRuntime advance_window requires an initialized CoreNEURON model");
            }
        }
    }
}

void MicroRuntime::ensure_device_runtime_ready() {
    if (!uses_gpu()) {
        for (auto& thread: core_data_->threads) {
            thread.compute_gpu = 0;
        }
        return;
    }
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
    if (runtime_ref_held_) {
        return;
    }
    ensure_core_globals_bound();
    if (!core_data_->gpu_device_runtime_active) {
        coreneuron::init_gpu();
        coreneuron::cnrn_target_copyin(&coreneuron::celsius);
        coreneuron::cnrn_target_copyin(&coreneuron::pi);
        coreneuron::cnrn_target_copyin(&coreneuron::secondorder);
        coreneuron::nrnran123_initialise_global_state_on_device();
        coreneuron::setup_nrnthreads_on_device(coreneuron::nrn_threads, coreneuron::nrn_nthread);
        coreneuron::allocate_data_in_mechanism_nrn_init();
        core_data_->gpu_device_runtime_active = true;
    }
    ++core_data_->gpu_runtime_ref_count;
    runtime_ref_held_ = true;
#else
    throw std::runtime_error(
        "MIND_Sim was built without CoreNEURON GPU support; rebuild with -DMIND_SIM_ENABLE_GPU=ON");
#endif
}

void MicroRuntime::release_runtime_ref() noexcept {
    if (!runtime_ref_held_ || core_data_ == nullptr) {
        return;
    }
    runtime_ref_held_ = false;
    if (core_data_->gpu_runtime_ref_count > 0) {
        --core_data_->gpu_runtime_ref_count;
    }
    if (core_data_->gpu_runtime_ref_count <= 0) {
        release_core_neuron_device_runtime(*core_data_);
    }
}

void MicroRuntime::ensure_gpu_worker_started() {
    if (gpu_worker_.joinable()) {
        return;
    }
    gpu_worker_stop_ = false;
    gpu_worker_has_job_ = false;
    gpu_worker_job_done_ = true;
    gpu_worker_exception_ = nullptr;
    gpu_worker_ = std::thread([this]() { gpu_worker_loop(); });
}

void MicroRuntime::submit_gpu_worker_window(double stop_time) {
    ensure_gpu_worker_started();
    {
        std::lock_guard<std::mutex> lock(gpu_worker_mutex_);
        if (!gpu_worker_job_done_ || gpu_worker_has_job_) {
            throw std::runtime_error("MicroRuntime gpu worker already has an active window");
        }
        gpu_worker_stop_time_ = stop_time;
        gpu_worker_exception_ = nullptr;
        gpu_worker_job_done_ = false;
        gpu_worker_has_job_ = true;
    }
    gpu_worker_cv_.notify_one();
}

void MicroRuntime::wait_gpu_worker_window() {
    std::unique_lock<std::mutex> lock(gpu_worker_mutex_);
    gpu_worker_cv_.wait(lock, [this]() { return gpu_worker_job_done_; });
    auto exception = gpu_worker_exception_;
    gpu_worker_exception_ = nullptr;
    if (exception != nullptr) {
        std::rethrow_exception(exception);
    }
}

void MicroRuntime::gpu_worker_loop() {
    for (;;) {
        double stop_time = 0.0;
        {
            std::unique_lock<std::mutex> lock(gpu_worker_mutex_);
            gpu_worker_cv_.wait(lock, [this]() { return gpu_worker_stop_ || gpu_worker_has_job_; });
            if (gpu_worker_stop_ && !gpu_worker_has_job_) {
                return;
            }
            stop_time = gpu_worker_stop_time_;
            gpu_worker_has_job_ = false;
        }

        std::exception_ptr exception = nullptr;
        try {
            std::lock_guard<std::mutex> runtime_lock(core_runtime_mutex());
            run_prepared_window(stop_time);
        } catch (...) {
            exception = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(gpu_worker_mutex_);
            gpu_worker_exception_ = exception;
            gpu_worker_job_done_ = true;
        }
        gpu_worker_cv_.notify_all();
    }
}

void MicroRuntime::stop_gpu_worker() noexcept {
    try {
        if (!gpu_worker_.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(gpu_worker_mutex_);
            gpu_worker_stop_ = true;
        }
        gpu_worker_cv_.notify_all();
        gpu_worker_.join();
    } catch (...) {
    }
}

void MicroRuntime::finitialize(double voltage) {
    ensure_core_globals_bound();
    ensure_registered_mechanisms();
    initialize_private_mechanism_storage();
    private_mechanism_storage_checked_ = true;
    ensure_device_runtime_ready();
    coreneuron::spikevec_time.clear();
    coreneuron::spikevec_gid.clear();
    coreneuron::nrn_finitialize(1, voltage);
}

MicroWindowToken MicroRuntime::submit_window(double start_time, double stop_time) {
    if (!std::isfinite(start_time) || !std::isfinite(stop_time) || stop_time < start_time) {
        throw std::runtime_error("MicroRuntime advance window requires finite ordered times");
    }
    if (window_active_) {
        throw std::runtime_error("MicroRuntime already has an active submitted window");
    }
    std::unique_lock<std::mutex> lock(core_runtime_mutex(), std::defer_lock);
    if (uses_gpu()) {
        lock.lock();
    }
    ensure_core_globals_bound();
    ensure_registered_mechanisms();
    ensure_private_mechanism_storage_initialized();
    ensure_device_runtime_ready();

    const auto spike_begin = coreneuron::spikevec_time.size();
    auto& thread = core_data_->threads.front();
    enqueue_due_events(scheduled_events_, stop_time, thread);
    const double step_ratio = (stop_time - start_time) / core_data_->dt;
    if (std::abs(step_ratio - std::round(step_ratio)) > 1e-9) {
        throw std::runtime_error("MicroRuntime advance window must be an integer multiple of dt");
    }

    MicroWindowToken token{
        .start_time = start_time,
        .stop_time = stop_time,
        .spike_begin = spike_begin,
        .submitted = true,
        .ran_synchronously = true,
    };
    if (stop_time <= start_time || !uses_gpu()) {
        run_prepared_window(stop_time);
        return token;
    }

    token.ran_synchronously = false;
    window_active_ = true;
    submit_gpu_worker_window(stop_time);
    return token;
}

void MicroRuntime::run_prepared_window(double stop_time) {
    ensure_core_globals_bound();
    if (stop_time > coreneuron::nrn_threads->_t) {
        coreneuron::ncs2nrn_integrate(stop_time);
    }
    auto& thread = core_data_->threads.front();
    thread._t = coreneuron::nrn_threads->_t;
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::spike_view_from(
    std::size_t spike_begin) const {
    const auto spike_count = coreneuron::spikevec_time.size() - spike_begin;
    if (spike_count == 0) {
        return {};
    }
    return mind_sim::micro::sim::MicroSpikeTableView{
        .time = coreneuron::spikevec_time.data() + spike_begin,
        .gid = coreneuron::spikevec_gid.data() + spike_begin,
        .count = spike_count,
    };
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::finish_window(MicroWindowToken& token) {
    if (!token.submitted) {
        throw std::runtime_error("MicroRuntime finish_window received an empty token");
    }
    if (!token.ran_synchronously) {
        try {
            wait_gpu_worker_window();
            window_active_ = false;
        } catch (...) {
            window_active_ = false;
            throw;
        }
    }
    return spike_view_from(token.spike_begin);
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::advance_window(double start_time,
                                                                       double stop_time) {
    auto token = submit_window(start_time, stop_time);
    return finish_window(token);
}

void release_core_neuron_device_runtime(CoreNeuronData& core_data) noexcept {
#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
    if (!core_data.gpu_device_runtime_active) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(core_runtime_mutex());
        if (!core_data.gpu_device_runtime_active) {
            return;
        }
        core_data.bind();
        if (!core_data.threads.empty()) {
            coreneuron::nrn_nthread = static_cast<int>(core_data.threads.size());
            coreneuron::nrn_threads = core_data.threads.data();
            coreneuron::delete_nrnthreads_on_device(coreneuron::nrn_threads,
                                                    coreneuron::nrn_nthread);
            for (auto& thread: core_data.threads) {
                thread.compute_gpu = 0;
            }
        }
        coreneuron::nrnran123_destroy_global_state_on_device();
        coreneuron::cnrn_target_delete(&coreneuron::secondorder);
        coreneuron::cnrn_target_delete(&coreneuron::pi);
        coreneuron::cnrn_target_delete(&coreneuron::celsius);
        core_data.gpu_device_runtime_active = false;
        core_data.gpu_runtime_ref_count = 0;
    } catch (...) {
        core_data.gpu_device_runtime_active = false;
        core_data.gpu_runtime_ref_count = 0;
    }
#else
    (void) core_data;
#endif
}

}  // namespace mind_sim::micro::sim
