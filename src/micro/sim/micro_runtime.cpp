#include "micro/sim/micro_runtime.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/io/nrn_setup.hpp"
#include "coreneuron/io/output_spikes.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/nrnconf.h"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/sim/multicore.hpp"
#include "micro/sim/coreneuron_host_support.hpp"
#include "micro/sim/device.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

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

constexpr double kCoreNeuronFixedStepTolerance = 1.0e-9;

mind_sim::micro::sim::CoreNeuronData* bound_core_data = nullptr;

std::mutex& core_runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

void enqueue_event(const mind_sim::micro::sim::MicroEventTable& events,
                   std::size_t i,
                   mind_sim::micro::sim::CoreNeuronData& core_data) {
    const double event_time = events.time[i];
    const int event_index = events.index[i];
    if (event_index < 0 ||
        static_cast<std::size_t>(event_index) >= core_data.input_event_targets.size()) {
        throw std::runtime_error("micro input event index is out of range");
    }
    const auto& target = core_data.input_event_targets[static_cast<std::size_t>(event_index)];
    if (target.thread_index < 0 ||
        static_cast<std::size_t>(target.thread_index) >= core_data.threads.size()) {
        throw std::runtime_error("micro input event resolved an invalid CoreNEURON thread");
    }
    auto& thread = core_data.threads[static_cast<std::size_t>(target.thread_index)];
    if (target.input_presyn_index < 0 ||
        static_cast<std::size_t>(target.input_presyn_index) >= thread.input_presyns.size()) {
        throw std::runtime_error("micro input event resolved an invalid InputPreSyn");
    }
    auto* const runtime_threads = coreneuron::nrn_threads;
    if (runtime_threads == nullptr) {
        throw std::runtime_error("micro input event requires bound CoreNEURON runtime threads");
    }
    auto* const nt = runtime_threads + target.thread_index;
    thread.input_presyns[static_cast<std::size_t>(target.input_presyn_index)].send(
        event_time,
        coreneuron::net_cvode_instance,
        nt);
}

void enqueue_due_events(mind_sim::micro::sim::MicroEventTable& events,
                        double stop_time,
                        mind_sim::micro::sim::CoreNeuronData& core_data) {
    std::size_t write = 0;
    for (std::size_t read = 0; read < events.size(); ++read) {
        if (events.time[read] < stop_time) {
            enqueue_event(events, read, core_data);
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

void install_single_process_network_maps(mind_sim::micro::sim::CoreNeuronData& core_data) {
    const int thread_count = static_cast<int>(core_data.threads.size());
    coreneuron::gid2out.clear();
    coreneuron::gid2in.clear();
    coreneuron::neg_gid2out.assign(static_cast<std::size_t>(thread_count), {});
    coreneuron::nrnthreads_netcon_negsrcgid_tid.assign(static_cast<std::size_t>(thread_count), {});
    coreneuron::nrnthreads_netcon_srcgid.assign(static_cast<std::size_t>(thread_count), nullptr);

    coreneuron::netcon_in_presyn_order_.clear();
    coreneuron::netcon_in_presyn_order_.reserve(core_data.netcon_presyn_order.size());
    for (const auto& ref: core_data.netcon_presyn_order) {
        if (ref.thread_index < 0 ||
            static_cast<std::size_t>(ref.thread_index) >= core_data.threads.size()) {
            throw std::runtime_error("CoreNEURON NetCon order contains invalid thread index");
        }
        const auto& thread = core_data.threads[static_cast<std::size_t>(ref.thread_index)];
        if (ref.netcon_index < 0 ||
            static_cast<std::size_t>(ref.netcon_index) >= thread.netcon_storage.size()) {
            throw std::runtime_error("CoreNEURON NetCon order contains invalid NetCon index");
        }
        coreneuron::netcon_in_presyn_order_.push_back(
            const_cast<coreneuron::NetCon*>(thread.netcon_storage.data() + ref.netcon_index));
    }

    for (std::size_t tid = 0; tid < core_data.threads.size(); ++tid) {
        auto& thread = core_data.threads[tid];
        if (thread.presyn_source_gids.size() != thread.presyn_storage.size()) {
            throw std::runtime_error("CoreNEURON PreSyn source gid map is inconsistent");
        }
        if (thread.input_presyn_source_gids.size() != thread.input_presyns.size()) {
            throw std::runtime_error("CoreNEURON InputPreSyn source gid map is inconsistent");
        }
        if (thread.netcon_source_gids.size() != thread.netcon_storage.size()) {
            throw std::runtime_error("CoreNEURON NetCon source gid map is inconsistent");
        }

        for (std::size_t i = 0; i < thread.presyn_storage.size(); ++i) {
            coreneuron::gid2out[thread.presyn_source_gids[i]] = thread.presyn_storage.data() + i;
        }
        for (std::size_t i = 0; i < thread.input_presyns.size(); ++i) {
            coreneuron::gid2in[thread.input_presyn_source_gids[i]] = thread.input_presyns.data() + i;
        }
        coreneuron::nrnthreads_netcon_srcgid[tid] =
            thread.netcon_source_gids.empty() ? nullptr : thread.netcon_source_gids.data();
    }
}

int chunk_steps_for_mindelay(double mindelay, double dt) {
    if (!std::isfinite(mindelay) || mindelay <= 0.0) {
        return 1;
    }
    int steps = std::max(
        1,
        static_cast<int>(std::floor(mindelay / dt + kCoreNeuronFixedStepTolerance)));
    while (steps > 1 &&
           static_cast<double>(steps) * dt > mindelay + kCoreNeuronFixedStepTolerance) {
        --steps;
    }
    return steps;
}

}  // namespace

MicroRuntime::MicroRuntime(CoreNeuronData& core_data): core_data_(&core_data) {}

MicroRuntime::~MicroRuntime() {
    stop_window_worker();
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

    core_data_->bind();
    auto& first_thread = core_data_->threads.front();
    const int thread_count = static_cast<int>(core_data_->threads.size());

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
    coreneuron::nrn_nthread = thread_count;
    coreneuron::nrn_threads = core_data_->nrn_threads();
#ifdef _OPENMP
    if (!uses_gpu()) {
        omp_set_num_threads(thread_count);
    }
#endif
    coreneuron::dt = core_data_->dt;
    coreneuron::celsius = core_data_->celsius;
    coreneuron::rev_dt = static_cast<int>(std::llround(1.0 / core_data_->dt));
    coreneuron::t = first_thread._t;
    coreneuron::mk_netcvode();
    coreneuron::nrn_p_construct();
    coreneuron::nrn_mk_table_check();

    auto& pnttype2presyn = coreneuron::corenrn.get_pnttype2presyn();
    pnttype2presyn.assign(static_cast<std::size_t>(core_data_->mechanism_capacity()), -1);
    for (const auto& thread: core_data_->threads) {
        for (std::size_t type = 0; type < thread.pnt2presyn_ix_ptrs.size(); ++type) {
            if (thread.pnt2presyn_ix_ptrs[type] != nullptr) {
                pnttype2presyn[type] = static_cast<int>(type);
            }
        }
    }

    install_single_process_network_maps(*core_data_);
    core_data_->effective_mindelay = coreneuron::set_mindelay(coreneuron::corenrn_param.mindelay);
    bound_core_data = core_data_;
    core_globals_bound_ = true;
}

void MicroRuntime::ensure_core_globals_bound() {
    if (!core_globals_bound_ || bound_core_data != core_data_) {
        bind_core_globals();
        return;
    }
    coreneuron::nrn_nthread = static_cast<int>(core_data_->threads.size());
    coreneuron::nrn_threads = core_data_->nrn_threads();
    coreneuron::dt = core_data_->dt;
    coreneuron::celsius = core_data_->celsius;
    coreneuron::rev_dt = static_cast<int>(std::llround(1.0 / core_data_->dt));
    coreneuron::t = core_data_->threads.front()._t;
#ifdef _OPENMP
    if (!uses_gpu()) {
        omp_set_num_threads(static_cast<int>(core_data_->threads.size()));
    }
#endif
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

void MicroRuntime::ensure_window_worker_started() {
    if (window_worker_.joinable()) {
        return;
    }
    window_worker_stop_ = false;
    window_worker_has_job_ = false;
    window_worker_job_done_ = true;
    window_worker_spike_end_ = 0;
    window_worker_exception_ = nullptr;
    window_worker_ = std::thread([this]() { window_worker_loop(); });
}

void MicroRuntime::submit_worker_window(double stop_time) {
    ensure_window_worker_started();
    {
        std::lock_guard<std::mutex> lock(window_worker_mutex_);
        if (!window_worker_job_done_ || window_worker_has_job_) {
            throw std::runtime_error("MicroRuntime window worker already has an active window");
        }
        window_worker_stop_time_ = stop_time;
        window_worker_spike_end_ = 0;
        window_worker_exception_ = nullptr;
        window_worker_job_done_ = false;
        window_worker_has_job_ = true;
    }
    window_worker_cv_.notify_one();
}

void MicroRuntime::wait_worker_window() {
    std::unique_lock<std::mutex> lock(window_worker_mutex_);
    window_worker_cv_.wait(lock, [this]() { return window_worker_job_done_; });
    auto exception = window_worker_exception_;
    window_worker_exception_ = nullptr;
    if (exception != nullptr) {
        std::rethrow_exception(exception);
    }
}

void MicroRuntime::window_worker_loop() {
    for (;;) {
        double stop_time = 0.0;
        {
            std::unique_lock<std::mutex> lock(window_worker_mutex_);
            window_worker_cv_.wait(lock, [this]() {
                return window_worker_stop_ || window_worker_has_job_;
            });
            if (window_worker_stop_ && !window_worker_has_job_) {
                return;
            }
            stop_time = window_worker_stop_time_;
            window_worker_has_job_ = false;
        }

        std::exception_ptr exception = nullptr;
        std::size_t spike_end = 0;
        try {
            std::lock_guard<std::mutex> runtime_lock(core_runtime_mutex());
            run_prepared_window(stop_time);
            spike_end = coreneuron::spikevec_time.size();
        } catch (...) {
            exception = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(window_worker_mutex_);
            window_worker_spike_end_ = spike_end;
            window_worker_exception_ = exception;
            window_worker_job_done_ = true;
        }
        window_worker_cv_.notify_all();
    }
}

void MicroRuntime::stop_window_worker() noexcept {
    try {
        if (!window_worker_.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(window_worker_mutex_);
            window_worker_stop_ = true;
        }
        window_worker_cv_.notify_all();
        window_worker_.join();
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
    core_data_->sync_threads_from_runtime();
}

MicroWindowToken MicroRuntime::prepare_window(double start_time, double stop_time) {
    if (!std::isfinite(start_time) || !std::isfinite(stop_time) || stop_time < start_time) {
        throw std::runtime_error("MicroRuntime advance window requires finite ordered times");
    }
    if (window_active_) {
        throw std::runtime_error("MicroRuntime already has an active submitted window");
    }
    ensure_core_globals_bound();
    ensure_registered_mechanisms();
    ensure_private_mechanism_storage_initialized();
    ensure_device_runtime_ready();

    const auto spike_begin = coreneuron::spikevec_time.size();
    enqueue_due_events(scheduled_events_, stop_time, *core_data_);
    const double step_ratio = (stop_time - start_time) / core_data_->dt;
    if (std::abs(step_ratio - std::round(step_ratio)) > kCoreNeuronFixedStepTolerance) {
        throw std::runtime_error("MicroRuntime advance window must be an integer multiple of dt");
    }

    MicroWindowToken token{
        .start_time = start_time,
        .stop_time = stop_time,
        .spike_begin = spike_begin,
        .spike_end = spike_begin,
        .submitted = true,
        .ran_synchronously = true,
    };
    return token;
}

MicroWindowToken MicroRuntime::submit_window(double start_time, double stop_time) {
    std::unique_lock<std::mutex> lock(core_runtime_mutex());
    auto token = prepare_window(start_time, stop_time);
    if (stop_time <= start_time) {
        return token;
    }
    token.ran_synchronously = false;
    window_active_ = true;
    submit_worker_window(stop_time);
    return token;
}

void MicroRuntime::run_prepared_window(double stop_time) {
    ensure_core_globals_bound();
    const double half_dt = 0.5 * core_data_->dt;
    const double start_time = coreneuron::nrn_threads->_t;
    const int step_count = static_cast<int>(
        std::llround(std::max(0.0, stop_time - start_time) / core_data_->dt));
    if (step_count <= 0) {
        core_data_->sync_threads_from_runtime();
        return;
    }
    const double target_stop_time = start_time + static_cast<double>(step_count) * core_data_->dt;
    const auto integrate_to = [&](double target_time, bool catch_up) {
        coreneuron::ncs2nrn_integrate(target_time);
        if (!catch_up) {
            return;
        }
        int attempts = std::max(
            1,
            static_cast<int>(std::ceil(std::max(0.0, target_time - coreneuron::nrn_threads->_t) /
                                       core_data_->dt)) +
                2);
        while (attempts-- > 0 && target_time - coreneuron::nrn_threads->_t > half_dt) {
            const double before = coreneuron::nrn_threads->_t;
            coreneuron::ncs2nrn_integrate(target_time);
            if (coreneuron::nrn_threads->_t <= before) {
                break;
            }
        }
    };
    if (target_stop_time - start_time > half_dt) {
        const double mindelay = core_data_->threads.size() > 1
                                     ? core_data_->effective_mindelay
                                     : std::numeric_limits<double>::infinity();
        const double remaining = target_stop_time - start_time;
        bool chunked = false;
        if (std::isfinite(mindelay) && mindelay > 0.0 && mindelay < remaining) {
            chunked = true;
            const int chunk_steps = chunk_steps_for_mindelay(mindelay, core_data_->dt);
            const double chunk_dt = static_cast<double>(chunk_steps) * core_data_->dt;
            for (double chunk_stop = start_time + chunk_dt;
                 chunk_stop < target_stop_time - half_dt;
                 chunk_stop += chunk_dt) {
                integrate_to(chunk_stop, true);
            }
        }
        integrate_to(target_stop_time, chunked);
    }
    core_data_->sync_threads_from_runtime();
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::spike_view_from(
    std::size_t spike_begin,
    std::size_t spike_end) const {
    if (spike_end < spike_begin || spike_end > coreneuron::spikevec_time.size()) {
        throw std::runtime_error("MicroRuntime spike window is inconsistent");
    }
    const auto spike_count = spike_end - spike_begin;
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
            wait_worker_window();
            token.spike_end = window_worker_spike_end_;
            window_active_ = false;
        } catch (...) {
            window_active_ = false;
            throw;
        }
    }
    return spike_view_from(token.spike_begin, token.spike_end);
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::advance_window(double start_time,
                                                                       double stop_time) {
    std::lock_guard<std::mutex> runtime_lock(core_runtime_mutex());
    auto token = prepare_window(start_time, stop_time);
    if (stop_time > start_time) {
        run_prepared_window(stop_time);
        token.spike_end = coreneuron::spikevec_time.size();
    }
    return spike_view_from(token.spike_begin, token.spike_end);
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
            coreneuron::nrn_threads = core_data.nrn_threads();
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
