#include "micro/sim/micro_runtime.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/io/output_spikes.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/sim/multicore.hpp"

#include <cmath>
#include <stdexcept>

namespace coreneuron {
void ncs2nrn_integrate(double tstop);
}

namespace mind_sim::micro::sim {

namespace {

mind_sim::micro::sim::CoreNeuronData* bound_core_data = nullptr;

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

mind_sim::micro::sim::MicroEventTable& MicroRuntime::scheduled_events() noexcept {
    return scheduled_events_;
}

void MicroRuntime::bind_core_globals() {
    if (core_data_->threads.empty()) {
        throw std::runtime_error("MicroRuntime requires non-empty CoreNeuronData");
    }
    if (core_data_->threads.size() != 1) {
        throw std::runtime_error("MicroRuntime currently supports one CoreNEURON CPU thread");
    }

    core_data_->bind();
    auto& thread = core_data_->threads.front();

    coreneuron::nrn_configure_embedded_cpu_runtime();
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

void MicroRuntime::finitialize(double voltage) {
    ensure_core_globals_bound();
    ensure_registered_mechanisms();
    initialize_private_mechanism_storage();
    private_mechanism_storage_checked_ = true;
    coreneuron::spikevec_time.clear();
    coreneuron::spikevec_gid.clear();
    coreneuron::nrn_finitialize(1, voltage);
}

mind_sim::micro::sim::MicroSpikeTableView MicroRuntime::advance_window(double start_time,
                                                                       double stop_time) {
    if (!std::isfinite(start_time) || !std::isfinite(stop_time) || stop_time < start_time) {
        throw std::runtime_error("MicroRuntime advance window requires finite ordered times");
    }
    ensure_core_globals_bound();
    ensure_registered_mechanisms();
    ensure_private_mechanism_storage_initialized();

    const auto spike_begin = coreneuron::spikevec_time.size();
    auto& thread = core_data_->threads.front();
    enqueue_due_events(scheduled_events_, stop_time, thread);
    const double step_ratio = (stop_time - start_time) / core_data_->dt;
    if (std::abs(step_ratio - std::round(step_ratio)) > 1e-9) {
        throw std::runtime_error("MicroRuntime advance window must be an integer multiple of dt");
    }
    if (stop_time > start_time) {
        coreneuron::ncs2nrn_integrate(stop_time);
    }
    thread._t = coreneuron::nrn_threads->_t;

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

}  // namespace mind_sim::micro::sim
