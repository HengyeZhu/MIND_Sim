#include "cosim/hybrid_simulator.hpp"

#include "macro/sim/runtime_core.hpp"
#include "micro/sim/micro_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mind_sim::cosim {

namespace {

constexpr double kCoreNeuronFixedStepTolerance = 1.0e-9;

bool is_integer_multiple(double value, double unit) {
    const double ratio = value / unit;
    return std::abs(ratio - std::round(ratio)) <= kCoreNeuronFixedStepTolerance;
}

int integer_step_count(double value, double unit) {
    return static_cast<int>(std::llround(value / unit));
}

double canonical_micro_time(double time, double dt_micro) {
    return static_cast<double>(std::llround(time / dt_micro)) * dt_micro;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text == "1" || text == "true" || text == "on" || text == "yes";
}

using ProfileClock = std::chrono::steady_clock;

template <typename Fn>
void measure_if(bool enabled, double& seconds, Fn&& fn) {
    if (!enabled) {
        fn();
        return;
    }
    const auto start = ProfileClock::now();
    fn();
    seconds += std::chrono::duration<double>(ProfileClock::now() - start).count();
}

struct SerialProfile {
    bool enabled{false};
    double transform_seconds{0.0};
    double micro_seconds{0.0};
    double macro_seconds{0.0};
    int exchange_windows{0};
    int macro_steps{0};
};

struct AsyncProfile {
    bool enabled{false};
    double prepare_input_seconds{0.0};
    double submit_seconds{0.0};
    double wait_micro_seconds{0.0};
    double output_transform_seconds{0.0};
    double macro_seconds{0.0};
    double macro_input_seconds{0.0};
    int exchange_windows{0};
    int macro_steps{0};
};

void report_serial_profile(const SerialProfile& profile) {
    if (!profile.enabled) {
        return;
    }
    const double total =
        profile.transform_seconds + profile.micro_seconds + profile.macro_seconds;
    std::cerr
        << "MIND Sim serial profile: "
        << "transform_s=" << profile.transform_seconds
        << ", micro_s=" << profile.micro_seconds
        << ", macro_s=" << profile.macro_seconds
        << ", profiled_total_s=" << total
        << ", exchange_windows=" << profile.exchange_windows
        << ", macro_steps=" << profile.macro_steps
        << ".\n";
}

void report_async_profile(const AsyncProfile& profile) {
    if (!profile.enabled) {
        return;
    }
    const double total =
        profile.prepare_input_seconds +
        profile.submit_seconds +
        profile.wait_micro_seconds +
        profile.output_transform_seconds +
        profile.macro_seconds +
        profile.macro_input_seconds;
    std::cerr
        << "MIND Sim async profile: "
        << "prepare_input_s=" << profile.prepare_input_seconds
        << ", submit_s=" << profile.submit_seconds
        << ", wait_micro_s=" << profile.wait_micro_seconds
        << ", output_transform_s=" << profile.output_transform_seconds
        << ", macro_s=" << profile.macro_seconds
        << ", macro_input_s=" << profile.macro_input_seconds
        << ", profiled_total_s=" << total
        << ", exchange_windows=" << profile.exchange_windows
        << ", macro_steps=" << profile.macro_steps
        << ".\n";
}

double infer_micro_dt(const mind_sim::macro::frontend::Network& network) {
    const auto& circuits = network.micro_circuits();
    if (circuits.empty()) {
        throw std::runtime_error(
            "cosimulation requires at least one ROI.use_micro(); use MacroRuntime for macro-only runs");
    }

    double dt = 0.0;
    bool have_dt = false;
    for (const auto& circuit: circuits) {
        if (!circuit.core_data) {
            throw std::runtime_error("micro circuit has no CoreNEURON data");
        }
        const double circuit_dt = circuit.core_data->dt;
        if (circuit_dt <= 0.0 || !std::isfinite(circuit_dt)) {
            throw std::runtime_error("micro dt must be positive and finite");
        }
        if (!have_dt) {
            dt = circuit_dt;
            have_dt = true;
            continue;
        }
        if (dt != circuit_dt) {
            throw std::runtime_error("all attached micro circuits must use the same dt");
        }
    }
    return dt;
}

int minimum_active_connectivity_delay_steps(
    const mind_sim::macro::frontend::Network& network,
    double dt_macro) {
    const auto& connectivity = network.connectivity();
    const auto& weights = connectivity.weights();
    const auto& delays = connectivity.delays();
    if (weights.size() != delays.size()) {
        throw std::runtime_error("connectivity weights and delays have different sizes");
    }

    int min_delay_steps = std::numeric_limits<int>::max();
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (weights[index] == 0.0) {
            continue;
        }
        if (delays[index] <= 0.0) {
            throw std::runtime_error("active connectivity edges must have positive delays");
        }
        min_delay_steps = std::min(min_delay_steps, integer_step_count(delays[index], dt_macro));
    }
    return min_delay_steps == std::numeric_limits<int>::max() ? 0 : min_delay_steps;
}

struct TraceAheadSafety {
    bool safe{true};
    int required_delay_steps{0};
    int min_delay_steps{std::numeric_limits<int>::max()};
    int target_roi{-1};
    int source_roi{-1};
};

TraceAheadSafety check_micro_input_trace_ahead(
    const mind_sim::macro::sim::MacroToMacroEvaluation& evaluation,
    int exchange_step_count) {
    TraceAheadSafety out;
    out.required_delay_steps = std::max(1, (2 * exchange_step_count) - 1);
    for (const auto& graph_view: evaluation.graphs) {
        if (graph_view.graph == nullptr) {
            continue;
        }
        const auto& graph = *graph_view.graph;
        for (int target: graph_view.targets) {
            if (target < 0 ||
                static_cast<std::size_t>(target + 1) >= graph.target_edge_offsets.size()) {
                throw std::runtime_error("macro2micro delay check found an invalid target ROI");
            }
            const int begin = graph.target_edge_offsets[static_cast<std::size_t>(target)];
            const int end = graph.target_edge_offsets[static_cast<std::size_t>(target + 1)];
            for (int edge = begin; edge < end; ++edge) {
                const int delay_steps = graph.edge_delay_steps[static_cast<std::size_t>(edge)];
                if (delay_steps < out.min_delay_steps) {
                    out.min_delay_steps = delay_steps;
                    out.target_roi = target;
                    out.source_roi = graph.edge_sources[static_cast<std::size_t>(edge)];
                }
                if (delay_steps < out.required_delay_steps) {
                    out.safe = false;
                }
            }
        }
    }
    if (out.min_delay_steps == std::numeric_limits<int>::max()) {
        out.min_delay_steps = out.required_delay_steps;
    }
    return out;
}

void append_events(mind_sim::micro::sim::MicroEventTable& target,
                   mind_sim::micro::sim::MicroEventTable& source) {
    if (source.size() == 0) {
        return;
    }
    if (target.size() == 0) {
        target.time.swap(source.time);
        target.index.swap(source.index);
        return;
    }
    target.time.insert(target.time.end(), source.time.begin(), source.time.end());
    target.index.insert(target.index.end(), source.index.begin(), source.index.end());
    source.clear();
}

int thread_index_for_pointer(const mind_sim::micro::sim::CoreNeuronData& core_data,
                             const double* ptr) {
    if (ptr == nullptr) {
        return -1;
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
    return -1;
}

struct MicroRecordBinding {
    double* value{nullptr};
    std::vector<double>* samples{nullptr};
    std::size_t sample_offset{0};
    int thread_index{-1};
};

struct ThreadTrajectoryRecording {
    std::vector<MicroRecordBinding> bindings{};
    std::vector<void*> vpr{};
    std::vector<double*> gather{};
    std::vector<double*> varrays{};
    coreneuron::TrajectoryRequests request{};

    [[nodiscard]] bool active() const noexcept {
        return !gather.empty();
    }
};

struct MicroRecordPlan {
    std::vector<ThreadTrajectoryRecording> threads{};

    [[nodiscard]] bool active() const noexcept {
        for (const auto& thread: threads) {
            if (thread.active()) {
                return true;
            }
        }
        return false;
    }
};

struct RecordedMicroWindow {
    mind_sim::micro::sim::MicroSpikeTableView spikes{};
};

void clear_trace_requests(mind_sim::micro::sim::CoreNeuronData& core_data,
                          const MicroRecordPlan& plan) {
    for (std::size_t tid = 0; tid < plan.threads.size(); ++tid) {
        if (!plan.threads[tid].active()) {
            continue;
        }
        core_data.threads[tid].trajec_requests = nullptr;
        if (tid < core_data.runtime_threads.size()) {
            core_data.runtime_threads[tid].trajec_requests = nullptr;
        }
    }
}

MicroRecordPlan make_micro_record_plan(
    const mind_sim::micro::sim::CoreNeuronData& core_data,
    const std::vector<MicroRecordBinding>& recorders) {
    MicroRecordPlan plan;
    plan.threads.resize(core_data.threads.size());
    if (recorders.empty()) {
        return plan;
    }
    if (core_data.device_config.kind != mind_sim::micro::sim::MicroDeviceKind::Cpu) {
        throw std::runtime_error("hybrid micro recording currently supports CPU micro circuits");
    }
    for (const auto& target: recorders) {
        if (target.samples == nullptr) {
            throw std::runtime_error("hybrid micro recorder has no sample buffer");
        }
        if (target.thread_index < 0 ||
            static_cast<std::size_t>(target.thread_index) >= plan.threads.size()) {
            throw std::runtime_error("hybrid micro recorder pointer does not belong to this micro circuit");
        }
        auto& recording = plan.threads[static_cast<std::size_t>(target.thread_index)];
        recording.bindings.push_back(target);
        recording.vpr.push_back(nullptr);
        recording.gather.push_back(target.value);
        recording.varrays.push_back(nullptr);
    }
    for (auto& recording: plan.threads) {
        if (!recording.active()) {
            continue;
        }
        auto& trajectory = recording.request;
        trajectory.vpr = recording.vpr.data();
        trajectory.scatter = nullptr;
        trajectory.varrays = recording.varrays.data();
        trajectory.gather = recording.gather.data();
        trajectory.n_pr = static_cast<int>(recording.gather.size());
        trajectory.n_trajec = static_cast<int>(recording.gather.size());
        trajectory.vsize = 0;
    }
    return plan;
}

void arm_trace_requests(mind_sim::micro::sim::CoreNeuronData& core_data,
                        MicroRecordPlan& plan,
                        int sample_count,
                        std::size_t sample_offset) {
    if (!plan.active() || sample_count <= 0) {
        return;
    }
    for (std::size_t tid = 0; tid < plan.threads.size(); ++tid) {
        auto& recording = plan.threads[tid];
        if (!recording.active()) {
            continue;
        }
        for (std::size_t index = 0; index < recording.bindings.size(); ++index) {
            const auto& binding = recording.bindings[index];
            recording.varrays[index] =
                binding.samples->data() + binding.sample_offset + sample_offset;
        }
        auto& thread = core_data.threads[tid];
        if (thread.trajec_requests != nullptr) {
            throw std::runtime_error("CoreNEURON trajectory recording is already active");
        }
        auto& trajectory = recording.request;
        trajectory.bsize = sample_count;
        trajectory.vsize = 0;
        thread.trajec_requests = &trajectory;
        if (tid < core_data.runtime_threads.size()) {
            core_data.runtime_threads[tid].trajec_requests = &trajectory;
        }
    }
}

void validate_recorded_sample_count(const MicroRecordPlan& plan,
                                    int sample_count) {
    if (!plan.active() || sample_count <= 0) {
        return;
    }
    int recorded = -1;
    for (const auto& recording: plan.threads) {
        if (!recording.active()) {
            continue;
        }
        if (recorded < 0) {
            recorded = recording.request.vsize;
        } else if (recorded != recording.request.vsize) {
            throw std::runtime_error("CoreNEURON trajectory recording count differs across threads");
        }
    }
    if (recorded != sample_count) {
        throw std::runtime_error("CoreNEURON trajectory sample count mismatch: expected " +
                                 std::to_string(sample_count) + ", got " +
                                 std::to_string(recorded));
    }
}

RecordedMicroWindow advance_micro_window_recorded(
    mind_sim::micro::sim::MicroRuntime& runtime,
    mind_sim::micro::sim::CoreNeuronData& core_data,
    MicroRecordPlan& plan,
    double window_start_time,
    double window_stop_time,
    int sample_count,
    std::size_t sample_offset) {
    RecordedMicroWindow out;
    if (!plan.active() || sample_count <= 0) {
        out.spikes = runtime.advance_window(window_start_time, window_stop_time);
        return out;
    }
    arm_trace_requests(core_data, plan, sample_count, sample_offset);
    try {
        out.spikes = runtime.advance_window(window_start_time, window_stop_time);
    } catch (...) {
        clear_trace_requests(core_data, plan);
        throw;
    }
    clear_trace_requests(core_data, plan);
    validate_recorded_sample_count(plan, sample_count);
    return out;
}

void prepare_micro_events_for_exchange(
    const mind_sim::macro::sim::MacroToMacroEvaluation& micro_macro_to_macro_evaluation,
    int roi_count,
    int input_count,
    int output_count,
    int exchange_start,
    int exchange_stop,
    int history_step_offset,
    double dt_macro,
    const std::vector<double>& history,
    std::vector<double>& micro_input_trace_soa,
    std::vector<double>& micro_exposure_trace_soa,
    std::vector<double>& sample_input_soa,
    std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    std::vector<mind_sim::micro::sim::MicroEventTable>& prepared_events,
    std::uint64_t macro2micro_seed) {
    for (auto& events: prepared_events) {
        events.clear();
    }
    if (micro_circuits.empty()) {
        return;
    }
    const int sample_count = exchange_stop - exchange_start;
    if (sample_count <= 0) {
        return;
    }
    const auto input_stride = static_cast<std::size_t>(input_count * roi_count);
    const auto exposure_stride = static_cast<std::size_t>(output_count * roi_count);
    micro_input_trace_soa.resize(static_cast<std::size_t>(sample_count) * input_stride);
    micro_exposure_trace_soa.resize(static_cast<std::size_t>(sample_count) * exposure_stride);
    if (sample_input_soa.size() != input_stride) {
        sample_input_soa.assign(input_stride, 0.0);
    }
    const auto exposure_base =
        static_cast<std::size_t>((history_step_offset + exchange_start) %
                                 micro_macro_to_macro_evaluation.history_capacity) *
        exposure_stride;
    for (int sample = 0; sample < sample_count; ++sample) {
        apply_macro_to_macro(micro_macro_to_macro_evaluation,
                             roi_count,
                             input_count,
                             output_count,
                             history_step_offset + exchange_start + sample + 1,
                             history,
                             sample_input_soa);
        const auto base = static_cast<std::size_t>(sample) * input_stride;
        std::copy(sample_input_soa.begin(),
                  sample_input_soa.end(),
                  micro_input_trace_soa.begin() + static_cast<std::ptrdiff_t>(base));
        const auto exposure_sample_base = static_cast<std::size_t>(sample) * exposure_stride;
        std::copy(history.begin() + static_cast<std::ptrdiff_t>(exposure_base),
                  history.begin() + static_cast<std::ptrdiff_t>(exposure_base + exposure_stride),
                  micro_exposure_trace_soa.begin() + static_cast<std::ptrdiff_t>(exposure_sample_base));
    }
    const double exchange_start_time = exchange_start * dt_macro;
    const double exchange_stop_time = exchange_stop * dt_macro;
    for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
        auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
        auto& events = prepared_events[static_cast<std::size_t>(circuit_index)];
        for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
            auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
            if (!binding.input_rule) {
                continue;
            }
            binding.input_rule->apply(micro_input_trace_soa,
                                      micro_exposure_trace_soa,
                                      sample_count,
                                      dt_macro,
                                      input_count,
                                      output_count,
                                      roi_count,
                                      binding.roi_index,
                                      binding.input_state,
                                      binding.input_params,
                                      exchange_start_time,
                                      exchange_stop_time,
                                      macro2micro_seed,
                                      exchange_start,
                                      binding.macro2micro_indices,
                                      binding.macro2micro_source_ids,
                                      events,
                                      binding.target_input_offsets,
                                      binding.source_exposure_offsets);
        }
    }
}

bool source_accepts_sid(const std::vector<int>& source_sids, int sid) {
    return source_sids.empty() ||
           std::binary_search(source_sids.begin(), source_sids.end(), sid);
}

bool macro2micro_uses_source_exposure(
    const std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits) {
    for (const auto& circuit: micro_circuits) {
        for (const auto& binding: circuit.bindings) {
            if (binding.input_rule && binding.input_rule->source_exposure_count() > 0) {
                return true;
            }
        }
    }
    return false;
}

void bind_spike_views(
    const mind_sim::micro::sim::MicroSpikeTableView& spikes,
    const mind_sim::macro::frontend::MicroCircuitOwner& circuit,
    mind_sim::micro::sim::MicroSpikeTable& pending_spikes,
    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTable>>& transform_spikes,
    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTableView>>& transform_views,
    double dt_micro,
    double window_start,
    double window_stop) {
    for (auto& binding_tables: transform_spikes) {
        for (auto& table: binding_tables) {
            table.clear();
        }
    }
    for (auto& binding_views: transform_views) {
        for (auto& view: binding_views) {
            view = {};
        }
    }
    if (transform_views.empty()) {
        return;
    }
    mind_sim::micro::sim::MicroSpikeTable window_spikes;
    mind_sim::micro::sim::MicroSpikeTable next_pending_spikes;
    const auto partition_spike = [&](double spike_time, int spike_gid) {
        const double event_time = canonical_micro_time(spike_time, dt_micro);
        if (event_time < window_start) {
            return;
        }
        if (event_time >= window_stop) {
            next_pending_spikes.append(event_time, spike_gid);
            return;
        }
        window_spikes.append(event_time, spike_gid);
    };
    for (std::size_t spike = 0; spike < pending_spikes.size(); ++spike) {
        partition_spike(pending_spikes.time[spike], pending_spikes.gid[spike]);
    }
    for (std::size_t spike = 0; spike < spikes.size(); ++spike) {
        partition_spike(spikes.time[spike], spikes.gid[spike]);
    }
    window_spikes.sort_by_time_gid();
    next_pending_spikes.sort_by_time_gid();
    pending_spikes = std::move(next_pending_spikes);
    const auto assigned_spikes = window_spikes.view(0, window_spikes.size());
    for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
        const auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
        auto& binding_tables = transform_spikes[static_cast<std::size_t>(binding_index)];
        auto& binding_views = transform_views[static_cast<std::size_t>(binding_index)];
        for (int transform_index = 0; transform_index < static_cast<int>(binding.output_transforms.size());
             ++transform_index) {
            const auto& transform = binding.output_transforms[static_cast<std::size_t>(transform_index)];
            auto& table = binding_tables[static_cast<std::size_t>(transform_index)];
            auto& view = binding_views[static_cast<std::size_t>(transform_index)];
            for (std::size_t spike = 0; spike < assigned_spikes.size(); ++spike) {
                // MicroSpikeTableView keeps CoreNEURON's field name "gid"; in
                // the Python modeling API this value is the registered sid.
                if (!source_accepts_sid(transform.source_sids, assigned_spikes.gid[spike])) {
                    continue;
                }
                table.append(assigned_spikes.time[spike], assigned_spikes.gid[spike]);
            }
            table.sort_by_time_gid();
            view = table.view(0, table.size());
        }
    }
}

}  // namespace

using mind_sim::macro::sim::append_record_table;
using mind_sim::macro::sim::apply_macro_to_macro;
using mind_sim::macro::sim::aggregate_field_outputs;
using mind_sim::macro::sim::build_macro_to_macro_runtime;
using mind_sim::macro::sim::build_region_groups;
using mind_sim::macro::sim::collect_roi_owners;
using mind_sim::macro::sim::macro_to_macro_evaluation_for_targets;
using mind_sim::macro::sim::continuous_macro_rois;
using mind_sim::macro::sim::output_buffers_to_soa;
using mind_sim::macro::sim::initialize_history;
using mind_sim::macro::sim::step_neural_field;
using mind_sim::macro::sim::validate_single_roi_owner;
using mind_sim::macro::sim::write_history_slot;

Simulator::Simulator(mind_sim::macro::frontend::Network network,
                     std::uint64_t macro2micro_seed)
    : network_(std::move(network)),
      macro2micro_seed_(macro2micro_seed) {
    configure_timing();
    validate_timing();
    exchange_step_count_ = integer_step_count(exchange_window_, dt_macro_);
}

void Simulator::configure_timing() {
    dt_micro_ = infer_micro_dt(network_);
    dt_macro_ = network_.dt();
    exchange_window_ = network_.exchange_window();
}

void Simulator::validate_timing() const {
    if (dt_micro_ <= 0.0 || dt_macro_ <= 0.0 || exchange_window_ <= 0.0) {
        throw std::runtime_error(
            "micro dt, macro dt, and exchange_window must be positive; call ms.macro.dt(...) and ms.macro.exchange_window(...) before cosimulation");
    }
    if (!std::isfinite(dt_micro_) || !std::isfinite(dt_macro_) || !std::isfinite(exchange_window_)) {
        throw std::runtime_error("micro dt, macro dt, and exchange_window must be finite");
    }
    if (!is_integer_multiple(dt_macro_, dt_micro_)) {
        throw std::runtime_error("dt_macro must be an integer multiple of dt_micro");
    }
    if (!is_integer_multiple(exchange_window_, dt_macro_)) {
        throw std::runtime_error("exchange_window must be an integer multiple of dt_macro");
    }
    const int min_delay_steps = minimum_active_connectivity_delay_steps(network_, dt_macro_);
    if (min_delay_steps <= 0) {
        throw std::runtime_error("exchange_window requires at least one active connectivity delay");
    }
    if (integer_step_count(exchange_window_, dt_macro_) > min_delay_steps) {
        throw std::runtime_error(
            "exchange_window must not exceed the minimum active connectivity delay");
    }
}

mind_sim::cosim::SimulationResult Simulator::run(double t_stop) {
    if (t_stop < 0.0) {
        throw std::runtime_error("t_stop must be non-negative");
    }
    if (t_stop > 0.0 && !is_integer_multiple(t_stop, dt_macro_)) {
        throw std::runtime_error("t_stop must be an integer multiple of dt_macro");
    }
    const int roi_count = network_.roi_count();
    const int input_count = network_.input_count();
    const int output_count = network_.output_count();
    const int step_count = integer_step_count(t_stop, dt_macro_);

    auto region_owners = network_.region_owners();
    auto field_owners = network_.neural_field_owners();
    auto micro_circuits = network_.micro_circuits();
    const auto roi_owners = collect_roi_owners(region_owners, field_owners, micro_circuits, true);
    validate_single_roi_owner(roi_count,
                              roi_owners,
                              "every ROI must have exactly one owner before run");
    const auto macro_rois = continuous_macro_rois(roi_owners);

    const auto macro_to_macro_runtime = build_macro_to_macro_runtime(network_, dt_macro_);
    const auto region_macro_to_macro_evaluation =
        macro_to_macro_evaluation_for_targets(macro_to_macro_runtime,
                                        macro_rois,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    const auto micro_macro_to_macro_evaluation =
        macro_to_macro_evaluation_for_targets(macro_to_macro_runtime,
                                        roi_owners.detailed_microcircuit_rois,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    std::vector<double> history(
        static_cast<std::size_t>(macro_to_macro_runtime.history_capacity * roi_count * output_count),
        0.0);
    auto current_output_soa =
        output_buffers_to_soa(network_.output_history_start(), roi_count, output_count);
    for (const auto& owner: field_owners) {
        aggregate_field_outputs(owner, current_output_soa);
    }
    const int history_step_offset =
        initialize_history(history,
                           macro_to_macro_runtime.history_capacity,
                           roi_count,
                           output_count,
                           current_output_soa,
                           network_.initial_history(),
                           network_.initial_history_time_count());

    std::vector<double> current_input_soa;
    apply_macro_to_macro(region_macro_to_macro_evaluation,
                    roi_count,
                    input_count,
                    output_count,
                    history_step_offset + 1,
                    history,
                    current_input_soa);

    auto region_groups = build_region_groups(region_owners);

    std::vector<std::unique_ptr<mind_sim::micro::sim::MicroRuntime>> micro_runtimes;
    micro_runtimes.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        micro_runtimes.push_back(
            std::make_unique<mind_sim::micro::sim::MicroRuntime>(*circuit.core_data));
    }
    const auto& micro_record_targets = network_.micro_record_targets();
    const auto& micro_time_record_targets = network_.micro_time_record_targets();
    const int micro_sample_count = integer_step_count(t_stop, dt_micro_);
    const double micro_start_time =
        micro_circuits.empty() || micro_circuits.front().core_data->threads.empty()
            ? 0.0
            : micro_circuits.front().core_data->threads.front()._t;
    for (auto* samples: micro_time_record_targets) {
        if (samples->empty()) {
            samples->push_back(micro_start_time);
        }
        const auto old_size = samples->size();
        samples->resize(old_size + static_cast<std::size_t>(micro_sample_count));
        for (int sample = 0; sample < micro_sample_count; ++sample) {
            (*samples)[old_size + static_cast<std::size_t>(sample)] =
                micro_start_time + (static_cast<double>(sample + 1) * dt_micro_);
        }
    }

    std::vector<std::vector<MicroRecordBinding>> micro_recorders_by_circuit(micro_circuits.size());
    for (int recorder = 0; recorder < static_cast<int>(micro_record_targets.size()); ++recorder) {
        const auto& target = micro_record_targets[static_cast<std::size_t>(recorder)];
        double* const pointer = target.value;
        if (target.samples == nullptr) {
            throw std::runtime_error("hybrid micro recorder has no sample buffer");
        }
        int matched_circuit = -1;
        int matched_thread = -1;
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            const auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            const int thread_index = thread_index_for_pointer(*circuit.core_data, pointer);
            if (thread_index >= 0) {
                matched_circuit = circuit_index;
                matched_thread = thread_index;
                break;
            }
        }
        if (matched_circuit < 0) {
            throw std::runtime_error("micro recorder pointer does not belong to any bound micro circuit");
        }
        if (target.samples->empty()) {
            target.samples->push_back(*pointer);
        }
        const auto sample_offset = target.samples->size();
        target.samples->resize(sample_offset + static_cast<std::size_t>(micro_sample_count));
        micro_recorders_by_circuit[static_cast<std::size_t>(matched_circuit)].push_back(MicroRecordBinding{
            .value = pointer,
            .samples = target.samples,
            .sample_offset = sample_offset,
            .thread_index = matched_thread,
        });
    }
    std::vector<MicroRecordPlan> micro_record_plans;
    micro_record_plans.reserve(micro_circuits.size());
    for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
        micro_record_plans.push_back(
            make_micro_record_plan(
                *micro_circuits[static_cast<std::size_t>(circuit_index)].core_data,
                micro_recorders_by_circuit[static_cast<std::size_t>(circuit_index)]));
    }

    SimulationResult result;
    result.times.resize(static_cast<std::size_t>(step_count) + 1);
    for (int step = 0; step <= step_count; ++step) {
        result.times[static_cast<std::size_t>(step)] = step * dt_macro_;
    }
    result.records.roi_count = roi_count;
    result.records.roi_indices = network_.recorded_rois();
    result.records.output_indices = network_.recorded_outputs();
    result.records.output_count =
        static_cast<int>(result.records.output_indices.size());
    const int output_sample_count =
        step_count == 0 ? 1 : ((step_count + exchange_step_count_ - 1) / exchange_step_count_) + 1;
    result.records.values.reserve(
        static_cast<std::size_t>(output_sample_count) *
        result.records.roi_indices.size() *
        static_cast<std::size_t>(result.records.output_count));
    append_record_table(result.records, current_output_soa, output_count);
    std::vector<std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTable>>> micro_transform_spikes;
    std::vector<std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTableView>>> micro_transform_views;
    std::vector<mind_sim::micro::sim::MicroSpikeTable> micro_pending_spikes(micro_circuits.size());
    micro_transform_spikes.reserve(micro_circuits.size());
    micro_transform_views.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        auto& circuit_spikes = micro_transform_spikes.emplace_back();
        auto& circuit_views = micro_transform_views.emplace_back();
        circuit_spikes.reserve(circuit.bindings.size());
        circuit_views.reserve(circuit.bindings.size());
        for (const auto& binding: circuit.bindings) {
            circuit_spikes.emplace_back(binding.output_transforms.size());
            circuit_views.emplace_back(binding.output_transforms.size());
        }
    }
    std::vector<double> micro_input_trace_soa;
    std::vector<double> micro_exposure_trace_soa;
    std::vector<double> micro_sample_input_soa;
    std::vector<double> next_input_soa;
    std::vector<double> micro_output_soa(current_output_soa.size(), 0.0);
    std::vector<double> micro_output_trace_soa;
    std::vector<std::size_t> micro_output_offsets;
    micro_output_offsets.reserve(
        roi_owners.detailed_microcircuit_rois.size() * static_cast<std::size_t>(output_count));
    for (int roi: roi_owners.detailed_microcircuit_rois) {
        for (int output = 0; output < output_count; ++output) {
            micro_output_offsets.push_back(static_cast<std::size_t>(output * roi_count + roi));
        }
    }
    std::vector<mind_sim::micro::sim::MicroWindowToken> micro_tokens(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> current_micro_events(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> next_micro_events(micro_circuits.size());
    const auto trace_ahead_safety =
        check_micro_input_trace_ahead(micro_macro_to_macro_evaluation, exchange_step_count_);
    const bool source_exposure_trace_ahead_unsafe =
        macro2micro_uses_source_exposure(micro_circuits);
    bool use_micro_pipeline =
        !micro_circuits.empty() && trace_ahead_safety.safe && !source_exposure_trace_ahead_unsafe;
    const bool force_serial_pipeline = env_flag_enabled("MIND_SIM_FORCE_SERIAL_PIPELINE");
    if (use_micro_pipeline && force_serial_pipeline) {
        use_micro_pipeline = false;
        std::cerr
            << "MIND Sim: async micro pipeline disabled by MIND_SIM_FORCE_SERIAL_PIPELINE; "
            << "running serial exchange windows.\n";
    }
    if (!micro_circuits.empty() && !trace_ahead_safety.safe) {
        std::cerr
            << "MIND Sim: async micro pipeline disabled; running serial exchange windows. "
            << "The shortest macro input delay feeding a micro ROI is "
            << trace_ahead_safety.min_delay_steps
            << " dt_macro steps, but trace-ahead scheduling requires at least "
            << trace_ahead_safety.required_delay_steps
            << " steps for the current exchange_window. Offending target ROI index="
            << trace_ahead_safety.target_roi
            << ", source ROI index="
            << trace_ahead_safety.source_roi
            << ".\n";
    }
    if (!micro_circuits.empty() && source_exposure_trace_ahead_unsafe) {
        std::cerr
            << "MIND Sim: async micro pipeline disabled; running serial exchange windows. "
            << "A macro2micro transform reads SOURCE_EXPOSURE, which requires boundary output "
            << "values from the previous exchange window.\n";
    }
    const bool serial_profile_requested = env_flag_enabled("MIND_SIM_SERIAL_PROFILE");
    if (serial_profile_requested && use_micro_pipeline) {
        std::cerr
            << "MIND Sim: serial profiling requested but async micro pipeline is active; "
            << "set MIND_SIM_FORCE_SERIAL_PIPELINE=1 to profile serial exchange windows.\n";
    }
    SerialProfile serial_profile{
        .enabled = serial_profile_requested && !use_micro_pipeline && !micro_circuits.empty(),
    };
    const bool async_profile_requested = env_flag_enabled("MIND_SIM_ASYNC_PROFILE");
    if (async_profile_requested && !use_micro_pipeline) {
        std::cerr
            << "MIND Sim: async profiling requested but async micro pipeline is inactive.\n";
    }
    AsyncProfile async_profile{
        .enabled = async_profile_requested && use_micro_pipeline && !micro_circuits.empty(),
    };
    const auto output_stride = current_output_soa.size();
    const auto prepare_micro_output_trace = [&](int sample_count) {
        micro_output_trace_soa.resize(static_cast<std::size_t>(sample_count) * output_stride);
        for (int sample = 0; sample < sample_count; ++sample) {
            const auto base = static_cast<std::size_t>(sample) * output_stride;
            for (const auto offset: micro_output_offsets) {
                micro_output_trace_soa[base + offset] = 0.0;
            }
        }
        for (const auto offset: micro_output_offsets) {
            micro_output_soa[offset] = 0.0;
        }
    };
    const auto apply_micro_output_trace_sample = [&](int sample_index) {
        const auto base = static_cast<std::size_t>(sample_index) * output_stride;
        for (const auto offset: micro_output_offsets) {
            current_output_soa[offset] = micro_output_trace_soa[base + offset];
        }
    };

    const auto submit_micro_window = [&](double window_start_time,
                                         double window_stop_time,
                                         std::vector<mind_sim::micro::sim::MicroEventTable>& events) {
        measure_if(async_profile.enabled, async_profile.submit_seconds, [&]() {
            const int micro_record_sample_count =
                integer_step_count(window_stop_time - window_start_time, dt_micro_);
            const auto micro_record_sample_offset =
                static_cast<std::size_t>(integer_step_count(window_start_time, dt_micro_));
            for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
                append_events(micro_runtimes[circuit]->scheduled_events(), events[circuit]);
            }
            for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
                auto& plan = micro_record_plans[circuit];
                arm_trace_requests(*micro_circuits[circuit].core_data,
                                   plan,
                                   micro_record_sample_count,
                                   micro_record_sample_offset);
                try {
                    micro_tokens[circuit] =
                        micro_runtimes[circuit]->submit_window(window_start_time, window_stop_time);
                } catch (...) {
                    clear_trace_requests(*micro_circuits[circuit].core_data, plan);
                    throw;
                }
            }
        });
    };

    const auto finish_micro_window = [&](double window_start_time, double window_stop_time) {
        if (async_profile.enabled) {
            ++async_profile.exchange_windows;
        }
        const int sample_count = integer_step_count(window_stop_time - window_start_time, dt_macro_);
        measure_if(async_profile.enabled, async_profile.output_transform_seconds, [&]() {
            prepare_micro_output_trace(sample_count);
        });
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            auto& plan = micro_record_plans[static_cast<std::size_t>(circuit_index)];
            mind_sim::micro::sim::MicroSpikeTableView spikes;
            try {
                measure_if(async_profile.enabled, async_profile.wait_micro_seconds, [&]() {
                    spikes = micro_runtimes[static_cast<std::size_t>(circuit_index)]->finish_window(
                        micro_tokens[static_cast<std::size_t>(circuit_index)]);
                });
            } catch (...) {
                clear_trace_requests(*circuit.core_data, plan);
                throw;
            }
            measure_if(async_profile.enabled, async_profile.output_transform_seconds, [&]() {
                clear_trace_requests(*circuit.core_data, plan);
                validate_recorded_sample_count(
                    plan,
                    integer_step_count(window_stop_time - window_start_time, dt_micro_));
                bind_spike_views(
                    spikes,
                    circuit,
                    micro_pending_spikes[static_cast<std::size_t>(circuit_index)],
                    micro_transform_spikes[static_cast<std::size_t>(circuit_index)],
                    micro_transform_views[static_cast<std::size_t>(circuit_index)],
                    dt_micro_,
                    window_start_time,
                    window_stop_time);
                for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
                    auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
                    auto& transform_views =
                        micro_transform_views[static_cast<std::size_t>(circuit_index)][static_cast<std::size_t>(binding_index)];
                    for (int transform_index = 0;
                         transform_index < static_cast<int>(binding.output_transforms.size());
                         ++transform_index) {
                        auto& transform = binding.output_transforms[static_cast<std::size_t>(transform_index)];
                        const auto spike_view = transform_views[static_cast<std::size_t>(transform_index)];
                        transform.rule->apply(spike_view,
                                              micro_output_soa,
                                              micro_output_trace_soa,
                                              roi_count,
                                              output_count,
                                              binding.roi_index,
                                              transform.state,
                                              transform.params,
                                              window_start_time,
                                              window_stop_time,
                                              sample_count,
                                              dt_macro_,
                                              transform.source_exposure_offsets);
                    }
                }
            });
        }

        measure_if(async_profile.enabled, async_profile.output_transform_seconds, [&]() {
            for (const auto offset: micro_output_offsets) {
                current_output_soa[offset] = micro_output_soa[offset];
            }
        });
    };

    const auto run_micro_window_synchronously = [&](double window_start_time,
                                                   double window_stop_time,
                                                   std::vector<mind_sim::micro::sim::MicroEventTable>& events) {
        if (serial_profile.enabled) {
            ++serial_profile.exchange_windows;
        }
        const int sample_count = integer_step_count(window_stop_time - window_start_time, dt_macro_);
        prepare_micro_output_trace(sample_count);
        const int micro_record_sample_count =
            micro_record_targets.empty()
                ? 0
                : integer_step_count(window_stop_time - window_start_time, dt_micro_);
        const auto micro_record_sample_offset =
            static_cast<std::size_t>(integer_step_count(window_start_time, dt_micro_));
        measure_if(serial_profile.enabled, serial_profile.transform_seconds, [&]() {
            for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
                append_events(micro_runtimes[circuit]->scheduled_events(), events[circuit]);
            }
        });
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            RecordedMicroWindow window;
            measure_if(serial_profile.enabled, serial_profile.micro_seconds, [&]() {
                window = advance_micro_window_recorded(
                    *micro_runtimes[static_cast<std::size_t>(circuit_index)],
                    *circuit.core_data,
                    micro_record_plans[static_cast<std::size_t>(circuit_index)],
                    window_start_time,
                    window_stop_time,
                    micro_record_sample_count,
                    micro_record_sample_offset);
            });
            measure_if(serial_profile.enabled, serial_profile.transform_seconds, [&]() {
                bind_spike_views(
                    window.spikes,
                    circuit,
                    micro_pending_spikes[static_cast<std::size_t>(circuit_index)],
                    micro_transform_spikes[static_cast<std::size_t>(circuit_index)],
                    micro_transform_views[static_cast<std::size_t>(circuit_index)],
                    dt_micro_,
                    window_start_time,
                    window_stop_time);
                for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
                    auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
                    auto& transform_views =
                        micro_transform_views[static_cast<std::size_t>(circuit_index)][static_cast<std::size_t>(binding_index)];
                    for (int transform_index = 0;
                         transform_index < static_cast<int>(binding.output_transforms.size());
                         ++transform_index) {
                        auto& transform = binding.output_transforms[static_cast<std::size_t>(transform_index)];
                        const auto spike_view = transform_views[static_cast<std::size_t>(transform_index)];
                        transform.rule->apply(spike_view,
                                              micro_output_soa,
                                              micro_output_trace_soa,
                                              roi_count,
                                              output_count,
                                              binding.roi_index,
                                              transform.state,
                                              transform.params,
                                              window_start_time,
                                              window_stop_time,
                                              sample_count,
                                              dt_macro_,
                                              transform.source_exposure_offsets);
                    }
                }
            });
        }

        measure_if(serial_profile.enabled, serial_profile.transform_seconds, [&]() {
            for (const auto offset: micro_output_offsets) {
                current_output_soa[offset] = micro_output_soa[offset];
            }
        });
    };

    bool next_micro_window_prepared = false;
    int next_micro_window_start = 0;
    int next_micro_window_stop = 0;
    const auto prepare_async_boundary_input = [&](int boundary_step) {
        measure_if(async_profile.enabled, async_profile.macro_input_seconds, [&]() {
            apply_macro_to_macro(region_macro_to_macro_evaluation,
                            roi_count,
                            input_count,
                            output_count,
                            history_step_offset + boundary_step + 1,
                            history,
                            next_input_soa);
        });
    };
    const auto prepare_async_micro_window = [&](int window_start) {
        if (window_start >= step_count) {
            next_micro_window_prepared = false;
            return;
        }
        const int window_stop = std::min(step_count, window_start + exchange_step_count_);
        measure_if(async_profile.enabled, async_profile.prepare_input_seconds, [&]() {
            prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                           roi_count,
                                           input_count,
                                           output_count,
                                           window_start,
                                           window_stop,
                                           history_step_offset,
                                           dt_macro_,
                                           history,
                                           micro_input_trace_soa,
                                           micro_exposure_trace_soa,
                                           micro_sample_input_soa,
                                           micro_circuits,
                                           next_micro_events,
                                           macro2micro_seed_);
        });
        next_micro_window_start = window_start;
        next_micro_window_stop = window_stop;
        next_micro_window_prepared = true;
    };

    if (use_micro_pipeline && step_count > 0) {
        const int first_exchange_stop = std::min(step_count, exchange_step_count_);
        measure_if(async_profile.enabled, async_profile.prepare_input_seconds, [&]() {
            prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                           roi_count,
                                           input_count,
                                           output_count,
                                           0,
                                           first_exchange_stop,
                                           history_step_offset,
                                           dt_macro_,
                                           history,
                                           micro_input_trace_soa,
                                           micro_exposure_trace_soa,
                                           micro_sample_input_soa,
                                           micro_circuits,
                                           current_micro_events,
                                           macro2micro_seed_);
        });
        submit_micro_window(0.0, first_exchange_stop * dt_macro_, current_micro_events);
        if (first_exchange_stop < step_count) {
            prepare_async_boundary_input(first_exchange_stop);
            prepare_async_micro_window(first_exchange_stop);
        }
    }

    for (int exchange_start = 0; exchange_start < step_count; exchange_start += exchange_step_count_) {
        const int exchange_stop = std::min(step_count, exchange_start + exchange_step_count_);
        const double exchange_start_time = exchange_start * dt_macro_;
        const double exchange_stop_time = exchange_stop * dt_macro_;

        if (!use_micro_pipeline) {
            measure_if(serial_profile.enabled, serial_profile.transform_seconds, [&]() {
                prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                               roi_count,
                                               input_count,
                                               output_count,
                                               exchange_start,
                                               exchange_stop,
                                               history_step_offset,
                                               dt_macro_,
                                               history,
                                               micro_input_trace_soa,
                                               micro_exposure_trace_soa,
                                               micro_sample_input_soa,
                                               micro_circuits,
                                               current_micro_events,
                                               macro2micro_seed_);
            });
            run_micro_window_synchronously(exchange_start_time,
                                           exchange_stop_time,
                                           current_micro_events);
        } else {
            finish_micro_window(exchange_start_time, exchange_stop_time);
            if (exchange_stop < step_count) {
                if (!next_micro_window_prepared ||
                    next_micro_window_start != exchange_stop) {
                    throw std::runtime_error("async micro pipeline has no prepared next window");
                }
                submit_micro_window(next_micro_window_start * dt_macro_,
                                    next_micro_window_stop * dt_macro_,
                                    next_micro_events);
                next_micro_window_prepared = false;
            }
        }

        for (int step = exchange_start; step < exchange_stop; ++step) {
            const double start_time = step * dt_macro_;
            const double stop_time = (step + 1) * dt_macro_;
            if (!micro_output_trace_soa.empty()) {
                apply_micro_output_trace_sample(step - exchange_start);
            }

            const auto step_macro_models = [&]() {
                for (auto& owner: field_owners) {
                    step_neural_field(owner,
                                      roi_count,
                                      current_input_soa,
                                      current_output_soa,
                                      start_time,
                                      stop_time - start_time);
                }
                for (auto& group: region_groups) {
                    group.rule->step_group(group.roi_indices,
                                           roi_count,
                                           current_input_soa,
                                           current_output_soa,
                                           group.state_soa,
                                           group.params_soa,
                                           group.target_input_offsets,
                                           group.source_exposure_offsets,
                                           start_time,
                                           stop_time - start_time);
                }
            };
            if (serial_profile.enabled) {
                measure_if(true, serial_profile.macro_seconds, step_macro_models);
            } else if (async_profile.enabled) {
                measure_if(true, async_profile.macro_seconds, step_macro_models);
            } else {
                step_macro_models();
            }
            if (serial_profile.enabled) {
                ++serial_profile.macro_steps;
            }
            if (async_profile.enabled) {
                ++async_profile.macro_steps;
            }

            if (step + 1 == exchange_stop) {
                continue;
            }

            write_history_slot(history,
                               (history_step_offset + step + 1) %
                                   macro_to_macro_runtime.history_capacity,
                               roi_count,
                               output_count,
                               current_output_soa);
            append_record_table(result.records, current_output_soa, output_count);

            apply_macro_to_macro(region_macro_to_macro_evaluation,
                            roi_count,
                            input_count,
                            output_count,
                            history_step_offset + step + 2,
                            history,
                            current_input_soa);
        }

        write_history_slot(history,
                           (history_step_offset + exchange_stop) %
                               macro_to_macro_runtime.history_capacity,
                           roi_count,
                           output_count,
                           current_output_soa);
        append_record_table(result.records, current_output_soa, output_count);

        if (use_micro_pipeline && exchange_stop < step_count) {
            current_input_soa.swap(next_input_soa);
            const int following_start = exchange_stop + exchange_step_count_;
            if (following_start < step_count) {
                prepare_async_boundary_input(following_start);
                prepare_async_micro_window(following_start);
            }
        }
        if (!use_micro_pipeline) {
            apply_macro_to_macro(region_macro_to_macro_evaluation,
                            roi_count,
                            input_count,
                            output_count,
                            history_step_offset + exchange_stop + 1,
                            history,
                            current_input_soa);
        }
    }

    report_serial_profile(serial_profile);
    report_async_profile(async_profile);
    return result;
}

}  // namespace mind_sim::cosim
