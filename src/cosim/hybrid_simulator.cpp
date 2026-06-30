#include "cosim/hybrid_simulator.hpp"

#include "macro/sim/runtime_core.hpp"
#include "micro/sim/micro_runtime.hpp"
#include "micro/sim/recording.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
        target.sort_by_time_index();
        return;
    }
    target.time.insert(target.time.end(), source.time.begin(), source.time.end());
    target.index.insert(target.index.end(), source.index.begin(), source.index.end());
    source.clear();
    target.sort_by_time_index();
}

void prepare_micro_events_for_exchange(
    const mind_sim::macro::sim::MacroToMacroEvaluation& micro_macro_to_macro_evaluation,
    int roi_count,
    int exposure_count,
    int exchange_start,
    int exchange_stop,
    int history_step_offset,
    double dt_macro,
    const std::vector<double>& history,
    std::vector<double>& sample_exposure_soa,
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
    const auto exposure_stride = static_cast<std::size_t>(exposure_count * roi_count);
    if (sample_exposure_soa.size() != exposure_stride) {
        sample_exposure_soa.assign(exposure_stride, 0.0);
    }

    struct BindingTrace {
        int circuit_index{-1};
        int binding_index{-1};
        int input_index{-1};
        int exposure_count{0};
        std::vector<int> exposure_offsets{};
        std::vector<double> trace_soa{};
    };
    std::vector<BindingTrace> binding_traces;
    for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
        auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
        for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
            auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
            for (int input_index = 0;
                 input_index < static_cast<int>(binding.input_transforms.size());
                 ++input_index) {
                auto& input =
                    binding.input_transforms[static_cast<std::size_t>(input_index)];
                if (!input.rule) {
                    continue;
                }
                const int rule_exposure_count = input.rule->exposure_count();
                BindingTrace trace{
                    .circuit_index = circuit_index,
                    .binding_index = binding_index,
                    .input_index = input_index,
                    .exposure_count = rule_exposure_count,
                };
                trace.exposure_offsets.reserve(static_cast<std::size_t>(rule_exposure_count));
                for (int exposure = 0; exposure < rule_exposure_count; ++exposure) {
                    trace.exposure_offsets.push_back(exposure * roi_count);
                }
                trace.trace_soa.assign(
                    static_cast<std::size_t>(sample_count) *
                        static_cast<std::size_t>(rule_exposure_count) *
                        static_cast<std::size_t>(roi_count),
                    0.0);
                binding_traces.push_back(std::move(trace));
            }
        }
    }
    if (binding_traces.empty()) {
        return;
    }

    for (int sample = 0; sample < sample_count; ++sample) {
        const auto history_base =
            static_cast<std::size_t>((history_step_offset + exchange_start + sample) %
                                     micro_macro_to_macro_evaluation.history_capacity) *
            exposure_stride;
        for (const auto& trace: binding_traces) {
            const auto& binding =
                micro_circuits[static_cast<std::size_t>(trace.circuit_index)]
                    .bindings[static_cast<std::size_t>(trace.binding_index)];
            const int roi = binding.roi_index;
            for (int exposure = 0; exposure < exposure_count; ++exposure) {
                const auto offset = static_cast<std::size_t>(exposure * roi_count + roi);
                sample_exposure_soa[offset] = history[history_base + offset];
            }
        }
        apply_macro_to_macro(micro_macro_to_macro_evaluation,
                             roi_count,
                             exposure_count,
                             history_step_offset + exchange_start + sample + 1,
                             history,
                             sample_exposure_soa);
        for (auto& trace: binding_traces) {
            const auto& binding =
                micro_circuits[static_cast<std::size_t>(trace.circuit_index)]
                    .bindings[static_cast<std::size_t>(trace.binding_index)];
            const auto& input =
                binding.input_transforms[static_cast<std::size_t>(trace.input_index)];
            const int roi = binding.roi_index;
            const auto sample_base =
                static_cast<std::size_t>(sample) *
                static_cast<std::size_t>(trace.exposure_count) *
                static_cast<std::size_t>(roi_count);
            for (int exposure = 0; exposure < trace.exposure_count; ++exposure) {
                trace.trace_soa[sample_base +
                                static_cast<std::size_t>(exposure * roi_count + roi)] =
                    sample_exposure_soa[static_cast<std::size_t>(
                        input.exposure_offsets[static_cast<std::size_t>(exposure)] + roi)];
            }
        }
    }
    const double exchange_start_time = exchange_start * dt_macro;
    const double exchange_stop_time = exchange_stop * dt_macro;
    for (auto& trace: binding_traces) {
        auto& circuit = micro_circuits[static_cast<std::size_t>(trace.circuit_index)];
        auto& binding = circuit.bindings[static_cast<std::size_t>(trace.binding_index)];
        auto& input = binding.input_transforms[static_cast<std::size_t>(trace.input_index)];
        auto& events = prepared_events[static_cast<std::size_t>(trace.circuit_index)];
        input.rule->apply(trace.trace_soa,
                          sample_count,
                          dt_macro,
                          trace.exposure_count,
                          roi_count,
                          binding.roi_index,
                          input.state,
                          input.params,
                          exchange_start_time,
                          exchange_stop_time,
                          macro2micro_seed,
                          input.rng_stream_id,
                          exchange_start,
                          input.macro2micro_indices,
                          input.macro2micro_source_ids,
                          events,
                          trace.exposure_offsets);
    }
}

bool source_accepts_sid(const std::vector<int>& source_sids, int sid) {
    return source_sids.empty() ||
           std::binary_search(source_sids.begin(), source_sids.end(), sid);
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
using mind_sim::macro::sim::build_macro_to_macro_runtime;
using mind_sim::macro::sim::build_region_groups;
using mind_sim::macro::sim::collect_roi_owners;
using mind_sim::macro::sim::macro_to_macro_evaluation_for_targets;
using mind_sim::macro::sim::continuous_macro_rois;
using mind_sim::macro::sim::output_buffers_to_soa;
using mind_sim::macro::sim::initialize_history;
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
    const int exposure_count = network_.output_count();
    const int step_count = integer_step_count(t_stop, dt_macro_);

    auto region_owners = network_.region_owners();
    auto micro_circuits = network_.micro_circuits();
    const auto roi_owners = collect_roi_owners(region_owners, micro_circuits, true);
    std::vector<unsigned char> region_seen(static_cast<std::size_t>(roi_count), 0);
    std::vector<unsigned char> micro_seen(static_cast<std::size_t>(roi_count), 0);
    for (int roi: roi_owners.neural_mass_rois) {
        ++region_seen[static_cast<std::size_t>(roi)];
    }
    for (int roi: roi_owners.detailed_microcircuit_rois) {
        ++micro_seen[static_cast<std::size_t>(roi)];
    }
    for (int roi = 0; roi < roi_count; ++roi) {
        const auto index = static_cast<std::size_t>(roi);
        if (region_seen[index] > 1 || micro_seen[index] > 1) {
            throw std::runtime_error("every ROI can have at most one owner of each kind before hybrid run");
        }
        if (region_seen[index] == 0 && micro_seen[index] == 0) {
            throw std::runtime_error("every ROI must have a macro or micro owner before hybrid run");
        }
    }
    const auto macro_rois = continuous_macro_rois(roi_owners);

    const auto macro_to_macro_runtime = build_macro_to_macro_runtime(network_, dt_macro_);
    const auto region_macro_to_macro_evaluation =
        macro_to_macro_evaluation_for_targets(macro_to_macro_runtime,
                                        macro_rois,
                                        roi_count);
    const auto micro_macro_to_macro_evaluation =
        macro_to_macro_evaluation_for_targets(macro_to_macro_runtime,
                                        roi_owners.detailed_microcircuit_rois,
                                        roi_count);
    std::vector<double> history(
        static_cast<std::size_t>(macro_to_macro_runtime.history_capacity * roi_count * exposure_count),
        0.0);
    auto current_exposure_soa =
        output_buffers_to_soa(network_.output_history_start(), roi_count, exposure_count);
    const int history_step_offset =
        initialize_history(history,
                           macro_to_macro_runtime.history_capacity,
                           roi_count,
                           exposure_count,
                           current_exposure_soa,
                           network_.initial_history(),
                           network_.initial_history_time_count());

    apply_macro_to_macro(region_macro_to_macro_evaluation,
                    roi_count,
                    exposure_count,
                    history_step_offset + 1,
                    history,
                    current_exposure_soa);

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

    std::vector<std::unique_ptr<mind_sim::micro::sim::CoreRecordingPlan>> micro_record_plans;
    micro_record_plans.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        micro_record_plans.push_back(
            std::make_unique<mind_sim::micro::sim::CoreRecordingPlan>(*circuit.core_data));
    }
    for (int recorder = 0; recorder < static_cast<int>(micro_record_targets.size()); ++recorder) {
        const auto& target = micro_record_targets[static_cast<std::size_t>(recorder)];
        double* const pointer = target.value;
        if (target.samples == nullptr) {
            throw std::runtime_error("hybrid micro recorder has no sample buffer");
        }
        int matched_circuit = -1;
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            const auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            try {
                (void) mind_sim::micro::sim::thread_index_for_data_pointer(*circuit.core_data, pointer);
                matched_circuit = circuit_index;
                break;
            } catch (const std::runtime_error&) {
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
        micro_record_plans[static_cast<std::size_t>(matched_circuit)]->add_target(
            pointer,
            target.samples->data() + sample_offset);
    }
    for (auto& plan: micro_record_plans) {
        plan->prepare(micro_sample_count);
        plan->activate();
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
    append_record_table(result.records, current_exposure_soa, exposure_count);
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
    std::vector<double> micro_sample_exposure_soa;
    std::vector<double> micro_output_soa(current_exposure_soa.size(), 0.0);
    std::vector<double> micro_output_trace_soa;
    std::vector<std::size_t> micro_output_offsets;
    const auto output_stride = current_exposure_soa.size();
    std::vector<unsigned char> micro_output_selected(output_stride, 0);
    for (const auto& circuit: micro_circuits) {
        for (const auto& binding: circuit.bindings) {
            for (const auto& transform: binding.output_transforms) {
                for (int offset: transform.exposure_offsets) {
                    const auto owned_offset =
                        static_cast<std::size_t>(offset + binding.roi_index);
                    if (micro_output_selected[owned_offset] == 0) {
                        micro_output_selected[owned_offset] = 1;
                        micro_output_offsets.push_back(owned_offset);
                    }
                }
            }
        }
    }
    std::vector<mind_sim::micro::sim::MicroWindowToken> micro_tokens(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> current_micro_events(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> next_micro_events(micro_circuits.size());
    const auto trace_ahead_safety =
        check_micro_input_trace_ahead(micro_macro_to_macro_evaluation, exchange_step_count_);
    bool use_micro_pipeline =
        !micro_circuits.empty() && trace_ahead_safety.safe;
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
            current_exposure_soa[offset] = micro_output_trace_soa[base + offset];
        }
    };

    const auto submit_micro_window = [&](double window_start_time,
                                         double window_stop_time,
                                         std::vector<mind_sim::micro::sim::MicroEventTable>& events) {
        for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
            append_events(micro_runtimes[circuit]->scheduled_events(), events[circuit]);
        }
        for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
            micro_tokens[circuit] =
                micro_runtimes[circuit]->submit_window(window_start_time, window_stop_time);
        }
    };

    const auto finish_micro_window = [&](double window_start_time, double window_stop_time) {
        const int sample_count = integer_step_count(window_stop_time - window_start_time, dt_macro_);
        prepare_micro_output_trace(sample_count);
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            auto spikes = micro_runtimes[static_cast<std::size_t>(circuit_index)]->finish_window(
                micro_tokens[static_cast<std::size_t>(circuit_index)]);
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
                                          exposure_count,
                                          binding.roi_index,
                                          transform.state,
                                          transform.params,
                                          window_start_time,
                                          window_stop_time,
                                          sample_count,
                                          dt_macro_,
                                          transform.exposure_offsets);
                }
            }
        }

        for (const auto offset: micro_output_offsets) {
            current_exposure_soa[offset] = micro_output_soa[offset];
        }
    };

    const auto run_micro_window_synchronously = [&](double window_start_time,
                                                   double window_stop_time,
                                                   std::vector<mind_sim::micro::sim::MicroEventTable>& events) {
        const int sample_count = integer_step_count(window_stop_time - window_start_time, dt_macro_);
        prepare_micro_output_trace(sample_count);
        for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
            append_events(micro_runtimes[circuit]->scheduled_events(), events[circuit]);
        }
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            auto spikes = micro_runtimes[static_cast<std::size_t>(circuit_index)]->advance_window(
                window_start_time,
                window_stop_time);
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
                                          exposure_count,
                                          binding.roi_index,
                                          transform.state,
                                          transform.params,
                                          window_start_time,
                                          window_stop_time,
                                          sample_count,
                                          dt_macro_,
                                          transform.exposure_offsets);
                }
            }
        }

        for (const auto offset: micro_output_offsets) {
            current_exposure_soa[offset] = micro_output_soa[offset];
        }
    };

    bool next_micro_window_prepared = false;
    int next_micro_window_start = 0;
    int next_micro_window_stop = 0;
    const auto prepare_async_micro_window = [&](int window_start) {
        if (window_start >= step_count) {
            next_micro_window_prepared = false;
            return;
        }
        const int window_stop = std::min(step_count, window_start + exchange_step_count_);
        prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                          roi_count,
                                          exposure_count,
                                          window_start,
                                          window_stop,
                                          history_step_offset,
                                          dt_macro_,
                                          history,
                                          micro_sample_exposure_soa,
                                          micro_circuits,
                                          next_micro_events,
                                          macro2micro_seed_);
        next_micro_window_start = window_start;
        next_micro_window_stop = window_stop;
        next_micro_window_prepared = true;
    };

    if (use_micro_pipeline && step_count > 0) {
        const int first_exchange_stop = std::min(step_count, exchange_step_count_);
        prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                          roi_count,
                                          exposure_count,
                                          0,
                                          first_exchange_stop,
                                          history_step_offset,
                                          dt_macro_,
                                          history,
                                          micro_sample_exposure_soa,
                                          micro_circuits,
                                          current_micro_events,
                                          macro2micro_seed_);
        submit_micro_window(0.0, first_exchange_stop * dt_macro_, current_micro_events);
        if (first_exchange_stop < step_count) {
            prepare_async_micro_window(first_exchange_stop);
        }
    }

    for (int exchange_start = 0; exchange_start < step_count; exchange_start += exchange_step_count_) {
        const int exchange_stop = std::min(step_count, exchange_start + exchange_step_count_);
        const double exchange_start_time = exchange_start * dt_macro_;
        const double exchange_stop_time = exchange_stop * dt_macro_;

        if (!use_micro_pipeline) {
            prepare_micro_events_for_exchange(micro_macro_to_macro_evaluation,
                                              roi_count,
                                              exposure_count,
                                              exchange_start,
                                              exchange_stop,
                                              history_step_offset,
                                              dt_macro_,
                                              history,
                                              micro_sample_exposure_soa,
                                              micro_circuits,
                                              current_micro_events,
                                              macro2micro_seed_);
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
            const int exchange_sample = step - exchange_start;
            if (!micro_output_trace_soa.empty()) {
                apply_micro_output_trace_sample(exchange_sample);
            }

            const auto step_macro_models = [&]() {
                for (auto& group: region_groups) {
                    group.rule->step_group(group.roi_indices,
                                           roi_count,
                                           current_exposure_soa,
                                           group.state_soa,
                                           group.params_soa,
                                           group.exposure_offsets,
                                           start_time,
                                           stop_time - start_time);
                }
            };
            step_macro_models();
            if (!micro_output_trace_soa.empty()) {
                apply_micro_output_trace_sample(exchange_sample);
            }

            if (step + 1 == exchange_stop) {
                continue;
            }

            write_history_slot(history,
                               (history_step_offset + step + 1) %
                                   macro_to_macro_runtime.history_capacity,
                               roi_count,
                               exposure_count,
                               current_exposure_soa);
            append_record_table(result.records, current_exposure_soa, exposure_count);

            apply_macro_to_macro(region_macro_to_macro_evaluation,
                            roi_count,
                            exposure_count,
                            history_step_offset + step + 2,
                            history,
                            current_exposure_soa);
        }

        write_history_slot(history,
                           (history_step_offset + exchange_stop) %
                               macro_to_macro_runtime.history_capacity,
                           roi_count,
                           exposure_count,
                           current_exposure_soa);
        append_record_table(result.records, current_exposure_soa, exposure_count);

        if (exchange_stop < step_count) {
            apply_macro_to_macro(region_macro_to_macro_evaluation,
                            roi_count,
                            exposure_count,
                            history_step_offset + exchange_stop + 1,
                            history,
                            current_exposure_soa);
        }
        if (use_micro_pipeline && exchange_stop < step_count) {
            const int following_start = exchange_stop + exchange_step_count_;
            if (following_start < step_count) {
                prepare_async_micro_window(following_start);
            }
        }
    }

    for (auto& plan: micro_record_plans) {
        if (!plan->active()) {
            continue;
        }
        plan->update_host();
        plan->validate_recorded_sample_count(micro_sample_count);
        plan->deactivate();
    }

    return result;
}

}  // namespace mind_sim::cosim
