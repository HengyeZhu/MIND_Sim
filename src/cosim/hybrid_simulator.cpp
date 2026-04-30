#include "cosim/hybrid_simulator.hpp"

#include "macro/sim/runtime_core.hpp"
#include "micro/sim/micro_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mind_sim::cosim {

namespace {

bool is_integer_multiple(double value, double unit) {
    const double ratio = value / unit;
    return std::abs(ratio - std::round(ratio)) <= 1e-9;
}

int integer_step_count(double value, double unit) {
    return static_cast<int>(std::llround(value / unit));
}

bool has_single_contiguous_binding(const mind_sim::macro::frontend::MicroCircuitOwner& circuit) {
    if (circuit.bindings.size() != 1) {
        return false;
    }
    for (int binding_index: circuit.gid_to_binding) {
        if (binding_index != 0) {
            return false;
        }
    }
    return true;
}

bool view_matches_entire_binding(const mind_sim::micro::sim::MicroSpikeTableView& spikes,
                                 const mind_sim::macro::frontend::MicroCircuitOwner& circuit,
                                 double window_start,
                                 double window_stop) {
    if (!has_single_contiguous_binding(circuit)) {
        return false;
    }
    for (std::size_t spike = 0; spike < spikes.size(); ++spike) {
        if (spikes.time[spike] < window_start || spikes.time[spike] >= window_stop) {
            return false;
        }
        const int gid = spikes.gid[spike];
        if (gid < circuit.gid_begin || gid >= circuit.gid_end) {
            return false;
        }
    }
    return true;
}

bool can_prepare_inputs_ahead(const mind_sim::macro::sim::CouplingRuntime& runtime,
                              int batch_step_count) {
    for (const auto& graph: runtime.graphs) {
        for (int delay_steps: graph.edge_delay_steps) {
            if (delay_steps < batch_step_count) {
                return false;
            }
        }
    }
    return true;
}

void clear_event_tables(std::vector<mind_sim::micro::sim::MicroEventTable>& tables) {
    for (auto& table: tables) {
        table.clear();
    }
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

void append_prepared_events_to_runtimes(
    std::vector<std::unique_ptr<mind_sim::micro::sim::MicroRuntime>>& micro_runtimes,
    std::vector<mind_sim::micro::sim::MicroEventTable>& prepared_events) {
    for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
        append_events(micro_runtimes[circuit]->scheduled_events(), prepared_events[circuit]);
    }
}

void prepare_micro_events_for_batch(
    const mind_sim::macro::sim::CouplingEvaluation& micro_coupling_evaluation,
    int roi_count,
    int input_count,
    int batch_start,
    int batch_stop,
    double dt_macro,
    const std::vector<float>& history,
    std::vector<float>& micro_input_soa,
    std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    std::vector<mind_sim::micro::sim::MicroEventTable>& prepared_events) {
    clear_event_tables(prepared_events);
    if (micro_circuits.empty()) {
        return;
    }
    apply_couplings(micro_coupling_evaluation,
                    roi_count,
                    input_count,
                    batch_start,
                    history,
                    micro_input_soa);
    const double batch_start_time = batch_start * dt_macro;
    const double batch_stop_time = batch_stop * dt_macro;
    for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
        auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
        auto& events = prepared_events[static_cast<std::size_t>(circuit_index)];
        for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
            auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
            binding.input_rule->apply(micro_input_soa,
                                      roi_count,
                                      binding.roi_index,
                                      binding.input_state,
                                      binding.input_params,
                                      batch_start_time,
                                      batch_stop_time,
                                      binding.input_port_bases,
                                      events);
        }
    }
}

void bind_spike_views(
    const mind_sim::micro::sim::MicroSpikeTableView& spikes,
    const mind_sim::macro::frontend::MicroCircuitOwner& circuit,
    std::vector<mind_sim::micro::sim::MicroSpikeTable>& binding_spikes,
    std::vector<mind_sim::micro::sim::MicroSpikeTableView>& binding_views,
    double window_start,
    double window_stop) {
    const bool single_binding = has_single_contiguous_binding(circuit);
    for (auto& table: binding_spikes) {
        table.clear();
    }
    for (auto& view: binding_views) {
        view = {};
    }
    if (single_binding && view_matches_entire_binding(spikes, circuit, window_start, window_stop)) {
        binding_views[0] = spikes;
        return;
    }
    for (std::size_t spike = 0; spike < spikes.size(); ++spike) {
        if (spikes.time[spike] < window_start || spikes.time[spike] >= window_stop) {
            continue;
        }
        const int gid = spikes.gid[spike];
        if (gid < circuit.gid_begin || gid >= circuit.gid_end) {
            continue;
        }
        const int binding_index =
            single_binding
                ? 0
                : circuit.gid_to_binding[static_cast<std::size_t>(gid - circuit.gid_begin)];
        if (binding_index < 0) {
            continue;
        }
        auto& table = binding_spikes[static_cast<std::size_t>(binding_index)];
        table.append(spikes.time[spike], gid);
    }
    for (std::size_t binding = 0; binding < binding_spikes.size(); ++binding) {
        binding_views[binding] = binding_spikes[binding].view(0, binding_spikes[binding].size());
    }
}

}  // namespace

using mind_sim::macro::sim::append_exposure_record;
using mind_sim::macro::sim::apply_couplings;
using mind_sim::macro::sim::build_coupling_runtime;
using mind_sim::macro::sim::build_region_groups;
using mind_sim::macro::sim::coupling_evaluation_for_targets;
using mind_sim::macro::sim::exposure_buffers_to_soa;
using mind_sim::macro::sim::initialize_history;
using mind_sim::macro::sim::write_history_slot;

Simulator::Simulator(mind_sim::macro::frontend::Network network,
                     double dt_micro,
                     double dt_macro,
                     double batch_window,
                     bool record_micro_spikes)
    : network_(std::move(network)),
      dt_micro_(dt_micro),
      dt_macro_(dt_macro),
      batch_window_(batch_window),
      record_micro_spikes_(record_micro_spikes) {
    if (dt_micro_ <= 0.0 || dt_macro_ <= 0.0 || batch_window_ <= 0.0) {
        throw std::runtime_error("dt_micro, dt_macro, and batch_window must be positive");
    }
    if (!is_integer_multiple(dt_macro_, dt_micro_)) {
        throw std::runtime_error("dt_macro must be an integer multiple of dt_micro");
    }
    if (!is_integer_multiple(batch_window_, dt_macro_)) {
        throw std::runtime_error("batch_window must be an integer multiple of dt_macro");
    }
    const double min_delay = network_.min_positive_delay();
    if (min_delay <= 0.0) {
        throw std::runtime_error("batch_window requires at least one positive connectivity delay");
    }
    if (batch_window_ - min_delay > 1e-9) {
        throw std::runtime_error(
            "batch_window must not exceed the minimum positive connectivity delay");
    }
    batch_step_count_ = integer_step_count(batch_window_, dt_macro_);
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
    const int exposure_count = network_.exposure_count();
    const int step_count = integer_step_count(t_stop, dt_macro_);

    std::vector<int> region_for_roi(static_cast<std::size_t>(roi_count), -1);
    std::vector<int> region_roi_indices;
    auto region_owners = network_.region_owners();
    for (int owner_index = 0; owner_index < static_cast<int>(region_owners.size()); ++owner_index) {
        const int roi = region_owners[static_cast<std::size_t>(owner_index)].roi_index;
        region_for_roi[static_cast<std::size_t>(roi)] = owner_index;
        region_roi_indices.push_back(roi);
    }

    std::vector<int> micro_for_roi(static_cast<std::size_t>(roi_count), -1);
    std::vector<int> micro_roi_indices;
    auto micro_circuits = network_.micro_circuits();
    for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
        const auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
        for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
            const int roi = circuit.bindings[static_cast<std::size_t>(binding_index)].roi_index;
            micro_for_roi[static_cast<std::size_t>(roi)] = circuit_index;
            micro_roi_indices.push_back(roi);
        }
    }

    for (int roi = 0; roi < roi_count; ++roi) {
        const bool has_region = region_for_roi[static_cast<std::size_t>(roi)] >= 0;
        const bool has_micro = micro_for_roi[static_cast<std::size_t>(roi)] >= 0;
        if (has_region == has_micro) {
            throw std::runtime_error("every ROI must have exactly one owner before run");
        }
    }

    const auto coupling_runtime = build_coupling_runtime(network_, dt_macro_);
    const auto region_coupling_evaluation =
        coupling_evaluation_for_targets(coupling_runtime,
                                        region_roi_indices,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    const auto micro_coupling_evaluation =
        coupling_evaluation_for_targets(coupling_runtime,
                                        micro_roi_indices,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    std::vector<float> history(
        static_cast<std::size_t>(coupling_runtime.history_capacity * roi_count * exposure_count),
        0.0F);
    auto current_exposure_soa =
        exposure_buffers_to_soa(network_.initial_exposures(), roi_count, exposure_count);
    initialize_history(history,
                       coupling_runtime.history_capacity,
                       roi_count,
                       exposure_count,
                       current_exposure_soa);

    std::vector<float> current_input_soa;
    apply_couplings(region_coupling_evaluation, roi_count, input_count, 0, history, current_input_soa);

    auto region_groups = build_region_groups(region_owners);

    std::vector<std::unique_ptr<mind_sim::micro::sim::MicroRuntime>> micro_runtimes;
    micro_runtimes.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        micro_runtimes.push_back(
            std::make_unique<mind_sim::micro::sim::MicroRuntime>(*circuit.core_data));
    }

    SimulationResult result;
    result.times.resize(static_cast<std::size_t>(step_count) + 1);
    for (int step = 0; step <= step_count; ++step) {
        result.times[static_cast<std::size_t>(step)] = step * dt_macro_;
    }
    result.exposures.roi_count = roi_count;
    result.exposures.exposure_count = exposure_count;
    result.exposures.roi_indices = network_.recorded_rois();
    result.exposures.values.reserve(
        (static_cast<std::size_t>(step_count) + 1) *
        result.exposures.roi_indices.size() *
        static_cast<std::size_t>(exposure_count));
    append_exposure_record(result.exposures, current_exposure_soa);
    result.micro_spikes_by_roi.resize(static_cast<std::size_t>(roi_count));

    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTable>> micro_binding_spikes;
    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTableView>> micro_binding_views;
    micro_binding_spikes.reserve(micro_circuits.size());
    micro_binding_views.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        const auto binding_count = circuit.bindings.size();
        micro_binding_spikes.emplace_back(binding_count);
        micro_binding_views.emplace_back(binding_count);
    }
    std::vector<float> micro_input_soa;
    std::vector<float> next_input_soa;
    std::vector<float> micro_exposure_soa(current_exposure_soa.size(), 0.0F);
    std::vector<std::size_t> micro_exposure_offsets;
    micro_exposure_offsets.reserve(
        micro_roi_indices.size() * static_cast<std::size_t>(exposure_count));
    for (int roi: micro_roi_indices) {
        for (int exposure = 0; exposure < exposure_count; ++exposure) {
            micro_exposure_offsets.push_back(static_cast<std::size_t>(exposure * roi_count + roi));
        }
    }
    std::vector<mind_sim::micro::sim::MicroWindowToken> micro_tokens(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> current_micro_events(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> next_micro_events(micro_circuits.size());
    const bool use_input_lookahead =
        !micro_circuits.empty() && can_prepare_inputs_ahead(coupling_runtime, batch_step_count_);

    if (use_input_lookahead && step_count > 0) {
        const int first_batch_stop = std::min(step_count, batch_step_count_);
        prepare_micro_events_for_batch(micro_coupling_evaluation,
                                       roi_count,
                                       input_count,
                                       0,
                                       first_batch_stop,
                                       dt_macro_,
                                       history,
                                       micro_input_soa,
                                       micro_circuits,
                                       current_micro_events);
    }

    for (int batch_start = 0; batch_start < step_count; batch_start += batch_step_count_) {
        const int batch_stop = std::min(step_count, batch_start + batch_step_count_);
        const double batch_start_time = batch_start * dt_macro_;
        const double batch_stop_time = batch_stop * dt_macro_;

        if (!use_input_lookahead) {
            prepare_micro_events_for_batch(micro_coupling_evaluation,
                                           roi_count,
                                           input_count,
                                           batch_start,
                                           batch_stop,
                                           dt_macro_,
                                           history,
                                           micro_input_soa,
                                           micro_circuits,
                                           current_micro_events);
        }
        append_prepared_events_to_runtimes(micro_runtimes, current_micro_events);

        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            micro_tokens[static_cast<std::size_t>(circuit_index)] =
                micro_runtimes[static_cast<std::size_t>(circuit_index)]->submit_window(
                    batch_start_time,
                    batch_stop_time);
        }

        for (int step = batch_start; step < batch_stop; ++step) {
            const double start_time = step * dt_macro_;
            const double stop_time = (step + 1) * dt_macro_;

            for (auto& group: region_groups) {
                group.rule->step_group(group.roi_indices,
                                       roi_count,
                                       current_input_soa,
                                       current_exposure_soa,
                                       group.state_soa,
                                       group.params_soa,
                                       start_time,
                                       stop_time - start_time);
            }

            if (step + 1 == batch_stop) {
                continue;
            }

            write_history_slot(history,
                               (step + 1) % coupling_runtime.history_capacity,
                               roi_count,
                               exposure_count,
                               current_exposure_soa);
            append_exposure_record(result.exposures, current_exposure_soa);

            apply_couplings(region_coupling_evaluation,
                            roi_count,
                            input_count,
                            step + 1,
                            history,
                            current_input_soa);
        }

        if (use_input_lookahead && batch_stop < step_count) {
            const int next_batch_stop = std::min(step_count, batch_stop + batch_step_count_);
            apply_couplings(region_coupling_evaluation,
                            roi_count,
                            input_count,
                            batch_stop,
                            history,
                            next_input_soa);
            prepare_micro_events_for_batch(micro_coupling_evaluation,
                                           roi_count,
                                           input_count,
                                           batch_stop,
                                           next_batch_stop,
                                           dt_macro_,
                                           history,
                                           micro_input_soa,
                                           micro_circuits,
                                           next_micro_events);
        }

        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            auto& token = micro_tokens[static_cast<std::size_t>(circuit_index)];
            const auto spikes =
                micro_runtimes[static_cast<std::size_t>(circuit_index)]->finish_window(token);
            bind_spike_views(
                spikes,
                circuit,
                micro_binding_spikes[static_cast<std::size_t>(circuit_index)],
                micro_binding_views[static_cast<std::size_t>(circuit_index)],
                batch_start_time,
                batch_stop_time);
            for (int binding_index = 0; binding_index < static_cast<int>(circuit.bindings.size()); ++binding_index) {
                auto& binding = circuit.bindings[static_cast<std::size_t>(binding_index)];
                const auto spike_view =
                    micro_binding_views[static_cast<std::size_t>(circuit_index)][static_cast<std::size_t>(binding_index)];
                binding.output_rule->apply(spike_view,
                                           micro_exposure_soa,
                                           roi_count,
                                           binding.roi_index,
                                           binding.output_state,
                                           binding.output_params,
                                           batch_start_time,
                                           batch_stop_time);
            }
            if (record_micro_spikes_) {
                const auto& binding_views =
                    micro_binding_views[static_cast<std::size_t>(circuit_index)];
                for (std::size_t binding_index = 0; binding_index < binding_views.size(); ++binding_index) {
                    const int roi = circuit.bindings[binding_index].roi_index;
                    result.micro_spikes_by_roi[static_cast<std::size_t>(roi)].append_view(
                        binding_views[binding_index]);
                }
            }
        }

        for (const auto offset: micro_exposure_offsets) {
            current_exposure_soa[offset] = micro_exposure_soa[offset];
        }

        write_history_slot(history,
                           batch_stop % coupling_runtime.history_capacity,
                           roi_count,
                           exposure_count,
                           current_exposure_soa);
        append_exposure_record(result.exposures, current_exposure_soa);

        if (use_input_lookahead) {
            if (batch_stop < step_count) {
                current_input_soa.swap(next_input_soa);
                current_micro_events.swap(next_micro_events);
                clear_event_tables(next_micro_events);
            }
        } else {
            apply_couplings(region_coupling_evaluation,
                            roi_count,
                            input_count,
                            batch_stop,
                            history,
                            current_input_soa);
        }
    }

    return result;
}

}  // namespace mind_sim::cosim
