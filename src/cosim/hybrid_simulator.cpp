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
                              int exchange_step_count) {
    for (const auto& graph: runtime.graphs) {
        for (int delay_steps: graph.edge_delay_steps) {
            if (delay_steps < exchange_step_count) {
                return false;
            }
        }
    }
    return true;
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

void prepare_micro_events_for_exchange(
    const mind_sim::macro::sim::CouplingEvaluation& micro_coupling_evaluation,
    int roi_count,
    int input_count,
    int exposure_count,
    int exchange_start,
    int exchange_stop,
    double dt_macro,
    const std::vector<double>& history,
    std::vector<double>& micro_input_soa,
    std::vector<mind_sim::macro::frontend::MicroCircuitOwner>& micro_circuits,
    std::vector<mind_sim::micro::sim::MicroEventTable>& prepared_events) {
    for (auto& events: prepared_events) {
        events.clear();
    }
    if (micro_circuits.empty()) {
        return;
    }
    apply_couplings(micro_coupling_evaluation,
                    roi_count,
                    input_count,
                    exposure_count,
                    exchange_start,
                    history,
                    micro_input_soa);
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
            binding.input_rule->apply(micro_input_soa,
                                      roi_count,
                                      binding.roi_index,
                                      binding.input_state,
                                      binding.input_params,
                                      binding.input_random_streams,
                                      exchange_start_time,
                                      exchange_stop_time,
                                      binding.input_port_bases,
                                      events,
                                      binding.input_read_offsets);
        }
    }
}

void bind_spike_views(
    const mind_sim::micro::sim::MicroSpikeTableView& spikes,
    const mind_sim::macro::frontend::MicroCircuitOwner& circuit,
    std::vector<mind_sim::micro::sim::MicroSpikeTable>& binding_spikes,
    std::vector<mind_sim::micro::sim::MicroSpikeTableView>& binding_views,
    bool single_binding,
    double window_start,
    double window_stop) {
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
using mind_sim::macro::sim::aggregate_field_exposures;
using mind_sim::macro::sim::build_coupling_runtime;
using mind_sim::macro::sim::build_region_groups;
using mind_sim::macro::sim::collect_roi_owners;
using mind_sim::macro::sim::coupling_evaluation_for_targets;
using mind_sim::macro::sim::continuous_macro_rois;
using mind_sim::macro::sim::exposure_buffers_to_soa;
using mind_sim::macro::sim::initialize_history;
using mind_sim::macro::sim::step_neural_field;
using mind_sim::macro::sim::validate_single_roi_owner;
using mind_sim::macro::sim::write_history_slot;

Simulator::Simulator(mind_sim::macro::frontend::Network network,
                     double dt_micro,
                     double dt_macro,
                     double exchange_window,
                     bool record_micro_spikes)
    : network_(std::move(network)),
      dt_micro_(dt_micro),
      dt_macro_(dt_macro),
      exchange_window_(exchange_window),
      record_micro_spikes_(record_micro_spikes) {
    if (dt_micro_ <= 0.0 || dt_macro_ <= 0.0 || exchange_window_ <= 0.0) {
        throw std::runtime_error("dt_micro, dt_macro, and exchange_window must be positive");
    }
    if (!is_integer_multiple(dt_macro_, dt_micro_)) {
        throw std::runtime_error("dt_macro must be an integer multiple of dt_micro");
    }
    if (!is_integer_multiple(exchange_window_, dt_macro_)) {
        throw std::runtime_error("exchange_window must be an integer multiple of dt_macro");
    }
    const double min_delay = network_.connectivity().min_positive_delay();
    if (min_delay <= 0.0) {
        throw std::runtime_error("exchange_window requires at least one positive connectivity delay");
    }
    if (exchange_window_ - min_delay > 1e-9) {
        throw std::runtime_error(
            "exchange_window must not exceed the minimum positive connectivity delay");
    }
    exchange_step_count_ = integer_step_count(exchange_window_, dt_macro_);
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

    auto region_owners = network_.region_owners();
    auto field_owners = network_.neural_field_owners();
    auto micro_circuits = network_.micro_circuits();
    const auto roi_owners = collect_roi_owners(region_owners, field_owners, micro_circuits, true);
    validate_single_roi_owner(roi_count,
                              roi_owners,
                              "every ROI must have exactly one owner before run");
    const auto macro_rois = continuous_macro_rois(roi_owners);

    const auto coupling_runtime = build_coupling_runtime(network_, dt_macro_);
    const auto region_coupling_evaluation =
        coupling_evaluation_for_targets(coupling_runtime,
                                        macro_rois,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    const auto micro_coupling_evaluation =
        coupling_evaluation_for_targets(coupling_runtime,
                                        roi_owners.detailed_microcircuit_rois,
                                        roi_count,
                                        input_count,
                                        network_.dc_inputs());
    std::vector<double> history(
        static_cast<std::size_t>(coupling_runtime.history_capacity * roi_count * exposure_count),
        0.0);
    auto current_exposure_soa =
        exposure_buffers_to_soa(network_.initial_exposures(), roi_count, exposure_count);
    for (const auto& owner: field_owners) {
        aggregate_field_exposures(owner, current_exposure_soa);
    }
    initialize_history(history,
                       coupling_runtime.history_capacity,
                       roi_count,
                       exposure_count,
                       current_exposure_soa);

    std::vector<double> current_input_soa;
    apply_couplings(region_coupling_evaluation,
                    roi_count,
                    input_count,
                    exposure_count,
                    0,
                    history,
                    current_input_soa);

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
    const int exposure_sample_count =
        step_count == 0 ? 1 : ((step_count + exchange_step_count_ - 1) / exchange_step_count_) + 1;
    result.exposures.values.reserve(
        static_cast<std::size_t>(exposure_sample_count) *
        result.exposures.roi_indices.size() *
        static_cast<std::size_t>(exposure_count));
    append_exposure_record(result.exposures, current_exposure_soa);
    result.micro_spikes_by_roi.resize(static_cast<std::size_t>(roi_count));

    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTable>> micro_binding_spikes;
    std::vector<std::vector<mind_sim::micro::sim::MicroSpikeTableView>> micro_binding_views;
    std::vector<unsigned char> micro_single_binding;
    micro_binding_spikes.reserve(micro_circuits.size());
    micro_binding_views.reserve(micro_circuits.size());
    micro_single_binding.reserve(micro_circuits.size());
    for (const auto& circuit: micro_circuits) {
        const auto binding_count = circuit.bindings.size();
        micro_binding_spikes.emplace_back(binding_count);
        micro_binding_views.emplace_back(binding_count);
        micro_single_binding.push_back(has_single_contiguous_binding(circuit) ? 1 : 0);
    }
    std::vector<double> micro_input_soa;
    std::vector<double> next_input_soa;
    std::vector<double> micro_exposure_soa(current_exposure_soa.size(), 0.0);
    std::vector<std::size_t> micro_exposure_offsets;
    micro_exposure_offsets.reserve(
        roi_owners.detailed_microcircuit_rois.size() * static_cast<std::size_t>(exposure_count));
    for (int roi: roi_owners.detailed_microcircuit_rois) {
        for (int exposure = 0; exposure < exposure_count; ++exposure) {
            micro_exposure_offsets.push_back(static_cast<std::size_t>(exposure * roi_count + roi));
        }
    }
    std::vector<mind_sim::micro::sim::MicroWindowToken> micro_tokens(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> current_micro_events(micro_circuits.size());
    std::vector<mind_sim::micro::sim::MicroEventTable> next_micro_events(micro_circuits.size());
    const bool use_micro_pipeline =
        !micro_circuits.empty() && can_prepare_inputs_ahead(coupling_runtime, exchange_step_count_);

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
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            const auto spikes =
                micro_runtimes[static_cast<std::size_t>(circuit_index)]->finish_window(
                    micro_tokens[static_cast<std::size_t>(circuit_index)]);
            bind_spike_views(
                spikes,
                circuit,
                micro_binding_spikes[static_cast<std::size_t>(circuit_index)],
                micro_binding_views[static_cast<std::size_t>(circuit_index)],
                micro_single_binding[static_cast<std::size_t>(circuit_index)] != 0,
                window_start_time,
                window_stop_time);
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
                                           window_start_time,
                                           window_stop_time,
                                           binding.output_write_offsets);
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
    };

    const auto run_micro_window_synchronously = [&](double window_start_time,
                                                   double window_stop_time,
                                                   std::vector<mind_sim::micro::sim::MicroEventTable>& events) {
        for (std::size_t circuit = 0; circuit < micro_runtimes.size(); ++circuit) {
            append_events(micro_runtimes[circuit]->scheduled_events(), events[circuit]);
        }
        for (int circuit_index = 0; circuit_index < static_cast<int>(micro_circuits.size()); ++circuit_index) {
            auto& circuit = micro_circuits[static_cast<std::size_t>(circuit_index)];
            const auto spikes =
                micro_runtimes[static_cast<std::size_t>(circuit_index)]->advance_window(
                    window_start_time,
                    window_stop_time);
            bind_spike_views(
                spikes,
                circuit,
                micro_binding_spikes[static_cast<std::size_t>(circuit_index)],
                micro_binding_views[static_cast<std::size_t>(circuit_index)],
                micro_single_binding[static_cast<std::size_t>(circuit_index)] != 0,
                window_start_time,
                window_stop_time);
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
                                           window_start_time,
                                           window_stop_time,
                                           binding.output_write_offsets);
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
    };

    if (use_micro_pipeline && step_count > 0) {
        const int first_exchange_stop = std::min(step_count, exchange_step_count_);
        prepare_micro_events_for_exchange(micro_coupling_evaluation,
                                       roi_count,
                                       input_count,
                                       exposure_count,
                                       0,
                                       first_exchange_stop,
                                       dt_macro_,
                                       history,
                                       micro_input_soa,
                                       micro_circuits,
                                       current_micro_events);
        submit_micro_window(0.0, first_exchange_stop * dt_macro_, current_micro_events);
    }

    for (int exchange_start = 0; exchange_start < step_count; exchange_start += exchange_step_count_) {
        const int exchange_stop = std::min(step_count, exchange_start + exchange_step_count_);
        const double exchange_start_time = exchange_start * dt_macro_;
        const double exchange_stop_time = exchange_stop * dt_macro_;

        if (!use_micro_pipeline) {
            prepare_micro_events_for_exchange(micro_coupling_evaluation,
                                           roi_count,
                                           input_count,
                                           exposure_count,
                                           exchange_start,
                                           exchange_stop,
                                           dt_macro_,
                                           history,
                                           micro_input_soa,
                                           micro_circuits,
                                           current_micro_events);
            run_micro_window_synchronously(exchange_start_time,
                                           exchange_stop_time,
                                           current_micro_events);
        } else {
            finish_micro_window(exchange_start_time, exchange_stop_time);
        }

        if (use_micro_pipeline && exchange_stop < step_count) {
            const int next_exchange_stop = std::min(step_count, exchange_stop + exchange_step_count_);
            apply_couplings(region_coupling_evaluation,
                            roi_count,
                            input_count,
                            exposure_count,
                            exchange_stop,
                            history,
                            next_input_soa);
            prepare_micro_events_for_exchange(micro_coupling_evaluation,
                                           roi_count,
                                           input_count,
                                           exposure_count,
                                           exchange_stop,
                                           next_exchange_stop,
                                           dt_macro_,
                                           history,
                                           micro_input_soa,
                                           micro_circuits,
                                           next_micro_events);
            submit_micro_window(exchange_stop_time,
                                next_exchange_stop * dt_macro_,
                                next_micro_events);
        }

        for (int step = exchange_start; step < exchange_stop; ++step) {
            const double start_time = step * dt_macro_;
            const double stop_time = (step + 1) * dt_macro_;

            for (auto& owner: field_owners) {
                step_neural_field(owner,
                                  roi_count,
                                  current_input_soa,
                                  current_exposure_soa,
                                  start_time,
                                  stop_time - start_time);
            }
            for (auto& group: region_groups) {
                group.rule->step_group(group.roi_indices,
                                       roi_count,
                                       current_input_soa,
                                       current_exposure_soa,
                                       group.state_soa,
                                       group.params_soa,
                                       group.read_input_offsets,
                                       group.write_exposure_offsets,
                                       start_time,
                                       stop_time - start_time);
            }

            if (step + 1 == exchange_stop) {
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
                            exposure_count,
                            step + 1,
                            history,
                            current_input_soa);
        }

        write_history_slot(history,
                           exchange_stop % coupling_runtime.history_capacity,
                           roi_count,
                           exposure_count,
                           current_exposure_soa);
        append_exposure_record(result.exposures, current_exposure_soa);

        if (use_micro_pipeline && exchange_stop < step_count) {
            current_input_soa.swap(next_input_soa);
            for (auto& events: next_micro_events) {
                events.clear();
            }
        }
        if (!use_micro_pipeline) {
            apply_couplings(region_coupling_evaluation,
                            roi_count,
                            input_count,
                            exposure_count,
                            exchange_stop,
                            history,
                            current_input_soa);
        }
    }

    return result;
}

}  // namespace mind_sim::cosim
