#include "macro/sim/runtime.hpp"

#include "macro/sim/runtime_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mind_sim::macro::sim {

MacroRuntime::MacroRuntime(mind_sim::macro::frontend::Network network)
    : network_(std::move(network)) {
    if (!network_.micro_circuits().empty()) {
        throw std::runtime_error("MacroRuntime only runs networks without micro ROI owners");
    }
}

mind_sim::macro::sim::MacroSimulationResult MacroRuntime::run(double t_stop) {
    return run(t_stop, network_.dt());
}

mind_sim::macro::sim::MacroSimulationResult MacroRuntime::run(double t_stop, double dt_macro) {
    if (t_stop < 0.0) {
        throw std::runtime_error("t_stop must be non-negative");
    }
    if (dt_macro <= 0.0) {
        throw std::runtime_error("dt_macro must be positive");
    }
    const int roi_count = network_.roi_count();
    const int input_count = network_.input_count();
    const int output_count = network_.output_count();
    const int step_count = static_cast<int>(std::ceil(t_stop / dt_macro - 1e-12));
    auto region_owners = network_.region_owners();
    auto field_owners = network_.neural_field_owners();
    const auto roi_owners =
        collect_roi_owners(region_owners, field_owners, network_.micro_circuits(), false);
    validate_single_roi_owner(roi_count,
                              roi_owners,
                              "macro-only run requires every ROI to have exactly one macro owner");
    const auto macro_rois = continuous_macro_rois(roi_owners);

    const auto macro_to_macro_runtime = build_macro_to_macro_runtime(network_, dt_macro);
    std::vector<double> history(
        static_cast<std::size_t>(macro_to_macro_runtime.history_capacity * roi_count * output_count),
        0.0);
    auto current_output_soa =
        output_buffers_to_soa(network_.output_history_start(), roi_count, output_count);
    for (const auto& owner: field_owners) {
        aggregate_field_outputs(owner, current_output_soa);
    }
    initialize_history(history,
                       macro_to_macro_runtime.history_capacity,
                       roi_count,
                       output_count,
                       current_output_soa,
                       network_.initial_history(),
                       network_.initial_history_time_count());

    auto region_groups = build_region_groups(region_owners);
    const auto macro_to_macro_evaluation = macro_to_macro_evaluation_for_targets(macro_to_macro_runtime,
                                                                     macro_rois,
                                                                     roi_count,
                                                                     input_count,
                                                                     network_.dc_inputs());
    std::vector<double> current_input_soa;
    apply_macro_to_macro(macro_to_macro_evaluation,
                    roi_count,
                    input_count,
                    output_count,
                    0,
                    history,
                    current_input_soa);

    MacroSimulationResult result;
    result.times.resize(static_cast<std::size_t>(step_count) + 1);
    for (int step = 0; step <= step_count; ++step) {
        result.times[static_cast<std::size_t>(step)] = std::min(t_stop, step * dt_macro);
    }
    result.records.roi_count = roi_count;
    result.records.roi_indices = network_.recorded_rois();
    result.records.output_indices = network_.recorded_outputs();
    result.records.output_count =
        static_cast<int>(result.records.output_indices.size());
    result.records.values.reserve(
        (static_cast<std::size_t>(step_count) + 1) *
        result.records.roi_indices.size() *
        static_cast<std::size_t>(result.records.output_count));
    append_record_table(result.records, current_output_soa, output_count);

    for (int step = 0; step < step_count; ++step) {
        const double start_time = step * dt_macro;
        const double stop_time = std::min(t_stop, start_time + dt_macro);

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

        write_history_slot(history,
                           (step + 1) % macro_to_macro_runtime.history_capacity,
                           roi_count,
                           output_count,
                           current_output_soa);
        append_record_table(result.records, current_output_soa, output_count);

        apply_macro_to_macro(macro_to_macro_evaluation,
                        roi_count,
                        input_count,
                        output_count,
                        step + 1,
                        history,
                        current_input_soa);
    }

    return result;
}

}  // namespace mind_sim::macro::sim
