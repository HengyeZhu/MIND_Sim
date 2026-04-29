#include "macro/sim/rule_codegen.hpp"

#include "utils/rule_source_common.hpp"

namespace mind_sim::macro::sim::codegen {

using namespace mind_sim::utils::rule_source;

std::string kernel_name(const std::string& name, const std::string& what) {
    return names({name}, what).front();
}

std::string region_rule_source(const std::vector<std::string>& inputs,
                               const std::vector<std::string>& exposures,
                               const std::vector<std::string>& states,
                               const std::vector<std::string>& params,
                               const std::string& update) {
    const auto schema_value =
        schema("RegionRule", inputs, exposures, states, params, region_runtime_symbols());
    const auto usage =
        validate_code("RegionRule", update, region_runtime_symbols(), region_members(schema_value));
    const auto input_fields = used_fields(schema_value.inputs, usage, "in");
    const auto output_fields = used_fields(schema_value.exposures, usage, "out");
    const auto state_fields = used_fields(schema_value.states, usage, "s");
    const auto param_fields = used_fields(schema_value.params, usage, "p");
    require_declared_fields_used("RegionRule", "state", state_fields, schema_value.states);
    require_declared_fields_used("RegionRule", "param", param_fields, schema_value.params);
    require_all_fields("RegionRule", "out", output_fields, schema_value.exposures);

    std::ostringstream source;
    source << common_helpers();
    if (!input_fields.empty()) {
        source << value_struct("mind_region_input", input_fields, "double");
    }
    source << ref_struct("mind_region_output", output_fields, "float");
    if (!state_fields.empty()) {
        source << ref_struct("mind_region_state", state_fields, "double");
    }
    if (!param_fields.empty()) {
        source << params_struct("mind_region_params", param_fields);
    }
    source << R"cpp(
extern "C" void mind_region_rule_step(
    int owner_count,
    const int* roi_indices,
    int roi_count,
    int input_count,
    int exposure_count,
    const float* input_soa,
    float* exposure_soa,
    int state_count,
    double* state_soa,
    int param_count,
    const double* params_soa,
    double t,
    double dt) {
    (void)input_count;
    (void)exposure_count;
    (void)state_count;
    (void)param_count;
    (void)t;
    (void)dt;
)cpp";
    source << R"cpp(    for (int unit = 0; unit < owner_count; ++unit) {
        const int roi = roi_indices[unit];
)cpp";
    if (!input_fields.empty()) {
        source << input_soa_init_selected("mind_region_input",
                                          "in",
                                          input_fields,
                                          schema_value.inputs,
                                          "input_soa",
                                          "roi_count",
                                          "roi",
                                          8);
    }
    source << output_soa_init_selected("mind_region_output",
                                       "out",
                                       output_fields,
                                       schema_value.exposures,
                                       "exposure_soa",
                                       "roi_count",
                                       "roi",
                                       8);
    if (!state_fields.empty()) {
        source << state_soa_init_selected("mind_region_state",
                                          "s",
                                          state_fields,
                                          schema_value.states,
                                          "state_soa",
                                          "owner_count",
                                          "unit",
                                          8);
    }
    if (!param_fields.empty()) {
        source << params_soa_init_selected("mind_region_params",
                                           "p",
                                           param_fields,
                                           schema_value.params,
                                           "params_soa",
                                           "owner_count",
                                           "unit",
                                           8);
    }
    source << indent_block(update, 8);
    source << R"cpp(    }
}
)cpp";
    return source.str();
}

std::string coupling_projection_rule_source(const std::vector<std::string>& inputs,
                                            const std::vector<std::string>& exposures,
                                            const std::vector<std::string>& params,
                                            const std::string& edge,
                                            const std::string& finish,
                                            int roi_count) {
    if (roi_count <= 0) {
        throw std::runtime_error("CouplingRule projection requires positive roi_count");
    }

    const auto schema_value =
        schema("CouplingRule", inputs, exposures, {}, params, coupling_edge_runtime_symbols());
    const auto edge_usage =
        validate_code("CouplingRule", edge, coupling_edge_runtime_symbols(), coupling_edge_members(schema_value));
    const auto finish_usage =
        validate_code("CouplingRule", finish, coupling_finish_runtime_symbols(), coupling_finish_members(schema_value));
    MemberUsage usage = edge_usage;
    merge_usage(usage, finish_usage);
    const auto src_fields = used_fields(schema_value.exposures, edge_usage, "src");
    const auto dst_fields = used_fields(schema_value.exposures, usage, "dst");
    const auto input_fields = used_fields(schema_value.inputs, usage, "in");
    const auto edge_fields = used_edge_fields(edge_usage);
    const auto param_fields = used_fields(schema_value.params, usage, "p");
    require_declared_fields_used("CouplingRule", "param", param_fields, schema_value.params);

    std::ostringstream source;
    source << common_helpers();
    if (!src_fields.empty()) {
        source << value_struct("mind_coupling_source", src_fields, "double");
    }
    if (!dst_fields.empty()) {
        source << value_struct("mind_coupling_target", dst_fields, "double");
    }
    if (!input_fields.empty()) {
        source << value_struct("mind_coupling_input", input_fields, "double");
    }
    if (!edge_fields.empty()) {
        source << edge_struct(edge_fields);
    }
    source << params_struct("mind_coupling_params", schema_value.params);
    source << "\nconstexpr int mind_roi_count = " << roi_count << ";\n";
    source << "constexpr int mind_input_count = " << schema_value.inputs.size() << ";\n";
    source << "constexpr int mind_exposure_count = " << schema_value.exposures.size() << ";\n";
    source << R"cpp(
extern "C" void mind_coupling_rule_apply(
    int roi_count,
    int input_count,
    int exposure_count,
    int history_capacity,
    int step,
    int target_count,
    const int* target_indices,
    const int* target_edge_offsets,
    const int* edge_sources,
    const float* edge_weights,
    const int* edge_delay_steps,
    const int* edge_delay_offsets,
    const float* history,
    float* inputs,
    int param_count,
    const double* params) {
    (void)roi_count;
    (void)input_count;
    (void)exposure_count;
    (void)target_count;
    (void)target_indices;
    (void)target_edge_offsets;
    (void)edge_sources;
    (void)edge_weights;
    (void)edge_delay_steps;
    (void)edge_delay_offsets;
    (void)history;
    (void)inputs;
    (void)param_count;
)cpp";
    source << params_flat_init("mind_coupling_params", "p", schema_value.params, "params", 4);
    if (!src_fields.empty() || !dst_fields.empty()) {
        source << R"cpp(    constexpr int mind_history_stride = mind_exposure_count * mind_roi_count;
    const int history_size = history_capacity * mind_history_stride;
    const int current_history_offset = (step % history_capacity) * mind_history_stride;
)cpp";
    }
    if (!dst_fields.empty()) {
        source << R"cpp(    const float* target_history_slot = history + current_history_offset;
)cpp";
    }

    source << R"cpp(    for (int target_pos = 0; target_pos < target_count; ++target_pos) {
        const int target_roi = target_indices[target_pos];
)cpp";

    if (!input_fields.empty()) {
        source << soa_value_init_selected("mind_coupling_input",
                                          "in",
                                          input_fields,
                                          schema_value.inputs,
                                          "inputs",
                                          "mind_roi_count",
                                          "target_roi",
                                          8);
    }
    if (!dst_fields.empty()) {
        source << soa_value_init_selected("mind_coupling_target",
                                          "dst",
                                          dst_fields,
                                          schema_value.exposures,
                                          "target_history_slot",
                                          "mind_roi_count",
                                          "target_roi",
                                          8);
    }

    source << R"cpp(        const int edge_begin = target_edge_offsets[target_roi];
        const int edge_end = target_edge_offsets[target_roi + 1];
        for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
            const int source_roi = edge_sources[edge_index];
)cpp";
    if (has_field(edge_fields, "weight")) {
        source << R"cpp(            const float weight = edge_weights[edge_index];
)cpp";
    }
    if (has_field(edge_fields, "delay_steps")) {
        source << R"cpp(            const int delay_steps_value = edge_delay_steps[edge_index];
)cpp";
    }
    if (has_field(edge_fields, "source_roi")) {
        source << R"cpp(            const int source_roi_value = source_roi;
)cpp";
    }
    if (has_field(edge_fields, "target_roi")) {
        source << R"cpp(            const int target_roi_value = target_roi;
)cpp";
    }
    if (!src_fields.empty()) {
        source << R"cpp(            int history_offset = current_history_offset + edge_delay_offsets[edge_index];
            if (history_offset >= history_size) {
                history_offset -= history_size;
            }
            const float* history_slot = history + history_offset;
)cpp";
        source << soa_value_init_selected("mind_coupling_source",
                                          "src",
                                          src_fields,
                                          schema_value.exposures,
                                          "history_slot",
                                          "mind_roi_count",
                                          "source_roi",
                                          12);
    }
    if (!edge_fields.empty()) {
        std::vector<std::string> init_fields = edge_fields;
        for (auto& field: init_fields) {
            if (field == "source_roi") {
                field = "source_roi_value";
            } else if (field == "target_roi") {
                field = "target_roi_value";
            } else if (field == "delay_steps") {
                field = "delay_steps_value";
            }
        }
        source << edge_init(init_fields, 12);
    }
    source << indent_block(edge, 12);
    source << R"cpp(        }
)cpp";
    source << indent_block(finish, 8);
    if (!input_fields.empty()) {
        source << soa_store_selected("in",
                                     input_fields,
                                     schema_value.inputs,
                                     "inputs",
                                     "mind_roi_count",
                                     "target_roi",
                                     8);
    }
    source << R"cpp(    }
}
)cpp";
    return source.str();
}

}  // namespace mind_sim::macro::sim::codegen
