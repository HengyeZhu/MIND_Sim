#include "bridge/sim/rule_codegen.hpp"

#include "utils/rule_source_common.hpp"

namespace mind_sim::bridge::sim::codegen {

using namespace mind_sim::utils::rule_source;

std::string micro_input_rule_source(const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& ports,
                                    const std::vector<std::string>& states,
                                    const std::vector<std::string>& params,
                                    const std::string& emit) {
    const auto port_names = names(ports, "input port");
    const auto schema_value =
        schema("MicroInputRule", inputs, {}, states, params, input_runtime_symbols(), false);
    const auto usage =
        validate_code("MicroInputRule", emit, input_runtime_symbols(), input_members(schema_value, port_names));
    const auto input_fields = used_fields(schema_value.inputs, usage, "in");
    const auto state_fields = used_fields(schema_value.states, usage, "s");
    const auto param_fields = used_fields(schema_value.params, usage, "p");
    const auto emit_fields = used_fields(port_names, usage, "emit");
    require_declared_fields_used("MicroInputRule", "state", state_fields, schema_value.states);
    require_declared_fields_used("MicroInputRule", "param", param_fields, schema_value.params);
    require_declared_fields_used("MicroInputRule", "input port", emit_fields, port_names);

    std::ostringstream source;
    source << common_helpers();
    if (!input_fields.empty()) {
        source << value_struct("mind_input_values", input_fields, "double");
    }
    if (!state_fields.empty()) {
        source << ref_struct("mind_input_state", state_fields, "double");
    }
    if (!param_fields.empty()) {
        source << params_struct("mind_input_params", param_fields);
    }
    source << R"cpp(
struct mind_event_writer {
    void* user;
    void (*write)(void*, double, int);
};

struct mind_window {
    double start;
    double stop;
    double duration;
};

struct mind_emit {
    mind_event_writer* event_writer;
    const int* port_bases;

)cpp";
    for (std::size_t port = 0; port < port_names.size(); ++port) {
        source << "    void " << port_names[port] << "(double time, int local_index) const {\n";
        source << "        event_writer->write(event_writer->user, time, port_bases[" << port
               << "] + local_index);\n";
        source << "    }\n";
    }
    source << R"cpp(};

extern "C" void mind_micro_input_rule_apply(
    int input_count,
    int roi_count,
    int roi,
    const float* input_soa,
    int state_count,
    double* state,
    int param_count,
    const double* params,
    double start_time,
    double stop_time,
    int input_port_count,
    const int* input_port_bases,
    mind_event_writer* event_writer) {
    (void)input_count;
    (void)roi_count;
    (void)roi;
    (void)state_count;
    (void)param_count;
    (void)input_port_count;
)cpp";
    if (!input_fields.empty()) {
        source << input_soa_init_selected("mind_input_values",
                                          "in",
                                          input_fields,
                                          schema_value.inputs,
                                          "input_soa",
                                          "roi_count",
                                          "roi",
                                          4);
    }
    if (!state_fields.empty()) {
        source << flat_ref_init_selected("mind_input_state",
                                         "s",
                                         state_fields,
                                         schema_value.states,
                                         "state",
                                         4);
    }
    if (!param_fields.empty()) {
        source << flat_value_init_selected("mind_input_params",
                                           "p",
                                           param_fields,
                                           schema_value.params,
                                           "params",
                                           4);
    }
    source << R"cpp(    mind_window window{start_time, stop_time, stop_time - start_time};
    mind_emit emit{event_writer, input_port_bases};
)cpp";
    source << indent_block(emit, 4);
    source << R"cpp(}
)cpp";
    return source.str();
}

std::string micro_output_rule_source(const std::vector<std::string>& exposures,
                                     const std::vector<std::string>& states,
                                     const std::vector<std::string>& params,
                                     const std::string& spike,
                                     const std::string& finish) {
    const auto schema_value =
        schema("MicroOutputRule", {}, exposures, states, params, exposure_runtime_symbols());
    const auto spike_usage =
        validate_code("MicroOutputRule", spike, exposure_runtime_symbols(), exposure_members(schema_value));
    const auto finish_usage =
        validate_code("MicroOutputRule", finish, exposure_runtime_symbols(), exposure_members(schema_value));
    MemberUsage usage = spike_usage;
    merge_usage(usage, finish_usage);
    const auto output_fields = used_fields(schema_value.exposures, usage, "out");
    const auto state_fields = used_fields(schema_value.states, usage, "s");
    const auto param_fields = used_fields(schema_value.params, usage, "p");
    require_declared_fields_used("MicroOutputRule", "state", state_fields, schema_value.states);
    require_declared_fields_used("MicroOutputRule", "param", param_fields, schema_value.params);
    require_all_fields("MicroOutputRule", "out", output_fields, schema_value.exposures);

    std::ostringstream source;
    source << common_helpers();
    source << ref_struct("mind_output_values", output_fields, "float");
    if (!state_fields.empty()) {
        source << ref_struct("mind_micro_output_state", state_fields, "double");
    }
    if (!param_fields.empty()) {
        source << params_struct("mind_micro_output_params", param_fields);
    }
    source << R"cpp(
struct mind_spike_table {
    const double* time;
    const int* gid;
    int size;
};

struct mind_window {
    double start;
    double stop;
    double duration;
};

struct mind_spike {
    double t;
    int gid;
};

extern "C" void mind_micro_output_rule_apply(
    int exposure_count,
    const mind_spike_table* spikes,
    int roi_count,
    int roi,
    float* exposure_soa,
    int state_count,
    double* state,
    int param_count,
    const double* params,
    double start_time,
    double stop_time) {
    (void)exposure_count;
    (void)roi_count;
    (void)roi;
    (void)state_count;
    (void)param_count;
    const int spike_count = spikes->size;
    const double duration = stop_time - start_time;
)cpp";
    source << output_soa_init_selected("mind_output_values",
                                       "out",
                                       output_fields,
                                       schema_value.exposures,
                                       "exposure_soa",
                                       "roi_count",
                                       "roi",
                                       4);
    if (!state_fields.empty()) {
        source << flat_ref_init_selected("mind_micro_output_state",
                                         "s",
                                         state_fields,
                                         schema_value.states,
                                         "state",
                                         4);
    }
    if (!param_fields.empty()) {
        source << flat_value_init_selected("mind_micro_output_params",
                                           "p",
                                           param_fields,
                                           schema_value.params,
                                           "params",
                                           4);
    }
    source << R"cpp(    mind_window window{start_time, stop_time, duration};
    for (int spike_index = 0; spike_index < spike_count; ++spike_index) {
        mind_spike spike{spikes->time[spike_index],
                         spikes->gid[spike_index]};
)cpp";
    source << indent_block(spike, 8);
    source << R"cpp(    }
)cpp";
    source << indent_block(finish, 4);
    source << R"cpp(}
)cpp";
    return source.str();
}

}  // namespace mind_sim::bridge::sim::codegen
