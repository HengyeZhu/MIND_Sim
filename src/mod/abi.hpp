#pragma once

#include <cstdint>

namespace mind_sim::mod {

constexpr int kModAbiVersion = 14;

enum class AbiRuleKind : int {
    MacroToMacro = 0,
    MicroInput = 1,
    MicroOutput = 2,
    Region = 3,
    NeuralField = 4,
};

struct AbiRuleDescriptor {
    int abi_version;
    int kind;
    const char* name;
    int target_input_count;
    const char* const* target_input_names;
    int source_exposure_count;
    const char* const* source_exposure_names;
    int param_count;
    const char* const* param_names;
    const double* param_defaults;
    int state_count;
    const char* const* state_names;
    const double* state_defaults;
    int local_state_count;
    const char* const* local_state_names;
};

struct AbiSpikeTable {
    const double* time;
    int size;
};

struct AbiMacroToMacroContext {
    int roi_count;
    int input_count;
    int exposure_count;
    int history_capacity;
    int step;
    int target_count;
    const int* target_indices;
    const int* target_edge_offsets;
    const int* edge_sources;
    const double* edge_weights;
    const int* edge_delay_steps;
    const int* edge_delay_offsets;
    const double* history;
    double* inputs;
    int param_count;
    const double* params;
    const int* source_exposure_offsets;
    const int* target_input_offsets;
};

struct AbiMicroInputContext {
    int input_count;
    int exposure_count;
    int roi_count;
    int target_roi;
    const double* input_trace_soa;
    const double* exposure_trace_soa;
    int sample_count;
    double sample_dt;
    int source_count;
    const int* source_indices;
    const int* source_ids;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    std::uint64_t rng_seed;
    int exchange_start_step;
    const int* target_input_offsets;
    const int* source_exposure_offsets;
    void* event_user_data;
    void (*emit_event)(void* user_data, double time, int source_index);
};

struct AbiMicroOutputContext {
    int exposure_count;
    const AbiSpikeTable* spikes;
    int roi_count;
    int target_roi;
    double* exposure_soa;
    int sample_count;
    double sample_dt;
    double* exposure_trace_soa;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    const int* source_exposure_offsets;
};

struct AbiRegionContext {
    int owner_count;
    const int* roi_indices;
    int roi_count;
    int input_count;
    const double* input_soa;
    int exposure_count;
    double* exposure_soa;
    int state_count;
    double* state_soa;
    int param_count;
    const double* params_soa;
    const int* target_input_offsets;
    const int* source_exposure_offsets;
    double t;
    double dt;
};

struct AbiNeuralFieldContext {
    int node_count;
    const int* node_to_roi;
    int roi_count;
    int input_count;
    const double* input_soa;
    int state_count;
    const double* previous_state_soa;
    double* state_soa;
    int param_count;
    const double* params;
    const int* local_indptr;
    const int* local_indices;
    const double* local_weights;
    const int* target_input_offsets;
    double t;
    double dt;
};

using MacroToMacroApplyFn = void (*)(const AbiMacroToMacroContext*);
using MicroInputApplyFn = void (*)(const AbiMicroInputContext*);
using MicroOutputApplyFn = void (*)(const AbiMicroOutputContext*);
using RegionApplyFn = void (*)(const AbiRegionContext*);
using NeuralFieldApplyFn = void (*)(const AbiNeuralFieldContext*);

struct AbiRuleEntry {
    const AbiRuleDescriptor* descriptor;
    MacroToMacroApplyFn macro_to_macro_apply;
    MicroInputApplyFn micro_input_apply;
    MicroOutputApplyFn micro_output_apply;
    RegionApplyFn region_apply;
    NeuralFieldApplyFn neural_field_apply;
};

struct AbiRuleRegistry {
    int abi_version;
    int rule_count;
    const AbiRuleEntry* rules;
};

using RuleRegistryFn = const AbiRuleRegistry* (*)();

}  // namespace mind_sim::mod
