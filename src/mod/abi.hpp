#pragma once

#include <cstdint>

namespace mind_sim::mod {

constexpr int kModAbiVersion = 0;

enum class AbiRuleKind : int {
    MacroToMacro = 0,
    MicroInput = 1,
    MicroOutput = 2,
    Region = 3,
};

struct AbiRuleDescriptor {
    int abi_version;
    int kind;
    const char* name;
    int exposure_count;
    const char* const* exposure_names;
    int read_source_count;
    const char* const* read_source_exposure_names;
    const char* const* read_source_variable_names;
    int read_target_count;
    const char* const* read_target_exposure_names;
    const char* const* read_target_variable_names;
    int write_source_count;
    const char* const* write_source_exposure_names;
    const char* const* write_source_variable_names;
    int write_target_count;
    const char* const* write_target_exposure_names;
    const char* const* write_target_variable_names;
    int param_count;
    const char* const* param_names;
    const double* param_defaults;
    int state_count;
    const char* const* state_names;
    const double* state_defaults;
};

struct AbiSpikeTable {
    const double* time;
    int size;
};

struct AbiMacroToMacroContext {
    int roi_count;
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
    const double* current_exposures;
    double* exposures;
    int param_count;
    const double* params;
    const int* read_source_offsets;
    const int* read_target_offsets;
    const int* write_source_offsets;
    const int* write_target_offsets;
};

struct AbiMicroInputContext {
    int exposure_count;
    int roi_count;
    int target_roi;
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
    const int* exposure_offsets;
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
    const int* exposure_offsets;
};

struct AbiRegionContext {
    int owner_count;
    const int* roi_indices;
    int roi_count;
    int exposure_count;
    double* exposure_soa;
    int state_count;
    double* state_soa;
    int param_count;
    const double* params_soa;
    const int* exposure_offsets;
    double t;
    double dt;
};

using MacroToMacroApplyFn = void (*)(const AbiMacroToMacroContext*);
using MicroInputApplyFn = void (*)(const AbiMicroInputContext*);
using MicroOutputApplyFn = void (*)(const AbiMicroOutputContext*);
using RegionApplyFn = void (*)(const AbiRegionContext*);

struct AbiRuleEntry {
    const AbiRuleDescriptor* descriptor;
    MacroToMacroApplyFn macro_to_macro_apply;
    MicroInputApplyFn micro_input_apply;
    MicroOutputApplyFn micro_output_apply;
    RegionApplyFn region_apply;
};

struct AbiRuleRegistry {
    int abi_version;
    int rule_count;
    const AbiRuleEntry* rules;
};

using RuleRegistryFn = const AbiRuleRegistry* (*)();

}  // namespace mind_sim::mod
