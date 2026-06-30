#pragma once

#include <cstdint>

#if defined(_WIN32)
#define MIND_RULE_EXPORT __declspec(dllexport)
#else
#define MIND_RULE_EXPORT __attribute__((visibility("default")))
#endif

namespace mind_sim::mod {

enum class RuleKind : int {
    MacroToMacro = 0,
    MicroInput = 1,
    MicroOutput = 2,
    Region = 3,
};

struct NameList {
    int count;
    const char* const* names;
};

struct BindingList {
    int count;
    const char* const* exposure_names;
    const char* const* variable_names;
};

struct ValueList {
    int count;
    const char* const* names;
    const double* defaults;
};

struct SpikeTable {
    const double* time;
    int size;
};

struct MacroToMacroContext {
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

struct MicroInputContext {
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
    std::uint64_t rng_stream_id;
    int exchange_start_step;
    const int* exposure_offsets;
    void* event_user_data;
    void (*emit_event)(void* user_data, double time, int source_index);
};

struct MicroOutputContext {
    int exposure_count;
    const SpikeTable* spikes;
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

struct RegionContext {
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

using MacroToMacroApplyFn = void (*)(const MacroToMacroContext*);
using MicroInputApplyFn = void (*)(const MicroInputContext*);
using MicroOutputApplyFn = void (*)(const MicroOutputContext*);
using RegionApplyFn = void (*)(const RegionContext*);

struct RuleRegistrar {
    void* user_data;
    void (*register_region)(void* user_data,
                            const char* name,
                            NameList exposures,
                            ValueList states,
                            ValueList params,
                            RegionApplyFn apply);
    void (*register_macro_to_macro)(void* user_data,
                                    const char* name,
                                    BindingList read_source,
                                    BindingList read_target,
                                    BindingList write_source,
                                    BindingList write_target,
                                    ValueList params,
                                    MacroToMacroApplyFn apply);
    void (*register_micro_input)(void* user_data,
                                 const char* name,
                                 BindingList read_source,
                                 ValueList states,
                                 ValueList params,
                                 MicroInputApplyFn apply);
    void (*register_micro_output)(void* user_data,
                                  const char* name,
                                  BindingList write_target,
                                  ValueList states,
                                  ValueList params,
                                  MicroOutputApplyFn apply);
};

using RuleRegFn = void (*)(RuleRegistrar*);

}  // namespace mind_sim::mod
