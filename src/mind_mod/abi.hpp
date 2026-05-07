#pragma once

namespace mind_sim::mind_mod {

constexpr int kMindModAbiVersion = 2;

enum class AbiRuleKind : int {
    Coupling = 0,
    MicroInput = 1,
    MicroOutput = 2,
};

struct AbiRuleDescriptor {
    int abi_version;
    int kind;
    const char* name;
    int read_count;
    const char* const* read_names;
    int write_count;
    const char* const* write_names;
    int emit_count;
    const char* const* emit_names;
    int param_count;
    const char* const* param_names;
    const double* param_defaults;
    int state_count;
    const char* const* state_names;
    const double* state_defaults;
    int random_count;
    const char* const* random_names;
};

struct AbiEventWriter {
    void* user;
    void (*write)(void*, double, int);
};

struct AbiSpikeTable {
    const double* time;
    const int* gid;
    int size;
};

struct AbiRandomStream {
    void* user;
    double (*uniform)(void*, int, int);
};

struct AbiCouplingContext {
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
    const int* read_exposure_offsets;
    const int* write_input_offsets;
};

struct AbiMicroInputContext {
    int input_count;
    int roi_count;
    int roi;
    const double* input_soa;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    int input_port_count;
    const int* input_port_bases;
    AbiEventWriter* event_writer;
    const int* read_input_offsets;
    int random_count;
    AbiRandomStream* random_streams;
};

struct AbiMicroOutputContext {
    int exposure_count;
    const AbiSpikeTable* spikes;
    int roi_count;
    int roi;
    double* exposure_soa;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    const int* write_exposure_offsets;
};

using DescriptorFn = const AbiRuleDescriptor* (*)();
using CouplingApplyFn = void (*)(const AbiCouplingContext*);
using MicroInputApplyFn = void (*)(const AbiMicroInputContext*);
using MicroOutputApplyFn = void (*)(const AbiMicroOutputContext*);

}  // namespace mind_sim::mind_mod
