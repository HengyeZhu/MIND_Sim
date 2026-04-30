#pragma once

#include <string>

namespace mind_sim::micro::sim {

enum class MicroDeviceKind {
    Cpu,
    Gpu,
};

struct MicroDeviceConfig {
    MicroDeviceKind kind{MicroDeviceKind::Cpu};
    int gpu_id{0};
};

[[nodiscard]] MicroDeviceConfig parse_micro_device(const std::string& device);
[[nodiscard]] int cell_permute_for_device(MicroDeviceKind kind) noexcept;

struct CoreNeuronData;
void release_core_neuron_device_runtime(CoreNeuronData& core_data) noexcept;

}  // namespace mind_sim::micro::sim
