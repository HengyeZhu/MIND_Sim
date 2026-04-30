#include "micro/sim/device.hpp"

#include <stdexcept>

namespace mind_sim::micro::sim {

MicroDeviceConfig parse_micro_device(const std::string& device) {
    if (device == "cpu") {
        return MicroDeviceConfig{
            .kind = MicroDeviceKind::Cpu,
            .gpu_id = 0,
        };
    }
    if (device == "gpu") {
        return MicroDeviceConfig{
            .kind = MicroDeviceKind::Gpu,
            .gpu_id = 0,
        };
    }
    throw std::runtime_error("micro device must be 'cpu' or 'gpu'");
}

int cell_permute_for_device(MicroDeviceKind kind) noexcept {
    return kind == MicroDeviceKind::Gpu ? 2 : 1;
}

}  // namespace mind_sim::micro::sim
