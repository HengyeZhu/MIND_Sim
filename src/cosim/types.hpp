#pragma once

#include "macro/sim/types.hpp"
#include "micro/sim/types.hpp"

#include <vector>

namespace mind_sim::cosim {

struct SimulationResult {
    std::vector<double> times{};
    mind_sim::macro::sim::ExposureRecord exposures{};
    std::vector<mind_sim::micro::sim::MicroSpikeTable> micro_spikes_by_roi{};
};

}  // namespace mind_sim::cosim
