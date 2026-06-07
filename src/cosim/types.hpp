#pragma once

#include "macro/sim/types.hpp"
#include "micro/sim/types.hpp"

#include <vector>

namespace mind_sim::cosim {

struct SimulationResult {
    std::vector<double> times{};
    mind_sim::macro::sim::RecordTable records{};
};

}  // namespace mind_sim::cosim
