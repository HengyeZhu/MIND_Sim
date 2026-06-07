#pragma once

#include "macro/frontend/network.hpp"
#include "macro/sim/types.hpp"

namespace mind_sim::macro::sim {

class MacroRuntime {
  public:
    explicit MacroRuntime(mind_sim::macro::frontend::Network network);

    [[nodiscard]] mind_sim::macro::sim::MacroSimulationResult run(double t_stop);
    [[nodiscard]] mind_sim::macro::sim::MacroSimulationResult run(double t_stop, double dt_macro);

  private:
    mind_sim::macro::frontend::Network network_;
};

}  // namespace mind_sim::macro::sim
