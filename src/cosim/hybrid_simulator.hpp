#pragma once

#include "cosim/types.hpp"
#include "macro/frontend/network.hpp"

namespace mind_sim::cosim {

class Simulator {
  public:
    Simulator(mind_sim::macro::frontend::Network network,
              double dt_micro,
              double dt_macro,
              double exchange_window,
              bool record_micro_spikes);

    [[nodiscard]] mind_sim::cosim::SimulationResult run(double t_stop);

  private:
    mind_sim::macro::frontend::Network network_;
    double dt_micro_{0.0};
    double dt_macro_{0.0};
    double exchange_window_{0.0};
    int exchange_step_count_{1};
    bool record_micro_spikes_{true};
};

}  // namespace mind_sim::cosim
