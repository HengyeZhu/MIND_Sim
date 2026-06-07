#pragma once

#include "cosim/types.hpp"
#include "macro/frontend/network.hpp"

#include <cstdint>

namespace mind_sim::cosim {

class Simulator {
  public:
    Simulator(mind_sim::macro::frontend::Network network,
              std::uint64_t macro2micro_seed = 1);

    [[nodiscard]] mind_sim::cosim::SimulationResult run(double t_stop);

  private:
    void configure_timing();
    void validate_timing() const;

    mind_sim::macro::frontend::Network network_;
    double dt_micro_{0.0};
    double dt_macro_{0.0};
    double exchange_window_{0.0};
    int exchange_step_count_{1};
    std::uint64_t macro2micro_seed_{1};
};

}  // namespace mind_sim::cosim
