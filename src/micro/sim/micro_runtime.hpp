#pragma once

#include "micro/sim/core_neuron_data.hpp"
#include "micro/sim/types.hpp"

#include <vector>

namespace mind_sim::micro::sim {

class MicroRuntime {
  public:
    explicit MicroRuntime(CoreNeuronData& core_data);

    void finitialize(double voltage);

    [[nodiscard]] mind_sim::micro::sim::MicroEventTable& scheduled_events() noexcept;

    mind_sim::micro::sim::MicroSpikeTableView advance_window(double start_time, double stop_time);

  private:
    void bind_core_globals();
    void ensure_core_globals_bound();
    void ensure_registered_mechanisms();
    void require_registered_mechanisms() const;
    void initialize_private_mechanism_storage();
    void ensure_private_mechanism_storage_initialized();
    void require_private_mechanism_storage_initialized() const;

    CoreNeuronData* core_data_{nullptr};
    mind_sim::micro::sim::MicroEventTable scheduled_events_{};
    bool core_globals_bound_{false};
    bool mechanisms_checked_{false};
    bool private_mechanism_storage_checked_{false};
};

}  // namespace mind_sim::micro::sim
