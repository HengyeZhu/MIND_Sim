#include "micro/sim/core_neuron_data.hpp"

namespace mind_sim::micro::sim {

CoreNeuronData::~CoreNeuronData() {
    release_core_neuron_device_runtime(*this);
}

}  // namespace mind_sim::micro::sim
