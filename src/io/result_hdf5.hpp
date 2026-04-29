#pragma once

#include "cosim/types.hpp"
#include "macro/sim/types.hpp"
#include "micro/sim/types.hpp"

#include <string>
#include <vector>

namespace mind_sim::io {

void save_macro_result_h5(const mind_sim::macro::sim::MacroSimulationResult& result,
                          const std::string& path,
                          const std::vector<std::string>& exposure_names,
                          const std::vector<std::string>& roi_labels,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata);

void save_cosim_result_h5(const mind_sim::cosim::SimulationResult& result,
                          const std::string& path,
                          const std::vector<std::string>& exposure_names,
                          const std::vector<std::string>& roi_labels,
                          int spike_roi,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata);

void save_micro_spikes_h5(const mind_sim::micro::sim::MicroSpikeTable& spikes,
                          const std::string& path,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata);

void save_vector_h5(const std::vector<double>& values,
                    const std::string& path,
                    const std::string& dataset_name);

}  // namespace mind_sim::io
