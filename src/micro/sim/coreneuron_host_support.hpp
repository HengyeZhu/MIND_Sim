#pragma once

namespace coreneuron {

void nrn_configure_embedded_cpu_runtime();
void nrn_configure_embedded_gpu_runtime(int num_gpus, int cell_permute);

}  // namespace coreneuron
