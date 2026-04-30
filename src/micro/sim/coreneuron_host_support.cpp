#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/sim/multicore.hpp"

namespace coreneuron {

void nrn_configure_embedded_cpu_runtime() {
    corenrn_param.verbose = corenrn_parameters_data::NONE;
    corenrn_param.gpu = false;
    corenrn_param.num_gpus = 0;
    corenrn_param.threading = false;
    corenrn_param.binqueue = false;
    nrn_use_bin_queue_ = false;
    corenrn_param.cell_interleave_permute = 1;
    interleave_permute_type = 1;
    cellorder_nwarp = static_cast<int>(corenrn_param.nwarp);
    nrn_nobanner_ = 1;
    use_solve_interleave = true;
}

void nrn_configure_embedded_gpu_runtime(int num_gpus, int cell_permute) {
    corenrn_param.verbose = corenrn_parameters_data::NONE;
    corenrn_param.gpu = true;
    corenrn_param.num_gpus = num_gpus > 0 ? static_cast<unsigned>(num_gpus) : 1U;
    corenrn_param.threading = false;
    corenrn_param.binqueue = false;
    corenrn_param.cuda_interface = false;
    nrn_use_bin_queue_ = false;
    corenrn_param.cell_interleave_permute =
        cell_permute > 0 ? static_cast<unsigned>(cell_permute) : 2U;
    interleave_permute_type = static_cast<int>(corenrn_param.cell_interleave_permute);
    cellorder_nwarp = static_cast<int>(corenrn_param.nwarp);
    nrn_nobanner_ = 1;
    use_solve_interleave = true;
}

}  // namespace coreneuron
