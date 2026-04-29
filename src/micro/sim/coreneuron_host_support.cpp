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
    nrn_nobanner_ = 1;
    use_solve_interleave = true;
}

}  // namespace coreneuron
