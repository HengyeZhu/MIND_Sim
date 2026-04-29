/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

// PatternStim is a nrnivmodl-generated mechanism. The CoreNEURON core keeps the
// file-mode entry points, but it does not link against pattern.mod symbols.

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/utils/nrnoc_aux.hpp"

namespace coreneuron {

int nrn_extra_thread0_vdata;

void nrn_set_extra_thread0_vdata() {
    const int type = nrn_get_mechtype("PatternStim");
    if (!corenrn.get_memb_func(type).initialize) {
        hoc_execerror("PatternStim requires a loaded nrnivmodl CoreNEURON mechanism library", nullptr);
    }
    nrn_extra_thread0_vdata = corenrn.get_prop_dparam_size()[type];
}

void nrn_mkPatternStim(const char* filename, double tstop) {
    (void) filename;
    (void) tstop;
    hoc_execerror("PatternStim raster injection must come from nrnivmodl mechanism code", nullptr);
}

}  // namespace coreneuron
