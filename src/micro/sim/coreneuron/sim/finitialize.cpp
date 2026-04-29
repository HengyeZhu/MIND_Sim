/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#include "coreneuron/nrnconf.h"
#include "coreneuron/network/netpar.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/sim/fast_imem.hpp"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/utils/profile/profiler_interface.h"
#include "coreneuron/coreneuron.hpp"

namespace coreneuron {

bool _nrn_skip_initmodel;

void allocate_data_in_mechanism_nrn_init() {
    // In case some nrn_init allocates data that we need. In this case
    // we want to call nrn_init but not execute initmodel i.e. INITIAL
    // block. For this, set _nrn_skip_initmodel to True temporarily
    // , execute nrn_init and return.
    _nrn_skip_initmodel = true;
    NrnThread* nt = nrn_threads;
    for (NrnThreadMembList* tml = nt->tml; tml; tml = tml->next) {
        Memb_list* ml = tml->ml;
        mod_f_t s = corenrn.get_memb_func(tml->index).initialize;
        if (s) {
            (*s)(nt, ml, tml->index);
        }
    }
    _nrn_skip_initmodel = false;
}

void nrn_finitialize(int setv, double v) {
    Instrumentor::phase_begin("finitialize");
    NrnThread* nt = nrn_threads;
    t = 0.;
    dt2thread(-1.);
    nrn_thread_table_check();
    clear_event_queue();
    nrn_spike_exchange_init();
#if VECTORIZE
    nrn_play_init(); /* Vector.play */
                     /// Play events should be executed before initializing events
    nrn_deliver_events(nt); /* The play events at t=0 */
    if (setv) {
        double* vec_v = nt->_actual_v;
        nrn_pragma_acc(parallel loop present(nt [0:1], vec_v [0:nt->end]) if (nt->compute_gpu))
        nrn_pragma_omp(target teams distribute parallel for simd if(nt->compute_gpu))
        for (int i = 0; i < nt->end; ++i) {
            vec_v[i] = v;
        }
    }

    if (nrn_have_gaps) {
        Instrumentor::phase p("gap-v-transfer");
        nrnmpi_v_transfer();
        nrnthread_v_transfer(nt);
    }

    nrn_ba(nt, BEFORE_INITIAL);
    /* the INITIAL blocks are ordered so that mechanisms that write
       concentrations are after ions and before mechanisms that read
       concentrations.
    */
    /* the memblist list in NrnThread is already so ordered */
    for (auto tml = nt->tml; tml; tml = tml->next) {
        mod_f_t s = corenrn.get_memb_func(tml->index).initialize;
        if (s) {
            (*s)(nt, tml->ml, tml->index);
        }
    }
#endif

    init_net_events();
    nrn_ba(nt, AFTER_INITIAL);
    nrn_deliver_events(nt); /* The INITIAL sent events at t=0 */
    setup_tree_matrix_minimal(nt);
    if (nrn_use_fast_imem) {
        nrn_calc_fast_imem_init(nt);
    }
    nrn_ba(nt, BEFORE_STEP);
    nrncore2nrn_send_init();
    nrncore2nrn_send_values(nt);
    // Consistent with NEURON. BEFORE_STEP and fixed_record_continuous before nrn_deliver_events.
    nrn_deliver_events(nt); /* The record events at t=0 */
#if NRNMPI
    nrn_spike_exchange(nt);
#endif
    Instrumentor::phase_end("finitialize");
}
}  // namespace coreneuron
