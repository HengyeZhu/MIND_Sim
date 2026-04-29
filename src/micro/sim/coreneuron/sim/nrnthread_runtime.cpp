/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#include <cstdlib>
#include <vector>

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/nrnconf.h"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/utils/memory.h"
#include "coreneuron/utils/nrnoc_aux.hpp"

namespace coreneuron {

CoreNeuron corenrn;
int nrn_nthread = 0;
NrnThread* nrn_threads = nullptr;
void (*nrn_mk_transfer_thread_data_)();

static std::vector<NrnThreadMembList*> table_check;

NrnThreadMembList* create_tml(NrnThread& nt,
                              int mech_id,
                              Memb_func& memb_func,
                              int& shadow_rhs_cnt,
                              const std::vector<int>& mech_types,
                              const std::vector<int>& nodecounts) {
    auto* tml = static_cast<NrnThreadMembList*>(emalloc_align(sizeof(NrnThreadMembList), 0));
    tml->next = nullptr;
    tml->index = mech_types[mech_id];
    tml->dependencies = nullptr;
    tml->ndependencies = 0;

    tml->ml = static_cast<Memb_list*>(ecalloc_align(1, sizeof(Memb_list), 0));
    tml->ml->_net_receive_buffer = nullptr;
    tml->ml->_net_send_buffer = nullptr;
    tml->ml->_permute = nullptr;
    if (memb_func.alloc == nullptr) {
        hoc_execerror(memb_func.sym, "mechanism does not exist");
    }
    tml->ml->nodecount = nodecounts[mech_id];
    if (!memb_func.sym) {
        printf("%s (type %d) is not available\n", nrn_get_mechname(tml->index), tml->index);
        exit(1);
    }
    tml->ml->_nodecount_padded = nrn_soa_padded_size(tml->ml->nodecount,
                                                     corenrn.get_mech_data_layout()[tml->index]);
    if (memb_func.is_point && corenrn.get_is_artificial()[tml->index] == 0 &&
        tml->ml->nodecount > shadow_rhs_cnt) {
        shadow_rhs_cnt = tml->ml->nodecount;
    }

    if (auto* const priv_ctor = corenrn.get_memb_func(tml->index).private_constructor) {
        priv_ctor(&nt, tml->ml, tml->index);
    }

    return tml;
}

void nrn_threads_create() {
    nrn_threads_free();
    nrn_nthread = 1;
    nrn_threads = new NrnThread[1];
    NrnThread& nt = *nrn_threads;
    nt.id = 0;
    for (int j = 0; j < BEFORE_AFTER_SIZE; ++j) {
        nt.tbl[j] = nullptr;
    }
    v_structure_change = 1;
    diam_changed = 1;
}

void nrn_threads_free() {
    delete[] nrn_threads;
    nrn_threads = nullptr;
    nrn_nthread = 0;
    table_check.clear();
}

void nrn_mk_table_check() {
    table_check.clear();
    auto& memb_func = corenrn.get_memb_funcs();
    std::vector<char> seen(memb_func.size(), 0);
    NrnThread& nt = *nrn_threads;
    for (auto* tml = nt.tml; tml; tml = tml->next) {
        const int type = tml->index;
        if (memb_func[type].thread_table_check_ != nullptr && !seen[static_cast<std::size_t>(type)]) {
            table_check.push_back(tml);
            seen[static_cast<std::size_t>(type)] = 1;
        }
    }
}

void nrn_thread_table_check() {
    for (auto* tml : table_check) {
        Memb_list* ml = tml->ml;
        (*corenrn.get_memb_func(tml->index).thread_table_check_)(
            0,
            ml->_nodecount_padded,
            ml->data,
            ml->pdata,
            ml->_thread,
            nrn_threads,
            ml,
            tml->index);
    }
}

}  // namespace coreneuron
