/*
# =============================================================================
# Copyright (c) 2016 - 2022 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#include "coreneuron/apps/corenrn_parameters.hpp"

namespace CLI {
struct App {};
}  // namespace CLI

namespace coreneuron {

corenrn_parameters::corenrn_parameters() = default;
corenrn_parameters::~corenrn_parameters() = default;

void corenrn_parameters::parse(int argc, char** argv) {
    (void) argc;
    (void) argv;
}

void corenrn_parameters::reset() {
    static_cast<corenrn_parameters_data&>(*this) = corenrn_parameters_data{};
}

std::string corenrn_parameters::config_to_str(bool default_also, bool write_description) const {
    (void) default_also;
    (void) write_description;
    return {};
}

std::ostream& operator<<(std::ostream& os, const corenrn_parameters& params) {
    return os << "--mpi=" << (params.mpi_enable ? "true" : "false") << '\n'
              << "--gpu=" << (params.gpu ? "true" : "false") << '\n'
              << "--dt=" << params.dt << '\n'
              << "--tstop=" << params.tstop << '\n'
              << "--celsius=" << params.celsius;
}

corenrn_parameters corenrn_param;
int nrn_nobanner_ = 1;

}  // namespace coreneuron
