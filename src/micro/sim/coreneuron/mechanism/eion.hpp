/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

/// THIS FILE IS AUTO GENERATED DONT MODIFY IT.

#pragma once

namespace coreneuron {

extern int nrn_is_ion(int);
extern void ion_reg(const char*, double);
extern int ion_register(const char*, double);
extern double nrn_ion_charge(int);
extern void nrn_verify_ion_charge_defined();

}  // namespace coreneuron
