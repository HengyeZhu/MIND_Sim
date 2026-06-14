NEURON {
    POINT_PROCESS tvb_E_to_c0_macro2macro
    RANGE E_source, c_0, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE E AS E_source
    WRITE_TARGET c_0
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    E_source
    c_0
    weight
    delay
}

BREAKPOINT {
    c_0 = a * weight * E_source
}
