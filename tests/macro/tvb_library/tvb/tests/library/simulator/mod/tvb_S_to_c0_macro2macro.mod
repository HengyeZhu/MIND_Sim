NEURON {
    POINT_PROCESS tvb_S_to_c0_macro2macro
    RANGE S_source, c_0, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE S AS S_source
    WRITE_TARGET c_0
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    S_source
    c_0
    weight
    delay
}

BREAKPOINT {
    c_0 = a * weight * S_source
}
