NEURON {
    POINT_PROCESS tvb_x_to_c_macro2macro
    RANGE x_source, c, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    WRITE_TARGET c
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    x_source
    c
    weight
    delay
}

BREAKPOINT {
    c = a * weight * x_source
}
