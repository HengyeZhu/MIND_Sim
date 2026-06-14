NEURON {
    POINT_PROCESS tvb_difference_macro2macro
    RANGE x_source, x_target, c, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    READ_TARGET x AS x_target
    WRITE_TARGET c
}

PARAMETER {
    a = 0.1
}

ASSIGNED {
    x_source
    x_target
    c
    weight
    delay
}

BREAKPOINT {
    c = a * weight * (x_source - x_target)
}
