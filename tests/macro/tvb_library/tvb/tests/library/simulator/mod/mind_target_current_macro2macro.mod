NEURON {
    POINT_PROCESS mind_target_current_macro2macro
    RANGE x_source, c, c_target, weight
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    READ_TARGET c AS c_target
    WRITE_TARGET c
}

ASSIGNED {
    x_source
    c
    c_target
    weight
}

BREAKPOINT {
    c = c_target + weight * x_source
}
