NEURON {
    POINT_PROCESS vep_x_macro2macro
    RANGE x_source, coupled_x, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    WRITE_TARGET coupled_x
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    x_source
    coupled_x
    weight
    delay
}

BREAKPOINT {
    coupled_x = a * weight * x_source
}
