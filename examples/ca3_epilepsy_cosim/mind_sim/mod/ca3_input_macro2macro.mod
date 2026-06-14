NEURON {
    POINT_PROCESS ca3_input_macro2macro
    RANGE x_source, ca3_input, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    WRITE_TARGET ca3_input
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    x_source
    ca3_input
    weight
    delay
}

BREAKPOINT {
    ca3_input = a * weight * x_source
}
