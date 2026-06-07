NEURON {
    POINT_PROCESS ca3_input_macro2macro
    RANGE x, ca3_input, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    SOURCE_EXPOSURE x
    TARGET_INPUT ca3_input
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    x
    ca3_input
    weight
    delay
}

BREAKPOINT {
    ca3_input = ca3_input + a * weight * x
}
