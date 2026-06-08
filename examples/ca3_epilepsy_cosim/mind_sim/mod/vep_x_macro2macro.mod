NEURON {
    POINT_PROCESS vep_x_macro2macro
    RANGE x, coupled_x, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    SOURCE_EXPOSURE x
    TARGET_INPUT coupled_x
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    x
    coupled_x
    weight
    delay
}

BREAKPOINT {
    coupled_x = coupled_x + a * weight * x
}
