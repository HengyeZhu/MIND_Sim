NEURON {
    POINT_PROCESS tvb_sum1d
    RANGE coupled_x, x
}

MIND {
    ROLE REGION
    EXPOSURE x, coupled_x
}

ASSIGNED {
    coupled_x
}

STATE {
    x
}

INITIAL {
    x = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    x' = coupled_x
}
