NEURON {
    POINT_PROCESS tvb_linear
    RANGE c, gamma, x
}

MIND {
    ROLE REGION
    EXPOSURE x, c
}

PARAMETER {
    gamma = -10.0
}

ASSIGNED {
    c
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
    x' = gamma * x + c
}
