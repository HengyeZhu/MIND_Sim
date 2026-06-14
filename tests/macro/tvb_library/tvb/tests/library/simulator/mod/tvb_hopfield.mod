NEURON {
    POINT_PROCESS tvb_hopfield
    RANGE c, taux, x, theta
}

MIND {
    ROLE REGION
    EXPOSURE x, theta, c
}

PARAMETER {
    taux = 1.0
}

ASSIGNED {
    c
}

STATE {
    x
    theta
}

INITIAL {
    x = 0.0
    theta = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    x' = (-x + c) / taux
    theta' = (-x + c) / taux
}
