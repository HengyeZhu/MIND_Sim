NEURON {
    POINT_PROCESS tvb_kuramoto
    RANGE c, omega, theta
}

MIND {
    ROLE REGION
    EXPOSURE theta, c
}

PARAMETER {
    omega = 1.0
}

ASSIGNED {
    c
}

STATE {
    theta
}

INITIAL {
    theta = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    theta' = omega + c
}
