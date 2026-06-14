NEURON {
    POINT_PROCESS tvb_presigmoidal_sum2d
    RANGE x, y, c_0
    RANGE H, Q, G, P
}

MIND {
    ROLE REGION
    EXPOSURE x, y, c_0
}

PARAMETER {
    H = 0.5
    Q = 1.0
    G = 60.0
    P = 1.0
}

ASSIGNED {
    c_0
}

STATE {
    x
    y
}

INITIAL {
    x = 0.0
    y = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    x' = c_0
    y' = H * (Q + tanh(G * (P * x - y)))
}
