NEURON {
    POINT_PROCESS tvb_sigmoidal_sum1d
    RANGE x, c_raw
    RANGE cmin, cmax, midpoint, a, sigma
}

MIND {
    ROLE REGION
    EXPOSURE x, c_raw
}

PARAMETER {
    cmin = -1.0
    cmax = 1.0
    midpoint = 0.0
    a = 1.0
    sigma = 230.0
}

ASSIGNED {
    c_raw
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
    x' = cmin + ((cmax - cmin) / (1.0 + exp(-a * ((c_raw - midpoint) / sigma))))
}
