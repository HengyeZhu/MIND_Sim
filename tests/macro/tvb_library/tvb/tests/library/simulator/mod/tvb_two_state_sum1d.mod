NEURON {
    POINT_PROCESS tvb_two_state_sum1d
    RANGE y1, y2, x, c
}

MIND {
    ROLE REGION
    EXPOSURE y1, y2, x, c
}

ASSIGNED {
    c
}

STATE {
    y1
    y2
    x
}

INITIAL {
    y1 = 0.0
    y2 = 0.0
    x = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    y1' = 0.0
    y2' = 0.0
    x' = c
}
