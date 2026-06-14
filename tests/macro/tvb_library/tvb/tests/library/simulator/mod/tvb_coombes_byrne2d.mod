NEURON {
    POINT_PROCESS tvb_coombes_byrne2d
    RANGE r, V, c_r
    RANGE Delta, v_syn, k, eta
}

MIND {
    ROLE REGION
    EXPOSURE r, V
}

PARAMETER {
    Delta = 1.0
    v_syn = -4.0
    k = 1.0
    eta = 2.0
    c_r = 0.0
}

STATE {
    r
    V
}

INITIAL {
    r = 0.0
    V = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL pi

    pi = 3.14159265358979323846
    r' = Delta / pi + 2.0 * V * r - k * pi * r * r
    V' = V * V - pi * pi * r * r + eta + (v_syn - V) * k * pi * r + c_r
}
