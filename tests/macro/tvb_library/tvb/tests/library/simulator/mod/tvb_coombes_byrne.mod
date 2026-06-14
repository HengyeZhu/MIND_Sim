NEURON {
    POINT_PROCESS tvb_coombes_byrne
    RANGE r, V, g, q, c_r
    RANGE Delta, alpha, v_syn, k, eta
}

MIND {
    ROLE REGION
    EXPOSURE r, V, g, q
}

PARAMETER {
    Delta = 0.5
    alpha = 0.95
    v_syn = -10.0
    k = 1.0
    eta = 20.0
    c_r = 0.0
}

STATE {
    r
    V
    g
    q
}

INITIAL {
    r = 0.0
    V = 0.0
    g = 0.0
    q = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL pi

    pi = 3.14159265358979323846
    r' = Delta / pi + 2.0 * V * r - g * r
    V' = V * V - pi * pi * r * r + eta + (v_syn - V) * g + c_r
    g' = alpha * q
    q' = alpha * (k * pi * r - g - 2.0 * q)
}
