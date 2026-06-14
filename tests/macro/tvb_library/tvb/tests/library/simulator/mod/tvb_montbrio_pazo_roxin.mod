NEURON {
    POINT_PROCESS tvb_montbrio_pazo_roxin
    RANGE r, V, c_r, c_V
    RANGE tau, I, Delta, J, eta, Gamma, cr, cv
}

MIND {
    ROLE REGION
    EXPOSURE r, V
}

PARAMETER {
    tau = 1.0
    I = 0.0
    Delta = 1.0
    J = 15.0
    eta = -5.0
    Gamma = 0.0
    cr = 1.0
    cv = 0.0
    c_r = 0.0
    c_V = 0.0
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
    r' = (Delta / (pi * tau) + 2.0 * V * r) / tau
    V' = (V * V - pi * pi * tau * tau * r * r + eta + J * tau * r + I + cr * c_r + cv * c_V) / tau
}
