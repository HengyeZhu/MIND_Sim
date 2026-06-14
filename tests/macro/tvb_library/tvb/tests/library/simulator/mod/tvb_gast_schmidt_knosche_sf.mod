NEURON {
    POINT_PROCESS tvb_gast_schmidt_knosche_sf
    RANGE r, V, A, B, c_r, c_V
    RANGE tau, tau_A, alpha, I, Delta, J, eta, cr, cv
}

MIND {
    ROLE REGION
    EXPOSURE r, V, A, B
}

PARAMETER {
    tau = 1.0
    tau_A = 10.0
    alpha = 10.0
    I = 0.0
    Delta = 2.0
    J = 21.2132
    eta = 1.0
    cr = 1.0
    cv = 0.0
    c_r = 0.0
    c_V = 0.0
}

STATE {
    r
    V
    A
    B
}

INITIAL {
    r = 0.0
    V = 0.0
    A = 0.0
    B = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL pi

    pi = 3.14159265358979323846
    r' = (Delta / (pi * tau) + 2.0 * V * r) / tau
    V' = (V * V - pi * pi * tau * tau * r * r + eta + J * tau * r + I - A + cr * c_r + cv * c_V) / tau
    A' = B / tau_A
    B' = (-2.0 * B - A + alpha * r) / tau_A
}
