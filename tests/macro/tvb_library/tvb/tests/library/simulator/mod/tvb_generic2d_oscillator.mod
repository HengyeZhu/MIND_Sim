NEURON {
    POINT_PROCESS tvb_generic2d_oscillator
    RANGE c_0, V, W
    RANGE tau, I, a, b, c, d, e, f, g, alpha, beta, gamma
}

MIND {
    ROLE REGION
    EXPOSURE V, W, c_0
}

PARAMETER {
    tau = 1.0
    I = 0.0
    a = -2.0
    b = -10.0
    c = 0.0
    d = 0.02
    e = 3.0
    f = 1.0
    g = 0.0
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
}

ASSIGNED {
    c_0
}

STATE {
    V
    W
}

INITIAL {
    V = 0.0
    W = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    V' = d * tau * (alpha * W - f * V * V * V + e * V * V + g * V + gamma * I + gamma * c_0)
    W' = d * (a + b * V + c * V * V - beta * W) / tau
}
