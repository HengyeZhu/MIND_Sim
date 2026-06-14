NEURON {
    POINT_PROCESS tvb_reduced_wong_wang
    RANGE c_0, S
    RANGE a, b, d, gamma, tau_s, w, J_N, I_o, sigma_noise
}

MIND {
    ROLE REGION
    EXPOSURE S
}

PARAMETER {
    a = 0.270
    b = 0.108
    d = 154.0
    gamma = 0.641
    tau_s = 100.0
    w = 0.6
    J_N = 0.2609
    I_o = 0.33
    sigma_noise = 1e-9
    c_0 = 0.0
}

STATE {
    S
}

INITIAL {
    S = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL x, H

    x = w * J_N * S + I_o + J_N * c_0
    H = (a * x - b) / (1.0 - exp(-d * (a * x - b)))
    S' = -S / tau_s + (1.0 - S) * H * gamma
}
