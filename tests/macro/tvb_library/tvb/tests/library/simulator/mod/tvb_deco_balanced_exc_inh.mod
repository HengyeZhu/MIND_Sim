NEURON {
    POINT_PROCESS tvb_deco_balanced_exc_inh
    RANGE S_e, S_i, c_0
    RANGE M_i, a_e, b_e, d_e, gamma_e, tau_e, w_p, W_e, J_N
    RANGE a_i, b_i, d_i, gamma_i, tau_i, W_i, J_i
    RANGE G, lamda, I_o, I_ext
}

MIND {
    ROLE REGION
    EXPOSURE S_e, S_i
}

PARAMETER {
    M_i = 1.0
    a_e = 310.0
    b_e = 125.0
    d_e = 0.160
    gamma_e = 0.000641
    tau_e = 100.0
    w_p = 1.4
    W_e = 1.0
    J_N = 0.15
    a_i = 615.0
    b_i = 177.0
    d_i = 0.087
    gamma_i = 0.001
    tau_i = 10.0
    W_i = 0.7
    J_i = 1.0
    G = 2.0
    lamda = 0.0
    I_o = 0.382
    I_ext = 0.0
    c_0 = 0.0
}

STATE {
    S_e
    S_i
}

INITIAL {
    S_e = 0.0
    S_i = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL coupling, J_N_S_e, inh, I_e, x_e, H_e, I_i, x_i, H_i

    coupling = G * J_N * c_0
    J_N_S_e = J_N * S_e
    inh = J_i * S_i

    I_e = W_e * I_o + w_p * J_N_S_e + coupling - inh + I_ext
    x_e = (a_e * I_e - b_e) * M_i
    H_e = x_e / (1.0 - exp(-d_e * x_e))
    S_e' = -(S_e / tau_e) + (1.0 - S_e) * H_e * gamma_e

    I_i = W_i * I_o + J_N_S_e - S_i + lamda * coupling
    x_i = (a_i * I_i - b_i) * M_i
    H_i = x_i / (1.0 - exp(-d_i * x_i))
    S_i' = -(S_i / tau_i) + H_i * gamma_i
}
