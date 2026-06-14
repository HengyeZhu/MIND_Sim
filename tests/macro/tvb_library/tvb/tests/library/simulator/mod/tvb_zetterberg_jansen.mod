NEURON {
    POINT_PROCESS tvb_zetterberg_jansen
    RANGE v1, y1, v2, y2, v3, y3, v4, y4, v5, y5, v6, v7, c_0
    RANGE He, Hi, ke, ki, e0, rho_2, rho_1
    RANGE gamma_1, gamma_2, gamma_3, gamma_4, gamma_5
    RANGE gamma_1T, gamma_2T, gamma_3T, P, U, Q
}

MIND {
    ROLE REGION
    EXPOSURE v1, y1, v2, y2, v3, y3, v4, y4, v5, y5, v6, v7
}

PARAMETER {
    He = 3.25
    Hi = 22.0
    ke = 0.1
    ki = 0.05
    e0 = 0.0025
    rho_2 = 6.0
    rho_1 = 0.56
    gamma_1 = 135.0
    gamma_2 = 108.0
    gamma_3 = 33.75
    gamma_4 = 33.75
    gamma_5 = 15.0
    gamma_1T = 1.0
    gamma_2T = 1.0
    gamma_3T = 1.0
    P = 0.12
    U = 0.12
    Q = 0.12
    c_0 = 0.0
}

STATE {
    v1
    y1
    v2
    y2
    v3
    y3
    v4
    y4
    v5
    y5
    v6
    v7
}

INITIAL {
    v1 = 0.0
    y1 = 0.0
    v2 = 0.0
    y2 = 0.0
    v3 = 0.0
    y3 = 0.0
    v4 = 0.0
    y4 = 0.0
    v5 = 0.0
    y5 = 0.0
    v6 = 0.0
    v7 = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL Heke, Hiki, ke_2, ki_2, keke, kiki, sig_v2_v3, sig_v1, sig_v4_v5, coupled_input

    Heke = He * ke
    Hiki = Hi * ki
    ke_2 = 2.0 * ke
    ki_2 = 2.0 * ki
    keke = ke * ke
    kiki = ki * ki

    coupled_input = (2.0 * e0) / (1.0 + exp(rho_1 * (rho_2 - c_0)))
    sig_v2_v3 = (2.0 * e0) / (1.0 + exp(rho_1 * (rho_2 - (v2 - v3))))
    sig_v1 = (2.0 * e0) / (1.0 + exp(rho_1 * (rho_2 - v1)))
    sig_v4_v5 = (2.0 * e0) / (1.0 + exp(rho_1 * (rho_2 - (v4 - v5))))

    v1' = y1
    y1' = Heke * (gamma_1 * sig_v2_v3 + gamma_1T * (U + coupled_input)) - ke_2 * y1 - keke * v1
    v2' = y2
    y2' = Heke * (gamma_2 * sig_v1 + gamma_2T * (P + coupled_input)) - ke_2 * y2 - keke * v2
    v3' = y3
    y3' = Hiki * (gamma_4 * sig_v4_v5) - ki_2 * y3 - kiki * v3
    v4' = y4
    y4' = Heke * (gamma_3 * sig_v2_v3 + gamma_3T * (Q + coupled_input)) - ke_2 * y4 - keke * v4
    v5' = y5
    y5' = Hiki * (gamma_5 * sig_v4_v5) - ki_2 * y5 - keke * v5
    v6' = y2 - y3
    v7' = y4 - y5
}
