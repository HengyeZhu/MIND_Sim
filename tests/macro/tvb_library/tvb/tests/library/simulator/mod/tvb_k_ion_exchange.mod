NEURON {
    POINT_PROCESS tvb_k_ion_exchange
    RANGE x, V, n, DKi, Kg, c_0
    RANGE E, K_bath, J, eta, Delta, c_minus, R_minus, c_plus, R_plus, Vstar
    RANGE Cm, tau_n, gamma, epsilon
}

MIND {
    ROLE REGION
    EXPOSURE x, V, n, DKi, Kg
}

PARAMETER {
    E = 0.0
    K_bath = 5.5
    J = 0.1
    eta = 0.0
    Delta = 1.0
    c_minus = -40.0
    R_minus = 0.5
    c_plus = -20.0
    R_plus = -0.5
    Vstar = -31.0
    Cm = 1.0
    tau_n = 4.0
    gamma = 0.04
    epsilon = 0.001
    c_0 = 0.0
}

STATE {
    x
    V
    n
    DKi
    Kg
}

INITIAL {
    x = 0.0
    V = 0.0
    n = 0.0
    DKi = 0.0
    Kg = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL pi, Cnap, DCnap, Ckp, DCkp, Cmna, DCmna, Cnk, DCnk, g_Cl, g_Na, g_K, g_Nal, g_Kl, rho, w_i, w_o, Na_i0, Na_o0, K_i0, K_o0, Cl_i0, Cl_o0, beta, DNa_i, DNa_o, DK_o, K_i, Na_i, Na_o, K_o, minf, ninf, hgate, I_K, I_Na, I_Cl, I_pump, rate, Vdot, R, c

    pi = 3.14159265358979323846
    Cnap = 21.0
    DCnap = 2.0
    Ckp = 5.5
    DCkp = 1.0
    Cmna = -24.0
    DCmna = 12.0
    Cnk = -19.0
    DCnk = 18.0
    g_Cl = 7.5
    g_Na = 40.0
    g_K = 22.0
    g_Nal = 0.02
    g_Kl = 0.12
    rho = 250.0
    w_i = 2160.0
    w_o = 720.0
    Na_i0 = 16.0
    Na_o0 = 138.0
    K_i0 = 130.0
    K_o0 = 4.8
    Cl_i0 = 5.0
    Cl_o0 = 112.0

    beta = w_i / w_o
    DNa_i = -DKi
    DNa_o = -beta * DNa_i
    DK_o = -beta * DKi
    K_i = K_i0 + DKi
    Na_i = Na_i0 + DNa_i
    Na_o = Na_o0 + DNa_o
    K_o = K_o0 + DK_o + Kg

    minf = 1.0 / (1.0 + exp((Cmna - V) / DCmna))
    ninf = 1.0 / (1.0 + exp((Cnk - V) / DCnk))
    hgate = 1.1 - 1.0 / (1.0 + exp(-8.0 * (n - 0.4)))
    I_K = (g_Kl + g_K * n) * (V - 26.64 * log(K_o / K_i))
    I_Na = (g_Nal + g_Na * minf * hgate) * (V - 26.64 * log(Na_o / Na_i))
    I_Cl = g_Cl * (V + 26.64 * log(Cl_o0 / Cl_i0))
    I_pump = rho * (1.0 / (1.0 + exp((Cnap - Na_i) / DCnap))) * (1.0 / (1.0 + exp((Ckp - K_o) / DCkp)))
    rate = R_minus * x / pi
    Vdot = (-1.0 / Cm) * (I_Na + I_K + I_Cl + I_pump)

    if (V <= Vstar) {
        R = R_minus
        c = c_minus
    } else {
        R = R_plus
        c = c_plus
    }

    x' = Delta + 2.0 * R * (V - c) * x - J * rate * x
    V' = Vdot - R * x * x + eta + (R_minus / pi) * c_0 * (E - V)
    n' = (ninf - n) / tau_n
    DKi' = -(gamma / w_i) * (I_K - 2.0 * I_pump)
    Kg' = epsilon * (K_bath - K_o)
}
