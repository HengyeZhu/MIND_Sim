NEURON {
    POINT_PROCESS tvb_dumont_gutkin
    RANGE r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii, c_r
    RANGE I_e, Delta_e, eta_e, tau_e, I_i, Delta_i, eta_i, tau_i
    RANGE tau_s, J_ee, J_ei, J_ie, J_ii, Gamma
}

MIND {
    ROLE REGION
    EXPOSURE r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii
}

PARAMETER {
    I_e = 0.0
    Delta_e = 1.0
    eta_e = -5.0
    tau_e = 10.0
    I_i = 0.0
    Delta_i = 1.0
    eta_i = -5.0
    tau_i = 10.0
    tau_s = 1.0
    J_ee = 0.0
    J_ei = 10.0
    J_ie = 0.0
    J_ii = 15.0
    Gamma = 5.0
    c_r = 0.0
}

STATE {
    r_e
    V_e
    s_ee
    s_ei
    r_i
    V_i
    s_ie
    s_ii
}

INITIAL {
    r_e = 0.0
    V_e = 0.0
    s_ee = 0.0
    s_ei = 0.0
    r_i = 0.0
    V_i = 0.0
    s_ie = 0.0
    s_ii = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL pi

    pi = 3.14159265358979323846
    r_e' = (Delta_e / (pi * tau_e) + 2.0 * V_e * r_e) / tau_e
    V_e' = (V_e * V_e + eta_e - tau_e * tau_e * pi * pi * r_e * r_e + tau_e * s_ee - tau_e * s_ei + I_e) / tau_e
    s_ee' = (-s_ee + J_ee * r_e + c_r) / tau_s
    s_ei' = (-s_ei + J_ei * r_i) / tau_s
    r_i' = (Delta_i / (pi * tau_i) + 2.0 * V_i * r_i) / tau_i
    V_i' = (V_i * V_i + eta_i - tau_i * tau_i * pi * pi * r_i * r_i + tau_i * s_ie - tau_i * s_ii + I_i) / tau_i
    s_ie' = (-s_ie + J_ie * r_e + Gamma * c_r) / tau_s
    s_ii' = (-s_ii + J_ii * r_i) / tau_s
}
