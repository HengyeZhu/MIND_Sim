NEURON {
    POINT_PROCESS tvb_wilson_cowan
    RANGE c_0, E, I
    RANGE c_ee, c_ei, c_ie, c_ii
    RANGE tau_e, tau_i
    RANGE a_e, b_e, c_e, theta_e
    RANGE a_i, b_i, c_i, theta_i
    RANGE r_e, r_i, k_e, k_i
    RANGE P, Q, alpha_e, alpha_i, shift_sigmoid
}

MIND {
    ROLE REGION
    EXPOSURE E, I
}

PARAMETER {
    c_ee = 12.0
    c_ei = 4.0
    c_ie = 13.0
    c_ii = 11.0
    tau_e = 10.0
    tau_i = 10.0
    a_e = 1.2
    b_e = 2.8
    c_e = 1.0
    theta_e = 0.0
    a_i = 1.0
    b_i = 4.0
    c_i = 1.0
    theta_i = 0.0
    r_e = 1.0
    r_i = 1.0
    k_e = 1.0
    k_i = 1.0
    P = 0.0
    Q = 0.0
    alpha_e = 1.0
    alpha_i = 1.0
    shift_sigmoid = 1.0
    c_0 = 0.0
}

STATE {
    E
    I
}

INITIAL {
    E = 0.0
    I = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL x_e, x_i, s_e, s_i, offset_e, offset_i

    x_e = alpha_e * (c_ee * E - c_ei * I + P - theta_e + c_0)
    x_i = alpha_i * (c_ie * E - c_ii * I + Q - theta_i)

    if (shift_sigmoid > 0.5) {
        offset_e = 1.0 / (1.0 + exp(-a_e * (-b_e)))
        offset_i = 1.0 / (1.0 + exp(-a_i * (-b_i)))
        s_e = c_e * (1.0 / (1.0 + exp(-a_e * (x_e - b_e))) - offset_e)
        s_i = c_i * (1.0 / (1.0 + exp(-a_i * (x_i - b_i))) - offset_i)
    } else {
        s_e = c_e / (1.0 + exp(-a_e * (x_e - b_e)))
        s_i = c_i / (1.0 + exp(-a_i * (x_i - b_i)))
    }

    E' = (-E + (k_e - r_e * E) * s_e) / tau_e
    I' = (-I + (k_i - r_i * I) * s_i) / tau_i
}
