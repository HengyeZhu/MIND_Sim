NEURON {
    POINT_PROCESS tvb_jansen_rit
    RANGE c_0, y0, y1, y2, y3, y4, y5
    RANGE A, B, a, b, v0, nu_max, r, J
    RANGE a_1, a_2, a_3, a_4, p_min, p_max, mu
}

MIND {
    ROLE REGION
    EXPOSURE y0, y1, y2, y3, y4, y5
}

PARAMETER {
    A = 3.25
    B = 22.0
    a = 0.1
    b = 0.05
    v0 = 5.52
    nu_max = 0.0025
    r = 0.56
    J = 135.0
    a_1 = 1.0
    a_2 = 0.8
    a_3 = 0.25
    a_4 = 0.25
    p_min = 0.12
    p_max = 0.32
    mu = 0.22
    c_0 = 0.0
}

STATE {
    y0
    y1
    y2
    y3
    y4
    y5
}

INITIAL {
    y0 = 0.0
    y1 = 0.0
    y2 = 0.0
    y3 = 0.0
    y4 = 0.0
    y5 = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL sigm_y1_y2, sigm_y0_1, sigm_y0_3

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (y1 - y2))))
    sigm_y0_1 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_1 * J * y0))))
    sigm_y0_3 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_3 * J * y0))))

    y0' = y3
    y1' = y4
    y2' = y5
    y3' = A * a * sigm_y1_y2 - 2.0 * a * y3 - a * a * y0
    y4' = A * a * (mu + a_2 * J * sigm_y0_1 + c_0) - 2.0 * a * y4 - a * a * y1
    y5' = B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b * b * y2
}
