NEURON {
    POINT_PROCESS tvb_epileptor2d
    RANGE coupled_x, x, z
    RANGE x0, a, b, c, d, r, slope, kvf, ks, tt, i_ext, modification
}

MIND {
    ROLE REGION
    TARGET_INPUT coupled_x
    SOURCE_EXPOSURE x, z
}

PARAMETER {
    x0 = -2.4
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    r = 0.00035
    slope = 0.0
    kvf = 0.35
    ks = 0.0
    tt = 1.0
    i_ext = 3.1
    modification = 0.0
}

ASSIGNED {
    coupled_x
}

STATE {
    x
    z
}

INITIAL {
    x = 0.0
    z = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL fast_term, z_minus, slow_term, h_term

    fast_term = a * x * x + (d - b) * x
    if (x >= 0.0) {
        z_minus = z - 4.0
        fast_term = -slope - 0.6 * z_minus * z_minus + d * x
    }

    slow_term = 0.0
    if (z < 0.0) {
        slow_term = -0.1 * z ^ 7
    }

    h_term = 4.0 * (x - x0) + slow_term
    if (modification > 0.5) {
        h_term = x0 + 3.0 / (1.0 + exp(-(x + 0.5) / 0.1))
    }

    x' = tt * (c - z + i_ext + kvf * coupled_x - fast_term * x)
    z' = tt * r * (h_term - z + ks * coupled_x)
}
