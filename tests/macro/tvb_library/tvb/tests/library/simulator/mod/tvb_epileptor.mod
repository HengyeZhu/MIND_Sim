NEURON {
    POINT_PROCESS tvb_epileptor
    RANGE x1, y1, z, x2, y2, g, c_pop1, c_pop2
    RANGE x0, Iext, Iext2, a, b, slope, tt, Kvf
    RANGE c, d, r, Ks, Kf, aa, bb, tau, modification
}

MIND {
    ROLE REGION
    EXPOSURE x1, y1, z, x2, y2, g
}

PARAMETER {
    x0 = -1.6
    Iext = 3.1
    Iext2 = 0.45
    a = 1.0
    b = 3.0
    slope = 0.0
    tt = 1.0
    Kvf = 0.0
    c = 1.0
    d = 5.0
    r = 0.00035
    Ks = 0.0
    Kf = 0.0
    aa = 6.0
    bb = 2.0
    tau = 10.0
    modification = 0.0
    c_pop1 = 0.0
    c_pop2 = 0.0
}

STATE {
    x1
    y1
    z
    x2
    y2
    g
}

INITIAL {
    x1 = 0.0
    y1 = 0.0
    z = 0.0
    x2 = 0.0
    y2 = 0.0
    g = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL fast_x1, h, f2

    if (x1 < 0.0) {
        fast_x1 = -a * x1 * x1 + b * x1
    } else {
        fast_x1 = slope - x2 + 0.6 * (z - 4.0) * (z - 4.0)
    }

    if (modification > 0.5) {
        h = x0 + 3.0 / (1.0 + exp(-(x1 + 0.5) / 0.1))
    } else {
        h = 4.0 * (x1 - x0)
        if (z < 0.0) {
            h = h - 0.1 * z ^ 7
        }
    }

    if (x2 < -0.25) {
        f2 = 0.0
    } else {
        f2 = aa * (x2 + 0.25)
    }

    x1' = tt * (y1 - z + Iext + Kvf * c_pop1 + fast_x1 * x1)
    y1' = tt * (c - d * x1 * x1 - y1)
    z' = tt * r * (h - z + Ks * c_pop1)
    x2' = tt * (-y2 + x2 - x2 * x2 * x2 + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2)
    y2' = tt * ((-y2 + f2) / tau)
    g' = tt * (-0.01 * (g - 0.1 * x1))
}
