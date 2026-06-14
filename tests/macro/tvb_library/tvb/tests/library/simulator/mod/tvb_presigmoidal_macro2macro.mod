NEURON {
    POINT_PROCESS tvb_presigmoidal_macro2macro
    RANGE x_source, y_source, c_0, weight, delay
    RANGE H, Q, G, P
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source, y AS y_source
    WRITE_TARGET c_0
}

PARAMETER {
    H = 0.5
    Q = 1.0
    G = 60.0
    P = 1.0
}

ASSIGNED {
    x_source
    y_source
    c_0
    weight
    delay
}

BREAKPOINT {
    c_0 = weight * H * (Q + tanh(G * (P * x_source - y_source)))
}
