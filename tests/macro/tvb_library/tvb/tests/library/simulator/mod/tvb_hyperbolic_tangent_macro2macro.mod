NEURON {
    POINT_PROCESS tvb_hyperbolic_tangent_macro2macro
    RANGE x_source, c, weight, delay
    RANGE a, b, midpoint, sigma
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    WRITE_TARGET c
}

PARAMETER {
    a = 1.0
    b = 1.0
    midpoint = 0.0
    sigma = 1.0
}

ASSIGNED {
    x_source
    c
    weight
    delay
}

BREAKPOINT {
    c = weight * a * (1.0 + tanh((b * x_source - midpoint) / sigma))
}
