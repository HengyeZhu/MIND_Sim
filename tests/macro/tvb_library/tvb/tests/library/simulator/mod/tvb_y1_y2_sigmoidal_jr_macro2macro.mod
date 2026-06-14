NEURON {
    POINT_PROCESS tvb_y1_y2_sigmoidal_jr_macro2macro
    RANGE y1_source, y2_source, c, weight, delay
    RANGE cmin, cmax, midpoint, r, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE y1 AS y1_source, y2 AS y2_source
    WRITE_TARGET c
}

PARAMETER {
    cmin = 0.0
    cmax = 0.005
    midpoint = 6.0
    r = 0.56
    a = 1.0
}

ASSIGNED {
    y1_source
    y2_source
    c
    weight
    delay
}

BREAKPOINT {
    c = weight * a * (cmin + (cmax - cmin) / (1.0 + exp(r * (midpoint - (y1_source - y2_source)))))
}
