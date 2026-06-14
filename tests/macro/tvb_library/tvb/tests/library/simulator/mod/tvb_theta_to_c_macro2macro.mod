NEURON {
    POINT_PROCESS tvb_theta_to_c_macro2macro
    RANGE theta_source, c, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE theta AS theta_source
    WRITE_TARGET c
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    theta_source
    c
    weight
    delay
}

BREAKPOINT {
    c = a * weight * theta_source
}
