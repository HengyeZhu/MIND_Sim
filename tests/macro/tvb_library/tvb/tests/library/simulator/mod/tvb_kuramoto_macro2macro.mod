NEURON {
    POINT_PROCESS tvb_kuramoto_macro2macro
    RANGE theta_source, theta_target, c, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE theta AS theta_source
    READ_TARGET theta AS theta_target
    WRITE_TARGET c
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    theta_source
    theta_target
    c
    weight
    delay
}

BREAKPOINT {
    c = a * weight * sin(theta_source - theta_target)
}
