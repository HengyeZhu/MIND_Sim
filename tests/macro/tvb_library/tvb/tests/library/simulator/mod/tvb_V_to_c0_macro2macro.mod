NEURON {
    POINT_PROCESS tvb_V_to_c0_macro2macro
    RANGE V_source, c_0, weight, delay, a
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE V AS V_source
    WRITE_TARGET c_0
}

PARAMETER {
    a = 1.0
}

ASSIGNED {
    V_source
    c_0
    weight
    delay
}

BREAKPOINT {
    c_0 = a * weight * V_source
}
