NEURON {
    POINT_PROCESS tvb_x_to_c_raw_macro2macro
    RANGE x_source, c_raw, weight, delay
}

MIND {
    ROLE MACRO2MACRO
    READ_SOURCE x AS x_source
    WRITE_TARGET c_raw
}

ASSIGNED {
    x_source
    c_raw
    weight
    delay
}

BREAKPOINT {
    c_raw = weight * x_source
}
