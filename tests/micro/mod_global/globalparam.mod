NEURON {
    SUFFIX globalparam
    NONSPECIFIC_CURRENT i
    RANGE g, i
}

PARAMETER {
    g = 0.001 (S/cm2)
    shift = 0 (mV)
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
}

BREAKPOINT {
    i = g * (v - shift)
}
