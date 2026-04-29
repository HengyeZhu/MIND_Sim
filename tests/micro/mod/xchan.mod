NEURON {
    SUFFIX xchan
    USEION x READ ex WRITE ix VALENCE 2
    RANGE gbar, ix
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.001 (S/cm2)
}

ASSIGNED {
    v (mV)
    ex (mV)
    ix (mA/cm2)
}

BREAKPOINT {
    ix = gbar * (v - ex)
}
