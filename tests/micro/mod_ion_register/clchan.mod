NEURON {
    SUFFIX clchan
    USEION cl READ ecl WRITE icl
    RANGE gbar, icl
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
    ecl (mV)
    icl (mA/cm2)
}

BREAKPOINT {
    icl = gbar * (v - ecl)
}
