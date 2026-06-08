NEURON {
    POINT_PROCESS ca3_pyr_spikes_to_vep
    RANGE x
    RANGE activity
    RANGE tau_ms, x_baseline, gain, population_size
}

MIND {
    ROLE MICRO2MACRO
    SOURCE_EXPOSURE x
}

PARAMETER {
    tau_ms = 50.0
    x_baseline = -1.8
    gain = 2.0
    population_size = 800.0
}

ASSIGNED {
    x
}

STATE {
    activity
}

INITIAL {
    activity = 0.0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    x = x_baseline + gain * activity
}

DERIVATIVE states {
    activity' = -activity / tau_ms
}

NET_RECEIVE(weight) {
    activity = activity + weight / population_size
}
