NEURON {
    POINT_PROCESS ca3_olm_spikes_to_vep
    RANGE x
    RANGE activity
    RANGE tau_ms, gain, population_size
}

MIND {
    ROLE MICRO2MACRO
    WRITE_TARGET x
}

PARAMETER {
    tau_ms = 80.0
    gain = -0.4
    population_size = 200.0
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
    x = gain * activity
}

DERIVATIVE states {
    activity' = -activity / tau_ms
}

NET_RECEIVE(weight) {
    activity = activity + weight / population_size
}
