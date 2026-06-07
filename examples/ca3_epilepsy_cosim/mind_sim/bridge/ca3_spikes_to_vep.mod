NEURON {
    POINT_PROCESS ca3_spikes_to_vep
    RANGE x
    RANGE pyr_activity, bas_activity, olm_activity
    RANGE tau_pyr_ms, tau_bas_ms, tau_olm_ms
    RANGE x_baseline, pyr_gain, bas_gain, olm_gain
}

MIND {
    ROLE MICRO2MACRO
    SOURCE_EXPOSURE x
}

PARAMETER {
    tau_pyr_ms = 50.0
    tau_bas_ms = 20.0
    tau_olm_ms = 80.0
    x_baseline = -1.8
    pyr_gain = 2.0
    bas_gain = -0.7
    olm_gain = -0.4
}

ASSIGNED {
    x
}

STATE {
    pyr_activity
    bas_activity
    olm_activity
}

INITIAL {
    pyr_activity = 0.0
    bas_activity = 0.0
    olm_activity = 0.0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    x = x_baseline + pyr_gain * pyr_activity + bas_gain * bas_activity + olm_gain * olm_activity
}

DERIVATIVE states {
    pyr_activity' = -pyr_activity / tau_pyr_ms
    bas_activity' = -bas_activity / tau_bas_ms
    olm_activity' = -olm_activity / tau_olm_ms
}

NET_RECEIVE(weight, gid) {
    if (gid >= 0.0 && gid < 800.0) {
        pyr_activity = pyr_activity + weight / 800.0
    }
    if (gid >= 800.0 && gid < 1000.0) {
        bas_activity = bas_activity + weight / 200.0
    }
    if (gid >= 1000.0 && gid < 1200.0) {
        olm_activity = olm_activity + weight / 200.0
    }
}
