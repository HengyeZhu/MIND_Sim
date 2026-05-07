MIND {
  MICRO_OUTPUT spikes_to_exposure
  WRITE S H
}

STATE {
  ca = 0.0
}

PARAMETER {
  cells = 100.0
  tau_ms = 100.0
  exposure_gain = 10.0
}

NET_RECEIVE {
  ca += 1.0 / cells;
}

BREAKPOINT {
  ca *= exp(-dt / tau_ms);
  activity = ca * exposure_gain;
  S = activity;
  H = activity;
}
