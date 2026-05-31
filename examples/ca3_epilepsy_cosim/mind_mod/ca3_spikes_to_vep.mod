MIND {
  MICRO_OUTPUT ca3_spikes_to_vep
  WRITE x z rate
}

STATE {
  activity = 0.0
  z_state = 0.0
}

PARAMETER {
  cells = 800.0
  tau_activity_ms = 50.0
  tau_z_ms = 2857.0
  x_baseline = -1.8
  x_gain = 2.0
  z_gain = 1.0
}

NET_RECEIVE {
  activity += 1.0 / cells;
}

BREAKPOINT {
  activity *= exp(-dt / tau_activity_ms);
  rate = 1000.0 * activity / tau_activity_ms;
  x = x_baseline + x_gain * activity;
  z_state += dt * ((z_gain * activity) - z_state) / tau_z_ms;
  z = z_state;
}
