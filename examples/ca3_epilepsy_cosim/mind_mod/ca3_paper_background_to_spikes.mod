MIND {
  MICRO_INPUT ca3_paper_background_to_spikes
  EMIT pyr_soma_ampa pyr_adend3_ampa pyr_soma_gaba pyr_adend3_gaba pyr_adend3_nmda
  EMIT bas_soma_ampa bas_soma_gaba olm_soma_ampa olm_soma_gaba
  EMIT bas_septal_gaba olm_septal_gaba
  RANDOM rng
}

PARAMETER {
  pyr_count = 800.0
  bas_count = 200.0
  olm_count = 200.0
  noise_enabled = 1.0
  background_enabled = 1.0
  septal_enabled = 1.0
  rate_fast_hz = 1000.0
  rate_nmda_hz = 10.0
  septal_start_ms = 50.0
  septal_interval_ms = 150.0
}

INPUT {
  if (background_enabled > 0.5) {
    if (noise_enabled > 0.5) {
      double p_fast = clamp(rate_fast_hz * dt / 1000.0, 0.0, 1.0);
      double p_nmda = clamp(rate_nmda_hz * dt / 1000.0, 0.0, 1.0);
      for (int cell = 0; cell < pyr_count; ++cell) {
        if (uniform(rng) < p_fast) { pyr_soma_ampa(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_fast) { pyr_adend3_ampa(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_fast) { pyr_soma_gaba(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_fast) { pyr_adend3_gaba(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_nmda) { pyr_adend3_nmda(t + uniform(rng) * dt, cell); }
      }
      for (int cell = 0; cell < bas_count; ++cell) {
        if (uniform(rng) < p_fast) { bas_soma_ampa(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_fast) { bas_soma_gaba(t + uniform(rng) * dt, cell); }
      }
      for (int cell = 0; cell < olm_count; ++cell) {
        if (uniform(rng) < p_fast) { olm_soma_ampa(t + uniform(rng) * dt, cell); }
        if (uniform(rng) < p_fast) { olm_soma_gaba(t + uniform(rng) * dt, cell); }
      }
    } else {
      double fast_phase = t - floor(t * rate_fast_hz / 1000.0) * (1000.0 / rate_fast_hz);
      double nmda_phase = t - floor(t * rate_nmda_hz / 1000.0) * (1000.0 / rate_nmda_hz);
      if (fast_phase < dt) {
        for (int cell = 0; cell < pyr_count; ++cell) {
          pyr_soma_ampa(t, cell);
          pyr_adend3_ampa(t, cell);
          pyr_soma_gaba(t, cell);
          pyr_adend3_gaba(t, cell);
        }
        for (int cell = 0; cell < bas_count; ++cell) {
          bas_soma_ampa(t, cell);
          bas_soma_gaba(t, cell);
        }
        for (int cell = 0; cell < olm_count; ++cell) {
          olm_soma_ampa(t, cell);
          olm_soma_gaba(t, cell);
        }
      }
      if (nmda_phase < dt) {
        for (int cell = 0; cell < pyr_count; ++cell) {
          pyr_adend3_nmda(t, cell);
        }
      }
    }
  }

  if (septal_enabled > 0.5) {
    double k = ceil((t - septal_start_ms) / septal_interval_ms);
    if (k < 0.0) {
      k = 0.0;
    }
    double event_time = septal_start_ms + k * septal_interval_ms;
    if (event_time >= t - 1.0e-9 && event_time < t + dt - 1.0e-9) {
      for (int cell = 0; cell < bas_count; ++cell) {
        bas_septal_gaba(event_time, cell);
      }
      for (int cell = 0; cell < olm_count; ++cell) {
        olm_septal_gaba(event_time, cell);
      }
    }
  }
}
