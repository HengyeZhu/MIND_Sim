MIND {
  MICRO_INPUT ca3_input_to_spikes
  READ ca3_input
  EMIT afferent
  RANDOM rng
}

PARAMETER {
  cells = 800.0
  base_hz = 1.0
  gain_hz = 45.0
  max_rate_hz = 120.0
  threshold = -0.35
  slope = 4.0
}

INPUT {
  cell_count = cells;
  drive01 = 1.0 / (1.0 + exp(-slope * (ca3_input - threshold)));
  event_rate = clamp(base_hz + gain_hz * drive01, 0.0, max_rate_hz);
  probability = clamp(event_rate * dt / 1000.0, 0.0, 1.0);
  for (int cell = 0; cell < cell_count; ++cell) {
      u = uniform(rng);
      offset = uniform(rng) * dt;
      if (u < probability) {
          afferent(t + offset, cell);
      }
  }
}
