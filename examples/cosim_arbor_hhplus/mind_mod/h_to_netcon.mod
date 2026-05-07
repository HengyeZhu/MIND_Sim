MIND {
  MICRO_INPUT h_to_netcon
  READ mean_H
  EMIT afferent
  RANDOM rng
}

PARAMETER {
  cells = 100.0
  scale = 5.0
  max_rate_hz = 150.0
}

INPUT {
  cell_count = cells;
  event_rate = mean_H * scale;
  event_rate = clamp(event_rate, 0.0, max_rate_hz);
  probability = clamp(event_rate * dt / 1000.0, 0.0, 1.0);
  for (int cell = 0; cell < cell_count; ++cell) {
      u = uniform(rng);
      offset = uniform(rng) * dt;
      if (u < probability) {
          afferent(t + offset, cell);
      }
  }
}
