MIND {
  COUPLING mean_h_input
  READ H
  WRITE mean_H
}

PARAMETER {
  inv_source_count = 1.0
}

EDGE {
  mean_H += inv_source_count * H;
}
