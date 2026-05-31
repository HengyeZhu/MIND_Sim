MIND {
  COUPLING vep_x_coupling
  READ x
  WRITE coupled_x
}

PARAMETER {
  a = 1.0
}

EDGE {
  coupled_x += a * weight * x;
}
