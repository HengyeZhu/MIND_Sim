MIND {
  COUPLING rww_s_coupling
  READ S
  WRITE coupled_S
}

PARAMETER {
  a = 0.096
}

EDGE {
  coupled_S += a * weight * S;
}
