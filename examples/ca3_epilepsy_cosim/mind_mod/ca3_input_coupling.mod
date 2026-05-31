MIND {
  COUPLING ca3_input_coupling
  READ x
  WRITE ca3_input
}

PARAMETER {
  a = 1.0
}

EDGE {
  ca3_input += a * weight * x;
}
