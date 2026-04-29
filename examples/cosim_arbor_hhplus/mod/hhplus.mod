NEURON {
  SUFFIX hhplus
  USEION na WRITE ina
  USEION k WRITE ik
  USEION cl READ ecl WRITE icl VALENCE -1
  NONSPECIFIC_CURRENT ipump
  RANGE ena, ek, kbath, gcl
}

STATE { n dki dkg nai nao ki ko }

CONSTANT {
  C = 26.64 (mV)
  A = 808.078016193 (um2)
  V = 2160.0 (um3)
  F = 9.648533e4 (A s / mol)
}

UNITS {
  (nA) = (nanoamp)
  (mV) = (millivolt)
  (uS) = (microsiemens)
}

ASSIGNED {
  ena (mV)
  ek (mV)
  ecl (mV)
  ina (mA/cm2)
  ik (mA/cm2)
  icl (mA/cm2)
  ipump (mA/cm2)
}

PARAMETER {
  v (mV)
  tau_n    =      0.25  (ms)
  gcl      =      7.5   (nS)
  gkleak   =      0.12  (nS)
  gk       =      22.0  (nS)
  gnaleak  =      0.02  (nS)
  gna      =      40.0  (nS)
  ki0      =  140.0      (mM)
  ko0      =    4.8      (mM)
  kbath    =    9.5      (mM)
  nai0     =   16.0      (mM)
  nao0     =  138.0      (mM)
  beta     =    3.0
  rho      =    250.0 (pA)
  epsilon  =    0.01   (/ms)
  gamma = 0.04 (mmol um3 / C)
}

INITIAL {
  n = n_inf(v)
  dki = -0.6
  dkg =  0.8
  ena = 0
  ek = 0
}

BREAKPOINT {
  LOCAL gnabar, gkbar, gclbar, gpump, g_to_S_cm2
  SOLVE dS METHOD cnexp

  ki  = ki0  + dki
  ko  = ko0  - beta*dki + dkg
  nai = nai0 - dki
  nao = nao0 + beta*dki

  ena = C*log(nao/nai)
  ek  = C*log(ko/ki)

  g_to_S_cm2 = 0.1 / A

  gnabar = g_to_S_cm2*(gnaleak + gna*m_inf(v)*h(n))
  gkbar  = g_to_S_cm2*(gkleak  + gk*n)
  gclbar = g_to_S_cm2*gcl
  gpump  = rho * g_to_S_cm2

  ina   = gnabar*(v - ena)
  ik    = gkbar*(v - ek)
  icl   = gclbar*(v - ecl)
  ipump = gpump/((1.0 + exp(10.5 - 0.5*nai))*(1.0 + exp(5.5 - ko)))
}

DERIVATIVE dS {
  LOCAL lki, lko, lnai, lik, lipump

  lki  = ki0  + dki
  lko  = ko0  - beta*dki + dkg
  lnai = nai0 - dki

  lik    = (gkleak  + gk*n)*(v - C*log(lko/lki))
  lipump = rho/((1.0 + exp(10.5 - 0.5*lnai))*(1.0 + exp(5.5 - lko)))

  n'   = (n_inf(v) - n) / tau_n
  dki' = gamma * (2*lipump - lik) / V
  dkg' = epsilon * (kbath - lko)
}

FUNCTION n_inf(u) { n_inf = 1.0 / (1.0 + exp(-(u + 19.0) / 18.0)) }
FUNCTION m_inf(u) { m_inf = 1.0 / (1.0 + exp(-(24.0 + u) / 12.0)) }
FUNCTION h(x)     { h = 1.1 - 1.0 / (1.0 + exp(3.2 - 8.0 * x)) }
