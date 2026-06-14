NEURON {
    POINT_PROCESS tvb_larter_breakspear
    RANGE c_0, V, W, Z
    RANGE gCa, gK, gL, phi, gNa, TK, TCa, TNa
    RANGE VCa, VK, VL, VNa
    RANGE d_K, tau_K, d_Na, d_Ca
    RANGE aei, aie, b, C, ane, ani, aee
    RANGE Iext, rNMDA, VT, d_V, ZT, d_Z, QV_max, QZ_max, t_scale
}

MIND {
    ROLE REGION
    EXPOSURE V, W, Z
}

PARAMETER {
    gCa = 1.1
    gK = 2.0
    gL = 0.5
    phi = 0.7
    gNa = 6.7
    TK = 0.0
    TCa = -0.01
    TNa = 0.3
    VCa = 1.0
    VK = -0.7
    VL = -0.5
    VNa = 0.53
    d_K = 0.3
    tau_K = 1.0
    d_Na = 0.15
    d_Ca = 0.15
    aei = 2.0
    aie = 2.0
    b = 0.1
    C = 0.1
    ane = 1.0
    ani = 0.4
    aee = 0.4
    Iext = 0.3
    rNMDA = 0.25
    VT = 0.0
    d_V = 0.65
    ZT = 0.0
    d_Z = 0.7
    QV_max = 1.0
    QZ_max = 1.0
    t_scale = 1.0
    c_0 = 0.0
}

STATE {
    V
    W
    Z
}

INITIAL {
    V = 0.0
    W = 0.0
    Z = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL m_Ca, m_Na, m_K, QV, QZ

    m_Ca = 0.5 * (1.0 + tanh((V - TCa) / d_Ca))
    m_Na = 0.5 * (1.0 + tanh((V - TNa) / d_Na))
    m_K = 0.5 * (1.0 + tanh((V - TK) / d_K))
    QV = 0.5 * QV_max * (1.0 + tanh((V - VT) / d_V))
    QZ = 0.5 * QZ_max * (1.0 + tanh((Z - ZT) / d_Z))

    V' = t_scale * (
        -(gCa + (1.0 - C) * rNMDA * aee * QV + C * rNMDA * aee * c_0) * m_Ca * (V - VCa)
        - gK * W * (V - VK)
        - gL * (V - VL)
        - (gNa * m_Na + (1.0 - C) * aee * QV + C * aee * c_0) * (V - VNa)
        - aie * Z * QZ
        + ane * Iext
    )
    W' = t_scale * phi * (m_K - W) / tau_K
    Z' = t_scale * b * (ani * Iext + aei * V * QV)
}
