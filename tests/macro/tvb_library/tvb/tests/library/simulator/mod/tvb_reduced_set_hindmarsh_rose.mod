NEURON {
    POINT_PROCESS tvb_reduced_set_hindmarsh_rose
    RANGE xi0, eta0, tau0, alpha0, beta0, gamma0, xi1, eta1, tau1, alpha1, beta1, gamma1, xi2, eta2, tau2, alpha2, beta2, gamma2, c_0
    RANGE r, s, K11, K12, K21
    RANGE A00, A01, A02, A10, A11, A12, A20, A21, A22
    RANGE B00, B01, B02, B10, B11, B12, B20, B21, B22
    RANGE C00, C01, C02, C10, C11, C12, C20, C21, C22
    RANGE ai0, ai1, ai2, bi0, bi1, bi2, ci0, ci1, ci2, di0, di1, di2
    RANGE ei0, ei1, ei2, fi0, fi1, fi2, hi0, hi1, hi2, pi0, pi1, pi2
    RANGE IE0, IE1, IE2, II0, II1, II2, m0, m1, m2, n0, n1, n2
}

MIND {
    ROLE REGION
    EXPOSURE xi0, eta0, tau0, alpha0, beta0, gamma0, xi1, eta1, tau1, alpha1, beta1, gamma1, xi2, eta2, tau2, alpha2, beta2, gamma2
}

PARAMETER {
    r = 0.006
    s = 4.0
    K11 = 0.5
    K12 = 0.1
    K21 = 0.15
    A00 = 0.33281658111902213
    A01 = 0.18525241779532706
    A02 = 0.33281658111902396
    A10 = 0.59845379335385496
    A11 = 0.33311144470274762
    A12 = 0.59845379335385829
    A20 = 0.33281658111902052
    A21 = 0.18525241779532614
    A22 = 0.33281658111902235
    B00 = 0.33281658111902213
    B01 = 0.59845379335385496
    B02 = 0.33281658111902052
    B10 = 0.18525241779532706
    B11 = 0.33311144470274762
    B12 = 0.18525241779532614
    B20 = 0.33281658111902396
    B21 = 0.59845379335385829
    B22 = 0.33281658111902235
    C00 = 0.33281658111902213
    C01 = 0.18525241779532706
    C02 = 0.33281658111902396
    C10 = 0.59845379335385496
    C11 = 0.33311144470274762
    C12 = 0.59845379335385829
    C20 = 0.33281658111902052
    C21 = 0.18525241779532614
    C22 = 0.33281658111902235
    ai0 = 1.1996982624881274
    ai1 = 3.8721678089880349
    ai2 = 1.1996982624881147
    bi0 = 3.2859221479507306
    bi1 = 5.9033473793172888
    bi2 = 3.2859221479507141
    ci0 = 0.9129857205749542
    ci1 = 0.50818625556589603
    ci2 = 0.9129857205749593
    di0 = 5.4765369132512181
    di1 = 9.838912298862148
    di2 = 5.4765369132511896
    ei0 = 1.1996982624881274
    ei1 = 3.8721678089880349
    ei2 = 1.1996982624881147
    fi0 = 3.2859221479507306
    fi1 = 5.9033473793172888
    fi2 = 3.2859221479507141
    hi0 = 0.9129857205749542
    hi1 = 0.50818625556589603
    hi2 = 0.9129857205749593
    pi0 = 5.4765369132512181
    pi1 = 9.838912298862148
    pi2 = 5.4765369132511896
    IE0 = 2.5144556680297034
    IE1 = 1.6770146433674569
    IE2 = 3.5112500877650179
    II0 = 2.5144556680297034
    II1 = 1.6770146433674569
    II2 = 3.5112500877650179
    m0 = -0.035058651670078246
    m1 = -0.019514352213730411
    m2 = -0.03505865167007844
    n0 = -0.035058651670078246
    n1 = -0.019514352213730411
    n2 = -0.03505865167007844
    c_0 = 0.0
}

STATE {
    xi0
    eta0
    tau0
    alpha0
    beta0
    gamma0
    xi1
    eta1
    tau1
    alpha1
    beta1
    gamma1
    xi2
    eta2
    tau2
    alpha2
    beta2
    gamma2
}

INITIAL {
    xi0 = 0.0
    eta0 = 0.0
    tau0 = 0.0
    alpha0 = 0.0
    beta0 = 0.0
    gamma0 = 0.0
    xi1 = 0.0
    eta1 = 0.0
    tau1 = 0.0
    alpha1 = 0.0
    beta1 = 0.0
    gamma1 = 0.0
    xi2 = 0.0
    eta2 = 0.0
    tau2 = 0.0
    alpha2 = 0.0
    beta2 = 0.0
    gamma2 = 0.0
}

BREAKPOINT {
    SOLVE states METHOD euler
}

DERIVATIVE states {
    LOCAL dotA0, dotA1, dotA2, dotB0, dotB1, dotB2, dotC0, dotC1, dotC2

    dotA0 = xi0 * A00 + xi1 * A10 + xi2 * A20
    dotA1 = xi0 * A01 + xi1 * A11 + xi2 * A21
    dotA2 = xi0 * A02 + xi1 * A12 + xi2 * A22
    dotB0 = alpha0 * B00 + alpha1 * B10 + alpha2 * B20
    dotB1 = alpha0 * B01 + alpha1 * B11 + alpha2 * B21
    dotB2 = alpha0 * B02 + alpha1 * B12 + alpha2 * B22
    dotC0 = xi0 * C00 + xi1 * C10 + xi2 * C20
    dotC1 = xi0 * C01 + xi1 * C11 + xi2 * C21
    dotC2 = xi0 * C02 + xi1 * C12 + xi2 * C22

    xi0' = eta0 - ai0 * xi0 * xi0 * xi0 + bi0 * xi0 * xi0 - tau0 + K11 * (dotA0 - xi0) - K12 * (dotB0 - xi0) + IE0 + c_0
    eta0' = ci0 - di0 * xi0 * xi0 - eta0
    tau0' = r * s * xi0 - r * tau0 - m0
    alpha0' = beta0 - ei0 * alpha0 * alpha0 * alpha0 + fi0 * alpha0 * alpha0 - gamma0 + K21 * (dotC0 - alpha0) + II0 + c_0
    beta0' = hi0 - pi0 * alpha0 * alpha0 - beta0
    gamma0' = r * s * alpha0 - r * gamma0 - n0

    xi1' = eta1 - ai1 * xi1 * xi1 * xi1 + bi1 * xi1 * xi1 - tau1 + K11 * (dotA1 - xi1) - K12 * (dotB1 - xi1) + IE1 + c_0
    eta1' = ci1 - di1 * xi1 * xi1 - eta1
    tau1' = r * s * xi1 - r * tau1 - m1
    alpha1' = beta1 - ei1 * alpha1 * alpha1 * alpha1 + fi1 * alpha1 * alpha1 - gamma1 + K21 * (dotC1 - alpha1) + II1 + c_0
    beta1' = hi1 - pi1 * alpha1 * alpha1 - beta1
    gamma1' = r * s * alpha1 - r * gamma1 - n1

    xi2' = eta2 - ai2 * xi2 * xi2 * xi2 + bi2 * xi2 * xi2 - tau2 + K11 * (dotA2 - xi2) - K12 * (dotB2 - xi2) + IE2 + c_0
    eta2' = ci2 - di2 * xi2 * xi2 - eta2
    tau2' = r * s * xi2 - r * tau2 - m2
    alpha2' = beta2 - ei2 * alpha2 * alpha2 * alpha2 + fi2 * alpha2 * alpha2 - gamma2 + K21 * (dotC2 - alpha2) + II2 + c_0
    beta2' = hi2 - pi2 * alpha2 * alpha2 - beta2
    gamma2' = r * s * alpha2 - r * gamma2 - n2
}
