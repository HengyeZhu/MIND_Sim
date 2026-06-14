NEURON {
    POINT_PROCESS tvb_reduced_set_fitzhugh_nagumo
    RANGE xi0, eta0, alpha0, beta0, xi1, eta1, alpha1, beta1, xi2, eta2, alpha2, beta2, c_0
    RANGE tau, b, K11, K12, K21
    RANGE A00, A01, A02, A10, A11, A12, A20, A21, A22
    RANGE B00, B01, B02, B10, B11, B12, B20, B21, B22
    RANGE C00, C01, C02, C10, C11, C12, C20, C21, C22
    RANGE e0, e1, e2, f0, f1, f2, IE0, IE1, IE2, II0, II1, II2, m0, m1, m2, n0, n1, n2
}

MIND {
    ROLE REGION
    EXPOSURE xi0, eta0, alpha0, beta0, xi1, eta1, alpha1, beta1, xi2, eta2, alpha2, beta2
}

PARAMETER {
    tau = 3.0
    b = 0.9
    K11 = 0.5
    K12 = 0.15
    K21 = 0.15
    A00 = 0.33281658111902224
    A01 = 0.18525241779532711
    A02 = 0.33281658111902407
    A10 = 0.59845379335385518
    A11 = 0.33311144470274773
    A12 = 0.5984537933538584
    A20 = 0.3328165811190204
    A21 = 0.18525241779532609
    A22 = 0.33281658111902218
    B00 = 0.33281658111902224
    B01 = 0.59845379335385518
    B02 = 0.3328165811190204
    B10 = 0.18525241779532711
    B11 = 0.33311144470274773
    B12 = 0.18525241779532609
    B20 = 0.33281658111902407
    B21 = 0.5984537933538584
    B22 = 0.33281658111902218
    C00 = 0.33281658111902224
    C01 = 0.18525241779532711
    C02 = 0.33281658111902407
    C10 = 0.59845379335385518
    C11 = 0.33311144470274773
    C12 = 0.5984537933538584
    C20 = 0.3328165811190204
    C21 = 0.18525241779532609
    C22 = 0.33281658111902218
    e0 = 1.0283127964183951
    e1 = 3.3190009791326007
    e2 = 1.0283127964183829
    f0 = 1.0283127964183951
    f1 = 3.3190009791326007
    f2 = 1.0283127964183829
    IE0 = -0.62805226593708963
    IE1 = 0.0
    IE2 = 0.62805226593709829
    II0 = -0.62805226593708963
    II1 = 0.0
    II2 = 0.62805226593709829
    m0 = 0.44376177872964995
    m1 = 0.24700675116130505
    m2 = 0.44376177872965239
    n0 = 0.44376177872964995
    n1 = 0.24700675116130505
    n2 = 0.44376177872965239
    c_0 = 0.0
}

STATE {
    xi0
    eta0
    alpha0
    beta0
    xi1
    eta1
    alpha1
    beta1
    xi2
    eta2
    alpha2
    beta2
}

INITIAL {
    xi0 = 0.0
    eta0 = 0.0
    alpha0 = 0.0
    beta0 = 0.0
    xi1 = 0.0
    eta1 = 0.0
    alpha1 = 0.0
    beta1 = 0.0
    xi2 = 0.0
    eta2 = 0.0
    alpha2 = 0.0
    beta2 = 0.0
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

    xi0' = tau * (xi0 - e0 * xi0 * xi0 * xi0 / 3.0 - eta0) + K11 * (dotA0 - xi0) - K12 * (dotB0 - xi0) + tau * (IE0 + c_0)
    eta0' = (xi0 - b * eta0 + m0) / tau
    alpha0' = tau * (alpha0 - f0 * alpha0 * alpha0 * alpha0 / 3.0 - beta0) + K21 * (dotC0 - alpha0) + tau * (II0 + c_0)
    beta0' = (alpha0 - b * beta0 + n0) / tau

    xi1' = tau * (xi1 - e1 * xi1 * xi1 * xi1 / 3.0 - eta1) + K11 * (dotA1 - xi1) - K12 * (dotB1 - xi1) + tau * (IE1 + c_0)
    eta1' = (xi1 - b * eta1 + m1) / tau
    alpha1' = tau * (alpha1 - f1 * alpha1 * alpha1 * alpha1 / 3.0 - beta1) + K21 * (dotC1 - alpha1) + tau * (II1 + c_0)
    beta1' = (alpha1 - b * beta1 + n1) / tau

    xi2' = tau * (xi2 - e2 * xi2 * xi2 * xi2 / 3.0 - eta2) + K11 * (dotA2 - xi2) - K12 * (dotB2 - xi2) + tau * (IE2 + c_0)
    eta2' = (xi2 - b * eta2 + m2) / tau
    alpha2' = tau * (alpha2 - f2 * alpha2 * alpha2 * alpha2 / 3.0 - beta2) + K21 * (dotC2 - alpha2) + tau * (II2 + c_0)
    beta2' = (alpha2 - b * beta2 + n2) / tau
}
