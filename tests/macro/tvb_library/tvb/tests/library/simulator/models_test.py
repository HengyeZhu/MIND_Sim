from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent / "mod"


@pytest.fixture(scope="session")
def macro_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def _configure_macro_runtime(macro_mod_dir, dt):
    ms.macro.load_mech(macro_mod_dir)
    ms.macro.dt(dt)
    ms.macro.exchange_window(dt)


def _records_cube(result):
    records = result.records
    return np.asarray(records.values).reshape(
        records.sample_count,
        records.recorded_roi_count,
        records.output_count,
    )


class TestModels:
    @staticmethod
    def _wilson_cowan_params(**overrides):
        params = {
            "c_ee": 12.0,
            "c_ei": 4.0,
            "c_ie": 13.0,
            "c_ii": 11.0,
            "tau_e": 10.0,
            "tau_i": 10.0,
            "a_e": 1.2,
            "b_e": 2.8,
            "c_e": 1.0,
            "theta_e": 0.0,
            "a_i": 1.0,
            "b_i": 4.0,
            "c_i": 1.0,
            "theta_i": 0.0,
            "r_e": 1.0,
            "r_i": 1.0,
            "k_e": 1.0,
            "k_i": 1.0,
            "P": 0.0,
            "Q": 0.0,
            "alpha_e": 1.0,
            "alpha_i": 1.0,
            "shift_sigmoid": 1.0,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _generic2d_params(**overrides):
        params = {
            "tau": 1.0,
            "I": 0.0,
            "a": -2.0,
            "b": -10.0,
            "c": 0.0,
            "d": 0.02,
            "e": 3.0,
            "f": 1.0,
            "g": 0.0,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _jansen_rit_params(**overrides):
        params = {
            "A": 3.25,
            "B": 22.0,
            "a": 0.1,
            "b": 0.05,
            "v0": 5.52,
            "nu_max": 0.0025,
            "r": 0.56,
            "J": 135.0,
            "a_1": 1.0,
            "a_2": 0.8,
            "a_3": 0.25,
            "a_4": 0.25,
            "p_min": 0.12,
            "p_max": 0.32,
            "mu": 0.22,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _reduced_wong_wang_params(**overrides):
        params = {
            "a": 0.270,
            "b": 0.108,
            "d": 154.0,
            "gamma": 0.641,
            "tau_s": 100.0,
            "w": 0.6,
            "J_N": 0.2609,
            "I_o": 0.33,
            "sigma_noise": 1e-9,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _larter_breakspear_params(**overrides):
        params = {
            "gCa": 1.1,
            "gK": 2.0,
            "gL": 0.5,
            "phi": 0.7,
            "gNa": 6.7,
            "TK": 0.0,
            "TCa": -0.01,
            "TNa": 0.3,
            "VCa": 1.0,
            "VK": -0.7,
            "VL": -0.5,
            "VNa": 0.53,
            "d_K": 0.3,
            "tau_K": 1.0,
            "d_Na": 0.15,
            "d_Ca": 0.15,
            "aei": 2.0,
            "aie": 2.0,
            "b": 0.1,
            "C": 0.1,
            "ane": 1.0,
            "ani": 0.4,
            "aee": 0.4,
            "Iext": 0.3,
            "rNMDA": 0.25,
            "VT": 0.0,
            "d_V": 0.65,
            "ZT": 0.0,
            "d_Z": 0.7,
            "QV_max": 1.0,
            "QZ_max": 1.0,
            "t_scale": 1.0,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _epileptor_params(**overrides):
        params = {
            "x0": -1.6,
            "Iext": 3.1,
            "Iext2": 0.45,
            "a": 1.0,
            "b": 3.0,
            "slope": 0.0,
            "tt": 1.0,
            "Kvf": 0.0,
            "c": 1.0,
            "d": 5.0,
            "r": 0.00035,
            "Ks": 0.0,
            "Kf": 0.0,
            "aa": 6.0,
            "bb": 2.0,
            "tau": 10.0,
            "modification": 0.0,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _wilson_cowan_rhs(E, I, c_0, params):
        x_e = params["alpha_e"] * (
            params["c_ee"] * E
            - params["c_ei"] * I
            + params["P"]
            - params["theta_e"]
            + c_0
        )
        x_i = params["alpha_i"] * (
            params["c_ie"] * E
            - params["c_ii"] * I
            + params["Q"]
            - params["theta_i"]
        )
        if params["shift_sigmoid"] > 0.5:
            s_e = params["c_e"] * (
                1.0 / (1.0 + np.exp(-params["a_e"] * (x_e - params["b_e"])))
                - 1.0 / (1.0 + np.exp(-params["a_e"] * (-params["b_e"])))
            )
            s_i = params["c_i"] * (
                1.0 / (1.0 + np.exp(-params["a_i"] * (x_i - params["b_i"])))
                - 1.0 / (1.0 + np.exp(-params["a_i"] * (-params["b_i"])))
            )
        else:
            s_e = params["c_e"] / (1.0 + np.exp(-params["a_e"] * (x_e - params["b_e"])))
            s_i = params["c_i"] / (1.0 + np.exp(-params["a_i"] * (x_i - params["b_i"])))
        dE = (-E + (params["k_e"] - params["r_e"] * E) * s_e) / params["tau_e"]
        dI = (-I + (params["k_i"] - params["r_i"] * I) * s_i) / params["tau_i"]
        return dE, dI

    @staticmethod
    def _generic2d_rhs(V, W, c_0, params):
        V2 = V * V
        dV = params["d"] * params["tau"] * (
            params["alpha"] * W
            - params["f"] * V2 * V
            + params["e"] * V2
            + params["g"] * V
            + params["gamma"] * params["I"]
            + params["gamma"] * c_0
        )
        dW = params["d"] * (
            params["a"] + params["b"] * V + params["c"] * V2 - params["beta"] * W
        ) / params["tau"]
        return dV, dW

    @staticmethod
    def _jansen_rit_rhs(state, c_0, params):
        y0, y1, y2, y3, y4, y5 = state
        sigm_y1_y2 = 2.0 * params["nu_max"] / (
            1.0 + np.exp(params["r"] * (params["v0"] - (y1 - y2)))
        )
        sigm_y0_1 = 2.0 * params["nu_max"] / (
            1.0 + np.exp(params["r"] * (params["v0"] - (params["a_1"] * params["J"] * y0)))
        )
        sigm_y0_3 = 2.0 * params["nu_max"] / (
            1.0 + np.exp(params["r"] * (params["v0"] - (params["a_3"] * params["J"] * y0)))
        )
        return np.array(
            [
                y3,
                y4,
                y5,
                params["A"] * params["a"] * sigm_y1_y2
                - 2.0 * params["a"] * y3
                - params["a"] * params["a"] * y0,
                params["A"]
                * params["a"]
                * (params["mu"] + params["a_2"] * params["J"] * sigm_y0_1 + c_0)
                - 2.0 * params["a"] * y4
                - params["a"] * params["a"] * y1,
                params["B"] * params["b"] * (params["a_4"] * params["J"] * sigm_y0_3)
                - 2.0 * params["b"] * y5
                - params["b"] * params["b"] * y2,
            ]
        )

    @staticmethod
    def _reduced_wong_wang_rhs(S, c_0, params):
        x = params["w"] * params["J_N"] * S + params["I_o"] + params["J_N"] * c_0
        H = (params["a"] * x - params["b"]) / (
            1.0 - np.exp(-params["d"] * (params["a"] * x - params["b"]))
        )
        return -S / params["tau_s"] + (1.0 - S) * H * params["gamma"]

    @staticmethod
    def _larter_breakspear_rhs(V, W, Z, c_0, params):
        m_Ca = 0.5 * (1.0 + np.tanh((V - params["TCa"]) / params["d_Ca"]))
        m_Na = 0.5 * (1.0 + np.tanh((V - params["TNa"]) / params["d_Na"]))
        m_K = 0.5 * (1.0 + np.tanh((V - params["TK"]) / params["d_K"]))
        QV = 0.5 * params["QV_max"] * (1.0 + np.tanh((V - params["VT"]) / params["d_V"]))
        QZ = 0.5 * params["QZ_max"] * (1.0 + np.tanh((Z - params["ZT"]) / params["d_Z"]))
        dV = params["t_scale"] * (
            -(
                params["gCa"]
                + (1.0 - params["C"]) * params["rNMDA"] * params["aee"] * QV
                + params["C"] * params["rNMDA"] * params["aee"] * c_0
            )
            * m_Ca
            * (V - params["VCa"])
            - params["gK"] * W * (V - params["VK"])
            - params["gL"] * (V - params["VL"])
            - (
                params["gNa"] * m_Na
                + (1.0 - params["C"]) * params["aee"] * QV
                + params["C"] * params["aee"] * c_0
            )
            * (V - params["VNa"])
            - params["aie"] * Z * QZ
            + params["ane"] * params["Iext"]
        )
        dW = params["t_scale"] * params["phi"] * (m_K - W) / params["tau_K"]
        dZ = params["t_scale"] * params["b"] * (
            params["ani"] * params["Iext"] + params["aei"] * V * QV
        )
        return dV, dW, dZ

    @staticmethod
    def _epileptor_rhs(state, c_pop1, c_pop2, params):
        x1, y1, z, x2, y2, g = state
        if x1 < 0.0:
            fast_x1 = -params["a"] * x1 * x1 + params["b"] * x1
        else:
            fast_x1 = params["slope"] - x2 + 0.6 * (z - 4.0) * (z - 4.0)
        if params["modification"] > 0.5:
            h = params["x0"] + 3.0 / (1.0 + np.exp(-(x1 + 0.5) / 0.1))
        else:
            h = 4.0 * (x1 - params["x0"])
            if z < 0.0:
                h -= 0.1 * z ** 7
        if x2 < -0.25:
            f2 = 0.0
        else:
            f2 = params["aa"] * (x2 + 0.25)
        return np.array(
            [
                params["tt"] * (y1 - z + params["Iext"] + params["Kvf"] * c_pop1 + fast_x1 * x1),
                params["tt"] * (params["c"] - params["d"] * x1 * x1 - y1),
                params["tt"] * params["r"] * (h - z + params["Ks"] * c_pop1),
                params["tt"]
                * (
                    -y2
                    + x2
                    - x2 * x2 * x2
                    + params["Iext2"]
                    + params["bb"] * g
                    - 0.3 * (z - 3.5)
                    + params["Kf"] * c_pop2
                ),
                params["tt"] * ((-y2 + f2) / params["tau"]),
                params["tt"] * (-0.01 * (g - 0.1 * x1)),
            ]
        )

    @staticmethod
    def _linear_rhs(x, c, gamma=-10.0):
        return gamma * x + c

    @staticmethod
    def _kuramoto_rhs(theta, c, omega=1.0):
        return omega + c

    @staticmethod
    def _hopfield_rhs(x, c, taux=1.0):
        dx = (-x + c) / taux
        return dx, dx

    @staticmethod
    def _two_roi_network(model, outputs, states, weights, coupling_rule, params=None):
        rois = ms.macro.load_rois(
            labels=["A", "B"],
            weights=weights,
            delays=np.zeros((2, 2)),
        )
        for roi in rois.rois():
            roi.use_macro(model, initial_state=states[roi.label], params=params or {})
            for output in outputs:
                roi.record(output)
        rois.roi("A").insert("B", coupling_rule)
        rois.roi("B").insert("A", coupling_rule)
        history = np.array(
            [[[states["A"][output], states["B"][output]] for output in outputs]],
            dtype=float,
        )
        rois.initial_history(history, outputs=outputs, layout="time_output_roi")
        return rois

    @staticmethod
    def _two_roi_model(model, outputs, states, constant_params=None, params=None):
        rois = ms.macro.load_rois(
            labels=["A", "B"],
            weights=np.zeros((2, 2)),
            delays=np.zeros((2, 2)),
        )
        for roi in rois.rois():
            roi_params = dict(params or {})
            if constant_params is not None:
                roi_params.update(constant_params[roi.label])
            roi.use_macro(model, initial_state=states[roi.label], params=roi_params)
            for output in outputs:
                roi.record(output)
        history = np.array(
            [[[states[label][output] for label in ("A", "B")] for output in outputs]],
            dtype=float,
        )
        rois.initial_history(history, outputs=outputs, layout="time_output_roi")
        return rois

    @staticmethod
    def _matrix_params(prefix, value):
        matrix = np.asarray(value)
        return {
            f"{prefix}{row}{col}": float(matrix[row, col])
            for row in range(matrix.shape[0])
            for col in range(matrix.shape[1])
        }

    @staticmethod
    def _fhn_params(**overrides):
        params = {"tau": 3.0, "b": 0.9, "K11": 0.5, "K12": 0.15, "K21": 0.15}
        nu = 1500
        nv = 1500
        number_of_modes = 3
        a = 0.45
        sigma = 0.35
        mu = 0.0
        stepu = 1.0 / (nu + 2 - 1)
        stepv = 1.0 / (nv + 2 - 1)
        distribution = norm(loc=mu, scale=sigma)
        zu = distribution.ppf(np.arange(stepu, 1.0, stepu))
        zv = distribution.ppf(np.arange(stepv, 1.0, stepv))
        v = np.zeros((number_of_modes, nv))
        u = np.zeros((number_of_modes, nu))
        nv_per_mode = nv // number_of_modes
        nu_per_mode = nu // number_of_modes
        for mode in range(number_of_modes):
            v[mode, mode * nv_per_mode : (mode + 1) * nv_per_mode] = 1.0
            u[mode, mode * nu_per_mode : (mode + 1) * nu_per_mode] = 1.0
        v = v / np.tile(np.sqrt(trapezoid(v * v, zv, axis=1)), (nv, 1)).T
        u = u / np.tile(np.sqrt(trapezoid(u * u, zu, axis=1)), (nv, 1)).T
        g1 = distribution.pdf(zv)
        g2 = distribution.pdf(zu)
        cv = np.conj(v)
        cu = np.conj(u)
        int_cv_dz = trapezoid(cv, zv, axis=1)[:, np.newaxis]
        int_cu_dz = trapezoid(cu, zu, axis=1)[:, np.newaxis]
        params.update(TestModels._matrix_params("A", np.dot(int_cv_dz, trapezoid(np.tile(g1, (3, 1)) * v, zv, axis=1)[np.newaxis, :]).T))
        params.update(TestModels._matrix_params("B", np.dot(int_cv_dz, trapezoid(np.tile(g2, (3, 1)) * u, zu, axis=1)[np.newaxis, :])))
        params.update(TestModels._matrix_params("C", np.dot(int_cu_dz, trapezoid(np.tile(g1, (3, 1)) * v, zv, axis=1)[np.newaxis, :]).T))
        for index, value in enumerate(trapezoid(cv * v**3, zv, axis=1)):
            params[f"e{index}"] = float(value)
        for index, value in enumerate(trapezoid(cu * u**3, zu, axis=1)):
            params[f"f{index}"] = float(value)
        for index, value in enumerate(trapezoid(zv * cv, zv, axis=1)):
            params[f"IE{index}"] = float(value)
        for index, value in enumerate(trapezoid(zu * cu, zu, axis=1)):
            params[f"II{index}"] = float(value)
        for index, value in enumerate((a * int_cv_dz).T[0]):
            params[f"m{index}"] = float(value)
        for index, value in enumerate((a * int_cu_dz).T[0]):
            params[f"n{index}"] = float(value)
        params.update(overrides)
        return params

    @staticmethod
    def _hr_params(**overrides):
        params = {"r": 0.006, "s": 4.0, "K11": 0.5, "K12": 0.1, "K21": 0.15}
        nu = 1500
        nv = 1500
        number_of_modes = 3
        a = 1.0
        b = 3.0
        c = 1.0
        d = 5.0
        xo = -1.6
        sigma = 0.3
        mu = 3.3
        stepu = 1.0 / (nu + 2 - 1)
        stepv = 1.0 / (nv + 2 - 1)
        distribution = norm(loc=mu, scale=sigma)
        iu = distribution.ppf(np.arange(stepu, 1.0, stepu))
        iv = distribution.ppf(np.arange(stepv, 1.0, stepv))
        v = np.zeros((number_of_modes, nv))
        u = np.zeros((number_of_modes, nu))
        nv_per_mode = nv // number_of_modes
        nu_per_mode = nu // number_of_modes
        for mode in range(number_of_modes):
            v[mode, mode * nv_per_mode : (mode + 1) * nv_per_mode] = 1.0
            u[mode, mode * nu_per_mode : (mode + 1) * nu_per_mode] = 1.0
        v = v / np.tile(np.sqrt(trapezoid(v * v, iv, axis=1)), (nv, 1)).T
        u = u / np.tile(np.sqrt(trapezoid(u * u, iu, axis=1)), (nu, 1)).T
        g1 = distribution.pdf(iv)
        g2 = distribution.pdf(iu)
        cv = np.conj(v)
        cu = np.conj(u)
        int_cv_di = trapezoid(cv, iv, axis=1)[:, np.newaxis]
        int_cu_di = trapezoid(cu, iu, axis=1)[:, np.newaxis]
        params.update(TestModels._matrix_params("A", np.dot(int_cv_di, trapezoid(np.tile(g1, (3, 1)) * v, iv, axis=1)[np.newaxis, :]).T))
        params.update(TestModels._matrix_params("B", np.dot(int_cv_di, trapezoid(np.tile(g2, (3, 1)) * u, iu, axis=1)[np.newaxis, :])))
        params.update(TestModels._matrix_params("C", np.dot(int_cu_di, trapezoid(np.tile(g1, (3, 1)) * v, iv, axis=1)[np.newaxis, :]).T))
        for index, value in enumerate(a * trapezoid(cv * v**3, iv, axis=1)):
            params[f"ai{index}"] = float(value)
        for index, value in enumerate(b * trapezoid(cv * v**2, iv, axis=1)):
            params[f"bi{index}"] = float(value)
        for index, value in enumerate((c * int_cv_di).T[0]):
            params[f"ci{index}"] = float(value)
        for index, value in enumerate(d * trapezoid(cv * v**2, iv, axis=1)):
            params[f"di{index}"] = float(value)
        for index, value in enumerate(a * trapezoid(cu * u**3, iu, axis=1)):
            params[f"ei{index}"] = float(value)
        for index, value in enumerate(b * trapezoid(cu * u**2, iu, axis=1)):
            params[f"fi{index}"] = float(value)
        for index, value in enumerate((c * int_cu_di).T[0]):
            params[f"hi{index}"] = float(value)
        for index, value in enumerate(d * trapezoid(cu * u**2, iu, axis=1)):
            params[f"pi{index}"] = float(value)
        for index, value in enumerate(trapezoid(iv * cv, iv, axis=1)):
            params[f"IE{index}"] = float(value)
        for index, value in enumerate(trapezoid(iu * cu, iu, axis=1)):
            params[f"II{index}"] = float(value)
        for index, value in enumerate((params["r"] * params["s"] * xo * int_cv_di).T[0]):
            params[f"m{index}"] = float(value)
        for index, value in enumerate((params["r"] * params["s"] * xo * int_cu_di).T[0]):
            params[f"n{index}"] = float(value)
        params.update(overrides)
        return params

    @staticmethod
    def _fhn_rhs(state, c_0, params):
        xi = np.array([state[0], state[4], state[8]])
        eta = np.array([state[1], state[5], state[9]])
        alpha = np.array([state[2], state[6], state[10]])
        beta = np.array([state[3], state[7], state[11]])
        a_matrix = np.array([[params[f"A{row}{col}"] for col in range(3)] for row in range(3)])
        b_matrix = np.array([[params[f"B{row}{col}"] for col in range(3)] for row in range(3)])
        c_matrix = np.array([[params[f"C{row}{col}"] for col in range(3)] for row in range(3)])
        e = np.array([params[f"e{i}"] for i in range(3)])
        f = np.array([params[f"f{i}"] for i in range(3)])
        ie = np.array([params[f"IE{i}"] for i in range(3)])
        ii = np.array([params[f"II{i}"] for i in range(3)])
        m = np.array([params[f"m{i}"] for i in range(3)])
        n = np.array([params[f"n{i}"] for i in range(3)])
        dxi = (
            params["tau"] * (xi - e * xi**3 / 3.0 - eta)
            + params["K11"] * (np.dot(xi, a_matrix) - xi)
            - params["K12"] * (np.dot(alpha, b_matrix) - xi)
            + params["tau"] * (ie + c_0)
        )
        deta = (xi - params["b"] * eta + m) / params["tau"]
        dalpha = (
            params["tau"] * (alpha - f * alpha**3 / 3.0 - beta)
            + params["K21"] * (np.dot(xi, c_matrix) - alpha)
            + params["tau"] * (ii + c_0)
        )
        dbeta = (alpha - params["b"] * beta + n) / params["tau"]
        return np.array([dxi[0], deta[0], dalpha[0], dbeta[0], dxi[1], deta[1], dalpha[1], dbeta[1], dxi[2], deta[2], dalpha[2], dbeta[2]])

    @staticmethod
    def _hr_rhs(state, c_0, params):
        xi = np.array([state[0], state[6], state[12]])
        eta = np.array([state[1], state[7], state[13]])
        tau_state = np.array([state[2], state[8], state[14]])
        alpha = np.array([state[3], state[9], state[15]])
        beta = np.array([state[4], state[10], state[16]])
        gamma = np.array([state[5], state[11], state[17]])
        a_matrix = np.array([[params[f"A{row}{col}"] for col in range(3)] for row in range(3)])
        b_matrix = np.array([[params[f"B{row}{col}"] for col in range(3)] for row in range(3)])
        c_matrix = np.array([[params[f"C{row}{col}"] for col in range(3)] for row in range(3)])
        ai = np.array([params[f"ai{i}"] for i in range(3)])
        bi = np.array([params[f"bi{i}"] for i in range(3)])
        ci = np.array([params[f"ci{i}"] for i in range(3)])
        di = np.array([params[f"di{i}"] for i in range(3)])
        ei = np.array([params[f"ei{i}"] for i in range(3)])
        fi = np.array([params[f"fi{i}"] for i in range(3)])
        hi = np.array([params[f"hi{i}"] for i in range(3)])
        pi = np.array([params[f"pi{i}"] for i in range(3)])
        ie = np.array([params[f"IE{i}"] for i in range(3)])
        ii = np.array([params[f"II{i}"] for i in range(3)])
        m = np.array([params[f"m{i}"] for i in range(3)])
        n = np.array([params[f"n{i}"] for i in range(3)])
        dxi = (
            eta
            - ai * xi**3
            + bi * xi**2
            - tau_state
            + params["K11"] * (np.dot(xi, a_matrix) - xi)
            - params["K12"] * (np.dot(alpha, b_matrix) - xi)
            + ie
            + c_0
        )
        deta = ci - di * xi**2 - eta
        dtau = params["r"] * params["s"] * xi - params["r"] * tau_state - m
        dalpha = (
            beta
            - ei * alpha**3
            + fi * alpha**2
            - gamma
            + params["K21"] * (np.dot(xi, c_matrix) - alpha)
            + ii
            + c_0
        )
        dbeta = hi - pi * alpha**2 - beta
        dgamma = params["r"] * params["s"] * alpha - params["r"] * gamma - n
        return np.array([dxi[0], deta[0], dtau[0], dalpha[0], dbeta[0], dgamma[0], dxi[1], deta[1], dtau[1], dalpha[1], dbeta[1], dgamma[1], dxi[2], deta[2], dtau[2], dalpha[2], dbeta[2], dgamma[2]])

    @staticmethod
    def _rww_exc_inh_params(**overrides):
        params = {
            "a_e": 310.0,
            "b_e": 125.0,
            "d_e": 0.160,
            "gamma_e": 0.641 / 1000.0,
            "tau_e": 100.0,
            "w_p": 1.4,
            "W_e": 1.0,
            "J_N": 0.15,
            "a_i": 615.0,
            "b_i": 177.0,
            "d_i": 0.087,
            "gamma_i": 1.0 / 1000.0,
            "tau_i": 10.0,
            "W_i": 0.7,
            "J_i": 1.0,
            "G": 2.0,
            "lamda": 0.0,
            "I_o": 0.382,
            "I_ext": 0.0,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _rww_exc_inh_rhs(S_e, S_i, c_0, params):
        coupling = params["G"] * params["J_N"] * c_0
        J_N_S_e = params["J_N"] * S_e
        x_e = (
            params["w_p"] * J_N_S_e
            - params["J_i"] * S_i
            + params["W_e"] * params["I_o"]
            + coupling
            + params["I_ext"]
        )
        x_e = params["a_e"] * x_e - params["b_e"]
        H_e = x_e / (1.0 - np.exp(-params["d_e"] * x_e))
        dS_e = -(S_e / params["tau_e"]) + (1.0 - S_e) * H_e * params["gamma_e"]
        x_i = J_N_S_e - S_i + params["W_i"] * params["I_o"] + params["lamda"] * coupling
        x_i = params["a_i"] * x_i - params["b_i"]
        H_i = x_i / (1.0 - np.exp(-params["d_i"] * x_i))
        dS_i = -(S_i / params["tau_i"]) + H_i * params["gamma_i"]
        return dS_e, dS_i

    @staticmethod
    def _deco_balanced_rhs(S_e, S_i, c_0, params):
        coupling = params["G"] * params["J_N"] * c_0
        J_N_S_e = params["J_N"] * S_e
        I_e = (
            params["W_e"] * params["I_o"]
            + params["w_p"] * J_N_S_e
            + coupling
            - params["J_i"] * S_i
            + params["I_ext"]
        )
        x_e = (params["a_e"] * I_e - params["b_e"]) * params["M_i"]
        H_e = x_e / (1.0 - np.exp(-params["d_e"] * x_e))
        dS_e = -(S_e / params["tau_e"]) + (1.0 - S_e) * H_e * params["gamma_e"]
        I_i = params["W_i"] * params["I_o"] + J_N_S_e - S_i + params["lamda"] * coupling
        x_i = (params["a_i"] * I_i - params["b_i"]) * params["M_i"]
        H_i = x_i / (1.0 - np.exp(-params["d_i"] * x_i))
        dS_i = -(S_i / params["tau_i"]) + H_i * params["gamma_i"]
        return dS_e, dS_i

    @staticmethod
    def _zetterberg_params(**overrides):
        params = {
            "He": 3.25,
            "Hi": 22.0,
            "ke": 0.1,
            "ki": 0.05,
            "e0": 0.0025,
            "rho_2": 6.0,
            "rho_1": 0.56,
            "gamma_1": 135.0,
            "gamma_2": 108.0,
            "gamma_3": 33.75,
            "gamma_4": 33.75,
            "gamma_5": 15.0,
            "gamma_1T": 1.0,
            "gamma_2T": 1.0,
            "gamma_3T": 1.0,
            "P": 0.12,
            "U": 0.12,
            "Q": 0.12,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _zetterberg_sigma(value, params):
        return 2.0 * params["e0"] / (1.0 + np.exp(params["rho_1"] * (params["rho_2"] - value)))

    @staticmethod
    def _zetterberg_rhs(state, c_0, params):
        v1, y1, v2, y2, v3, y3, v4, y4, v5, y5, v6, v7 = state
        Heke = params["He"] * params["ke"]
        Hiki = params["Hi"] * params["ki"]
        ke_2 = 2.0 * params["ke"]
        ki_2 = 2.0 * params["ki"]
        keke = params["ke"] ** 2
        kiki = params["ki"] ** 2
        coupled_input = TestModels._zetterberg_sigma(c_0, params)
        sig_v2_v3 = TestModels._zetterberg_sigma(v2 - v3, params)
        sig_v1 = TestModels._zetterberg_sigma(v1, params)
        sig_v4_v5 = TestModels._zetterberg_sigma(v4 - v5, params)
        return np.array(
            [
                y1,
                Heke * (params["gamma_1"] * sig_v2_v3 + params["gamma_1T"] * (params["U"] + coupled_input)) - ke_2 * y1 - keke * v1,
                y2,
                Heke * (params["gamma_2"] * sig_v1 + params["gamma_2T"] * (params["P"] + coupled_input)) - ke_2 * y2 - keke * v2,
                y3,
                Hiki * (params["gamma_4"] * sig_v4_v5) - ki_2 * y3 - kiki * v3,
                y4,
                Heke * (params["gamma_3"] * sig_v2_v3 + params["gamma_3T"] * (params["Q"] + coupled_input)) - ke_2 * y4 - keke * v4,
                y5,
                Hiki * (params["gamma_5"] * sig_v4_v5) - ki_2 * y5 - keke * v5,
                y2 - y3,
                y4 - y5,
            ]
        )

    @staticmethod
    def _montbrio_rhs(state, c_r, c_V, params):
        r, V = state
        tau = params["tau"]
        return np.array(
            [
                (params["Delta"] / (np.pi * tau) + 2.0 * V * r) / tau,
                (V * V - np.pi * np.pi * tau * tau * r * r + params["eta"] + params["J"] * tau * r + params["I"] + params["cr"] * c_r + params["cv"] * c_V) / tau,
            ]
        )

    @staticmethod
    def _coombes_byrne_rhs(state, c_r, params):
        r, V, g, q = state
        return np.array(
            [
                params["Delta"] / np.pi + 2.0 * V * r - g * r,
                V * V - np.pi * np.pi * r * r + params["eta"] + (params["v_syn"] - V) * g + c_r,
                params["alpha"] * q,
                params["alpha"] * (params["k"] * np.pi * r - g - 2.0 * q),
            ]
        )

    @staticmethod
    def _coombes_byrne2d_rhs(state, c_r, params):
        r, V = state
        return np.array(
            [
                params["Delta"] / np.pi + 2.0 * V * r - params["k"] * np.pi * r * r,
                V * V - np.pi * np.pi * r * r + params["eta"] + (params["v_syn"] - V) * params["k"] * np.pi * r + c_r,
            ]
        )

    @staticmethod
    def _gast_sd_rhs(state, c_r, c_V, params):
        r, V, A, B = state
        tau = params["tau"]
        return np.array(
            [
                (params["Delta"] / (np.pi * tau) + 2.0 * V * r) / tau,
                (V * V - np.pi * np.pi * tau * tau * r * r + params["eta"] + params["J"] * tau * r * (1.0 - A) + params["I"] + params["cr"] * c_r + params["cv"] * c_V) / tau,
                B / params["tau_A"],
                (-2.0 * B - A + params["alpha"] * r) / params["tau_A"],
            ]
        )

    @staticmethod
    def _gast_sf_rhs(state, c_r, c_V, params):
        r, V, A, B = state
        tau = params["tau"]
        return np.array(
            [
                (params["Delta"] / (np.pi * tau) + 2.0 * V * r) / tau,
                (V * V - np.pi * np.pi * tau * tau * r * r + params["eta"] + params["J"] * tau * r + params["I"] - A + params["cr"] * c_r + params["cv"] * c_V) / tau,
                B / params["tau_A"],
                (-2.0 * B - A + params["alpha"] * r) / params["tau_A"],
            ]
        )

    @staticmethod
    def _dumont_gutkin_rhs(state, c_r, params):
        r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii = state
        return np.array(
            [
                (params["Delta_e"] / (np.pi * params["tau_e"]) + 2.0 * V_e * r_e) / params["tau_e"],
                (V_e * V_e + params["eta_e"] - params["tau_e"] ** 2 * np.pi**2 * r_e * r_e + params["tau_e"] * s_ee - params["tau_e"] * s_ei + params["I_e"]) / params["tau_e"],
                (-s_ee + params["J_ee"] * r_e + c_r) / params["tau_s"],
                (-s_ei + params["J_ei"] * r_i) / params["tau_s"],
                (params["Delta_i"] / (np.pi * params["tau_i"]) + 2.0 * V_i * r_i) / params["tau_i"],
                (V_i * V_i + params["eta_i"] - params["tau_i"] ** 2 * np.pi**2 * r_i * r_i + params["tau_i"] * s_ie - params["tau_i"] * s_ii + params["I_i"]) / params["tau_i"],
                (-s_ie + params["J_ie"] * r_e + params["Gamma"] * c_r) / params["tau_s"],
                (-s_ii + params["J_ii"] * r_i) / params["tau_s"],
            ]
        )

    @staticmethod
    def _kionex_params(**overrides):
        params = {
            "E": 0.0,
            "K_bath": 5.5,
            "J": 0.1,
            "eta": 0.0,
            "Delta": 1.0,
            "c_minus": -40.0,
            "R_minus": 0.5,
            "c_plus": -20.0,
            "R_plus": -0.5,
            "Vstar": -31.0,
            "Cm": 1.0,
            "tau_n": 4.0,
            "gamma": 0.04,
            "epsilon": 0.001,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _kionex_rhs(state, coupling_term, params):
        x, V, n, DKi, Kg = state
        Cnap = 21.0
        DCnap = 2.0
        Ckp = 5.5
        DCkp = 1.0
        Cmna = -24.0
        DCmna = 12.0
        Cnk = -19.0
        DCnk = 18.0
        g_Cl = 7.5
        g_Na = 40.0
        g_K = 22.0
        g_Nal = 0.02
        g_Kl = 0.12
        rho = 250.0
        w_i = 2160.0
        w_o = 720.0
        Na_i0 = 16.0
        Na_o0 = 138.0
        K_i0 = 130.0
        K_o0 = 4.8
        Cl_i0 = 5.0
        Cl_o0 = 112.0
        beta = w_i / w_o
        DNa_i = -DKi
        DNa_o = -beta * DNa_i
        DK_o = -beta * DKi
        K_i = K_i0 + DKi
        Na_i = Na_i0 + DNa_i
        Na_o = Na_o0 + DNa_o
        K_o = K_o0 + DK_o + Kg
        minf = 1.0 / (1.0 + np.exp((Cmna - V) / DCmna))
        ninf = 1.0 / (1.0 + np.exp((Cnk - V) / DCnk))
        hgate = 1.1 - 1.0 / (1.0 + np.exp(-8.0 * (n - 0.4)))
        I_K = (g_Kl + g_K * n) * (V - 26.64 * np.log(K_o / K_i))
        I_Na = (g_Nal + g_Na * minf * hgate) * (V - 26.64 * np.log(Na_o / Na_i))
        I_Cl = g_Cl * (V + 26.64 * np.log(Cl_o0 / Cl_i0))
        I_pump = rho * (1.0 / (1.0 + np.exp((Cnap - Na_i) / DCnap))) * (1.0 / (1.0 + np.exp((Ckp - K_o) / DCkp)))
        rate = params["R_minus"] * x / np.pi
        Vdot = (-1.0 / params["Cm"]) * (I_Na + I_K + I_Cl + I_pump)
        if V <= params["Vstar"]:
            R = params["R_minus"]
            c = params["c_minus"]
        else:
            R = params["R_plus"]
            c = params["c_plus"]
        return np.array(
            [
                params["Delta"] + 2.0 * R * (V - c) * x - params["J"] * rate * x,
                Vdot - R * x * x + params["eta"] + (params["R_minus"] / np.pi) * coupling_term * (params["E"] - V),
                (ninf - n) / params["tau_n"],
                -(params["gamma"] / w_i) * (I_K - 2.0 * I_pump),
                params["epsilon"] * (params["K_bath"] - K_o),
            ]
        )

    def test_wilson_cowan(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._wilson_cowan_params()
        states = {"A": {"E": 0.2, "I": 0.1}, "B": {"E": 0.4, "I": 0.3}}
        coupling = {"A": {"c_0": 0.05}, "B": {"c_0": -0.02}}
        rois = self._two_roi_model(
            "tvb_wilson_cowan",
            ["E", "I"],
            states,
            coupling,
            params,
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dE_a, dI_a = self._wilson_cowan_rhs(
            states["A"]["E"],
            states["A"]["I"],
            coupling["A"]["c_0"],
            params,
        )
        dE_b, dI_b = self._wilson_cowan_rhs(
            states["B"]["E"],
            states["B"]["I"],
            coupling["B"]["c_0"],
            params,
        )
        expected = np.array(
            [
                [states["A"]["E"] + dt * dE_a, states["A"]["I"] + dt * dI_a],
                [states["B"]["E"] + dt * dE_b, states["B"]["I"] + dt * dI_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.2, 0.1], [0.4, 0.3]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_g2d(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._generic2d_params()
        states = {"A": {"V": 0.25, "W": -0.5}, "B": {"V": -0.75, "W": 0.4}}
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        rois = self._two_roi_network(
            "tvb_generic2d_oscillator",
            ["V", "W"],
            states,
            weights,
            "tvb_V_to_c0_macro2macro",
            params,
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dV_a, dW_a = self._generic2d_rhs(
            states["A"]["V"],
            states["A"]["W"],
            weights[0, 1] * states["B"]["V"],
            params,
        )
        dV_b, dW_b = self._generic2d_rhs(
            states["B"]["V"],
            states["B"]["W"],
            weights[1, 0] * states["A"]["V"],
            params,
        )
        expected = np.array(
            [
                [states["A"]["V"] + dt * dV_a, states["A"]["W"] + dt * dW_a],
                [states["B"]["V"] + dt * dV_b, states["B"]["W"] + dt * dW_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.25, -0.5], [-0.75, 0.4]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_jansen_rit(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._jansen_rit_params()
        outputs = ["y0", "y1", "y2", "y3", "y4", "y5"]
        states = {
            "A": {"y0": 0.02, "y1": 0.5, "y2": 0.1, "y3": 0.01, "y4": 0.03, "y5": 0.02},
            "B": {"y0": 0.04, "y1": 0.7, "y2": 0.2, "y3": 0.02, "y4": 0.04, "y5": 0.01},
        }
        coupling = {"A": {"c_0": 0.01}, "B": {"c_0": -0.02}}
        rois = self._two_roi_model("tvb_jansen_rit", outputs, states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        state_a = np.array([states["A"][output] for output in outputs])
        state_b = np.array([states["B"][output] for output in outputs])
        expected = np.array(
            [
                state_a + dt * self._jansen_rit_rhs(state_a, coupling["A"]["c_0"], params),
                state_b + dt * self._jansen_rit_rhs(state_b, coupling["B"]["c_0"], params),
            ]
        )

        np.testing.assert_allclose(cube[0], np.array([state_a, state_b]))
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_sj2d(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._fhn_params()
        outputs = ["xi0", "eta0", "alpha0", "beta0", "xi1", "eta1", "alpha1", "beta1", "xi2", "eta2", "alpha2", "beta2"]
        states = {
            "A": {"xi0": -0.4, "eta0": 0.2, "alpha0": 0.3, "beta0": -0.1, "xi1": 0.1, "eta1": -0.2, "alpha1": 0.25, "beta1": 0.15, "xi2": 0.35, "eta2": 0.05, "alpha2": -0.2, "beta2": 0.1},
            "B": {"xi0": -0.2, "eta0": 0.1, "alpha0": 0.15, "beta0": -0.05, "xi1": 0.2, "eta1": -0.1, "alpha1": 0.35, "beta1": 0.05, "xi2": 0.25, "eta2": 0.15, "alpha2": -0.1, "beta2": 0.2},
        }
        coupling = {"A": {"c_0": 0.03}, "B": {"c_0": -0.02}}
        rois = self._two_roi_model("tvb_reduced_set_fitzhugh_nagumo", outputs, states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        state_a = np.array([states["A"][output] for output in outputs])
        state_b = np.array([states["B"][output] for output in outputs])
        expected = np.array(
            [
                state_a + dt * self._fhn_rhs(state_a, coupling["A"]["c_0"], params),
                state_b + dt * self._fhn_rhs(state_b, coupling["B"]["c_0"], params),
            ]
        )

        np.testing.assert_allclose(cube[0], np.array([state_a, state_b]))
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-10)

    def test_sj3d(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._hr_params()
        outputs = [
            "xi0",
            "eta0",
            "tau0",
            "alpha0",
            "beta0",
            "gamma0",
            "xi1",
            "eta1",
            "tau1",
            "alpha1",
            "beta1",
            "gamma1",
            "xi2",
            "eta2",
            "tau2",
            "alpha2",
            "beta2",
            "gamma2",
        ]
        states = {
            "A": {"xi0": -0.3, "eta0": -1.0, "tau0": 4.5, "alpha0": 0.2, "beta0": -0.4, "gamma0": 3.0, "xi1": 0.1, "eta1": -0.5, "tau1": 4.0, "alpha1": 0.25, "beta1": -0.2, "gamma1": 3.2, "xi2": 0.35, "eta2": -0.25, "tau2": 4.2, "alpha2": -0.1, "beta2": 0.1, "gamma2": 2.8},
            "B": {"xi0": -0.2, "eta0": -0.8, "tau0": 4.3, "alpha0": 0.1, "beta0": -0.3, "gamma0": 2.9, "xi1": 0.2, "eta1": -0.4, "tau1": 3.9, "alpha1": 0.15, "beta1": -0.1, "gamma1": 3.1, "xi2": 0.25, "eta2": -0.15, "tau2": 4.1, "alpha2": -0.05, "beta2": 0.2, "gamma2": 2.7},
        }
        coupling = {"A": {"c_0": 0.04}, "B": {"c_0": -0.03}}
        rois = self._two_roi_model("tvb_reduced_set_hindmarsh_rose", outputs, states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        state_a = np.array([states["A"][output] for output in outputs])
        state_b = np.array([states["B"][output] for output in outputs])
        expected = np.array(
            [
                state_a + dt * self._hr_rhs(state_a, coupling["A"]["c_0"], params),
                state_b + dt * self._hr_rhs(state_b, coupling["B"]["c_0"], params),
            ]
        )

        np.testing.assert_allclose(cube[0], np.array([state_a, state_b]))
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-10)

    def test_reduced_wong_wang(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._reduced_wong_wang_params()
        states = {"A": {"S": 0.2}, "B": {"S": 0.45}}
        coupling = {"A": {"c_0": 0.05}, "B": {"c_0": -0.03}}
        rois = self._two_roi_model(
            "tvb_reduced_wong_wang",
            ["S"],
            states,
            coupling,
            params,
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dS_a = self._reduced_wong_wang_rhs(states["A"]["S"], coupling["A"]["c_0"], params)
        dS_b = self._reduced_wong_wang_rhs(states["B"]["S"], coupling["B"]["c_0"], params)
        expected = np.array(
            [
                [states["A"]["S"] + dt * dS_a],
                [states["B"]["S"] + dt * dS_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.2], [0.45]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_reduced_wong_wang_exc_inh(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._rww_exc_inh_params()
        states = {"A": {"S_e": 0.2, "S_i": 0.1}, "B": {"S_e": 0.35, "S_i": 0.15}}
        coupling = {"A": {"c_0": 0.05}, "B": {"c_0": -0.03}}
        rois = self._two_roi_model("tvb_reduced_wong_wang_exc_inh", ["S_e", "S_i"], states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dSe_a, dSi_a = self._rww_exc_inh_rhs(states["A"]["S_e"], states["A"]["S_i"], coupling["A"]["c_0"], params)
        dSe_b, dSi_b = self._rww_exc_inh_rhs(states["B"]["S_e"], states["B"]["S_i"], coupling["B"]["c_0"], params)
        expected = np.array(
            [
                [states["A"]["S_e"] + dt * dSe_a, states["A"]["S_i"] + dt * dSi_a],
                [states["B"]["S_e"] + dt * dSe_b, states["B"]["S_i"] + dt * dSi_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.2, 0.1], [0.35, 0.15]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_deco_balanced_exc_inh(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._rww_exc_inh_params(M_i=1.25)
        states = {"A": {"S_e": 0.18, "S_i": 0.11}, "B": {"S_e": 0.32, "S_i": 0.17}}
        coupling = {"A": {"c_0": 0.02}, "B": {"c_0": -0.01}}
        rois = self._two_roi_model("tvb_deco_balanced_exc_inh", ["S_e", "S_i"], states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dSe_a, dSi_a = self._deco_balanced_rhs(states["A"]["S_e"], states["A"]["S_i"], coupling["A"]["c_0"], params)
        dSe_b, dSi_b = self._deco_balanced_rhs(states["B"]["S_e"], states["B"]["S_i"], coupling["B"]["c_0"], params)
        expected = np.array(
            [
                [states["A"]["S_e"] + dt * dSe_a, states["A"]["S_i"] + dt * dSi_a],
                [states["B"]["S_e"] + dt * dSe_b, states["B"]["S_i"] + dt * dSi_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.18, 0.11], [0.32, 0.17]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_larter(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._larter_breakspear_params()
        states = {
            "A": {"V": 0.1, "W": 0.2, "Z": -0.1},
            "B": {"V": -0.2, "W": 0.25, "Z": 0.15},
        }
        coupling = {"A": {"c_0": 0.04}, "B": {"c_0": 0.08}}
        rois = self._two_roi_model(
            "tvb_larter_breakspear",
            ["V", "W", "Z"],
            states,
            coupling,
            params,
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dV_a, dW_a, dZ_a = self._larter_breakspear_rhs(
            states["A"]["V"],
            states["A"]["W"],
            states["A"]["Z"],
            coupling["A"]["c_0"],
            params,
        )
        dV_b, dW_b, dZ_b = self._larter_breakspear_rhs(
            states["B"]["V"],
            states["B"]["W"],
            states["B"]["Z"],
            coupling["B"]["c_0"],
            params,
        )
        expected = np.array(
            [
                [states["A"]["V"] + dt * dV_a, states["A"]["W"] + dt * dW_a, states["A"]["Z"] + dt * dZ_a],
                [states["B"]["V"] + dt * dV_b, states["B"]["W"] + dt * dW_b, states["B"]["Z"] + dt * dZ_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.1, 0.2, -0.1], [-0.2, 0.25, 0.15]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_epileptor(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._epileptor_params()
        outputs = ["x1", "y1", "z", "x2", "y2", "g"]
        states = {
            "A": {"x1": -1.2, "y1": -0.5, "z": 3.2, "x2": -0.4, "y2": 0.2, "g": 0.1},
            "B": {"x1": 0.2, "y1": -0.1, "z": -0.2, "x2": 0.1, "y2": 0.3, "g": -0.2},
        }
        coupling = {
            "A": {"c_pop1": 0.05, "c_pop2": -0.02},
            "B": {"c_pop1": -0.03, "c_pop2": 0.04},
        }
        rois = self._two_roi_model("tvb_epileptor", outputs, states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        state_a = np.array([states["A"][output] for output in outputs])
        state_b = np.array([states["B"][output] for output in outputs])
        expected = np.array(
            [
                state_a
                + dt
                * self._epileptor_rhs(
                    state_a,
                    coupling["A"]["c_pop1"],
                    coupling["A"]["c_pop2"],
                    params,
                ),
                state_b
                + dt
                * self._epileptor_rhs(
                    state_b,
                    coupling["B"]["c_pop1"],
                    coupling["B"]["c_pop2"],
                    params,
                ),
            ]
        )

        np.testing.assert_allclose(cube[0], np.array([state_a, state_b]))
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_zetterberg_jansen(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._zetterberg_params()
        outputs = ["v1", "y1", "v2", "y2", "v3", "y3", "v4", "y4", "v5", "y5", "v6", "v7"]
        states = {
            "A": {"v1": 0.1, "y1": 0.02, "v2": 0.2, "y2": 0.03, "v3": 0.05, "y3": -0.01, "v4": 0.15, "y4": 0.02, "v5": 0.04, "y5": -0.02, "v6": 0.1, "v7": 0.05},
            "B": {"v1": 0.2, "y1": 0.01, "v2": 0.1, "y2": 0.04, "v3": 0.02, "y3": -0.02, "v4": 0.18, "y4": 0.01, "v5": 0.06, "y5": -0.01, "v6": 0.12, "v7": 0.04},
        }
        coupling = {"A": {"c_0": 0.02}, "B": {"c_0": -0.03}}
        rois = self._two_roi_model("tvb_zetterberg_jansen", outputs, states, coupling, params)

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        state_a = np.array([states["A"][output] for output in outputs])
        state_b = np.array([states["B"][output] for output in outputs])
        expected = np.array(
            [
                state_a + dt * self._zetterberg_rhs(state_a, coupling["A"]["c_0"], params),
                state_b + dt * self._zetterberg_rhs(state_b, coupling["B"]["c_0"], params),
            ]
        )

        np.testing.assert_allclose(cube[0], np.array([state_a, state_b]))
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_infinite_theta(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        cases = [
            (
                "tvb_montbrio_pazo_roxin",
                ["r", "V"],
                {"tau": 1.0, "I": 0.0, "Delta": 1.0, "J": 15.0, "eta": -5.0, "Gamma": 0.0, "cr": 1.0, "cv": 0.0},
                {"A": {"r": 0.2, "V": -0.5}, "B": {"r": 0.35, "V": -0.2}},
                {"A": {"c_r": 0.04, "c_V": 0.01}, "B": {"c_r": -0.02, "c_V": 0.03}},
                lambda state, coupling, params: self._montbrio_rhs(state, coupling["c_r"], coupling["c_V"], params),
            ),
            (
                "tvb_coombes_byrne",
                ["r", "V", "g", "q"],
                {"Delta": 0.5, "alpha": 0.95, "v_syn": -10.0, "k": 1.0, "eta": 20.0},
                {"A": {"r": 0.5, "V": -1.0, "g": 1.2, "q": 0.1}, "B": {"r": 0.8, "V": -0.7, "g": 1.4, "q": -0.1}},
                {"A": {"c_r": 0.02}, "B": {"c_r": -0.01}},
                lambda state, coupling, params: self._coombes_byrne_rhs(state, coupling["c_r"], params),
            ),
            (
                "tvb_coombes_byrne2d",
                ["r", "V"],
                {"Delta": 1.0, "v_syn": -4.0, "k": 1.0, "eta": 2.0},
                {"A": {"r": 0.3, "V": -0.4}, "B": {"r": 0.6, "V": -0.2}},
                {"A": {"c_r": 0.03}, "B": {"c_r": -0.02}},
                lambda state, coupling, params: self._coombes_byrne2d_rhs(state, coupling["c_r"], params),
            ),
            (
                "tvb_gast_schmidt_knosche_sd",
                ["r", "V", "A", "B"],
                {"tau": 1.0, "tau_A": 10.0, "alpha": 0.5, "I": 0.0, "Delta": 2.0, "J": 21.2132, "eta": -6.0, "cr": 1.0, "cv": 0.0},
                {"A": {"r": 0.5, "V": -1.0, "A": 0.1, "B": 0.05}, "B": {"r": 0.7, "V": -0.8, "A": 0.2, "B": -0.03}},
                {"A": {"c_r": 0.02, "c_V": 0.01}, "B": {"c_r": -0.01, "c_V": 0.03}},
                lambda state, coupling, params: self._gast_sd_rhs(state, coupling["c_r"], coupling["c_V"], params),
            ),
            (
                "tvb_gast_schmidt_knosche_sf",
                ["r", "V", "A", "B"],
                {"tau": 1.0, "tau_A": 10.0, "alpha": 10.0, "I": 0.0, "Delta": 2.0, "J": 21.2132, "eta": 1.0, "cr": 1.0, "cv": 0.0},
                {"A": {"r": 0.4, "V": -0.6, "A": 0.1, "B": 0.02}, "B": {"r": 0.8, "V": -0.4, "A": -0.1, "B": -0.04}},
                {"A": {"c_r": 0.01, "c_V": 0.02}, "B": {"c_r": -0.02, "c_V": 0.01}},
                lambda state, coupling, params: self._gast_sf_rhs(state, coupling["c_r"], coupling["c_V"], params),
            ),
            (
                "tvb_dumont_gutkin",
                ["r_e", "V_e", "s_ee", "s_ei", "r_i", "V_i", "s_ie", "s_ii"],
                {"I_e": 0.0, "Delta_e": 1.0, "eta_e": -5.0, "tau_e": 10.0, "I_i": 0.0, "Delta_i": 1.0, "eta_i": -5.0, "tau_i": 10.0, "tau_s": 1.0, "J_ee": 0.0, "J_ei": 10.0, "J_ie": 0.0, "J_ii": 15.0, "Gamma": 5.0},
                {"A": {"r_e": 0.2, "V_e": -0.6, "s_ee": 0.1, "s_ei": 0.05, "r_i": 0.25, "V_i": -0.7, "s_ie": 0.08, "s_ii": 0.03}, "B": {"r_e": 0.3, "V_e": -0.5, "s_ee": 0.12, "s_ei": 0.04, "r_i": 0.2, "V_i": -0.8, "s_ie": 0.06, "s_ii": 0.02}},
                {"A": {"c_r": 0.02}, "B": {"c_r": -0.01}},
                lambda state, coupling, params: self._dumont_gutkin_rhs(state, coupling["c_r"], params),
            ),
        ]

        for model, outputs, params, states, coupling, rhs in cases:
            rois = self._two_roi_model(model, outputs, states, coupling, params)
            cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
            state_a = np.array([states["A"][output] for output in outputs])
            state_b = np.array([states["B"][output] for output in outputs])
            expected = np.array(
                [
                    state_a + dt * rhs(state_a, coupling["A"], params),
                    state_b + dt * rhs(state_b, coupling["B"], params),
                ]
            )
            np.testing.assert_allclose(cube[0], np.array([state_a, state_b]), err_msg=model)
            np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-11, err_msg=model)

    def test_kionex_numpy_numba_equivalence(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        params = self._kionex_params()
        labels = ["R0", "R1", "R2", "R3"]
        outputs = ["x", "V", "n", "DKi", "Kg"]
        states = {
            "R0": {"x": 0.2, "V": -50.0, "n": 0.3, "DKi": -0.5, "Kg": 0.1},
            "R1": {"x": 0.4, "V": -40.0, "n": 0.5, "DKi": -0.4, "Kg": 0.2},
            "R2": {"x": 0.6, "V": -25.0, "n": 0.7, "DKi": -0.3, "Kg": 0.3},
            "R3": {"x": 0.8, "V": -20.0, "n": 0.6, "DKi": -0.2, "Kg": 0.4},
        }
        coupling_values = {"R0": 0.2, "R1": 0.4, "R2": 0.6, "R3": 0.8}
        rois = ms.macro.load_rois(
            labels=labels,
            weights=np.zeros((4, 4)),
            delays=np.zeros((4, 4)),
        )
        for roi in rois.rois():
            roi_params = dict(params)
            roi_params["c_0"] = coupling_values[roi.label]
            roi.use_macro("tvb_k_ion_exchange", initial_state=states[roi.label], params=roi_params)
            for output in outputs:
                roi.record(output)
        history = np.array(
            [[[states[label][output] for label in labels] for output in outputs]],
            dtype=float,
        )
        rois.initial_history(history, outputs=outputs, layout="time_output_roi")

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        initial = np.array([[states[label][output] for output in outputs] for label in labels])
        expected = np.array(
            [
                initial[index] + dt * self._kionex_rhs(initial[index], coupling_values[label], params)
                for index, label in enumerate(labels)
            ]
        )

        np.testing.assert_allclose(cube[0], initial)
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-10)

    def test_hopfield(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        states = {"A": {"x": 0.25, "theta": 0.1}, "B": {"x": -0.5, "theta": 0.4}}
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        rois = self._two_roi_network(
            "tvb_hopfield",
            ["x", "theta"],
            states,
            weights,
            "tvb_x_to_c_macro2macro",
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dx_a, dtheta_a = self._hopfield_rhs(states["A"]["x"], weights[0, 1] * states["B"]["x"])
        dx_b, dtheta_b = self._hopfield_rhs(states["B"]["x"], weights[1, 0] * states["A"]["x"])
        expected = np.array(
            [
                [states["A"]["x"] + dt * dx_a, states["A"]["theta"] + dt * dtheta_a],
                [states["B"]["x"] + dt * dx_b, states["B"]["theta"] + dt * dtheta_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.25, 0.1], [-0.5, 0.4]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_kuramoto(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        states = {"A": {"theta": 0.25}, "B": {"theta": -0.5}}
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        rois = self._two_roi_network(
            "tvb_kuramoto",
            ["theta"],
            states,
            weights,
            "tvb_theta_to_c_macro2macro",
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dtheta_a = self._kuramoto_rhs(states["A"]["theta"], weights[0, 1] * states["B"]["theta"])
        dtheta_b = self._kuramoto_rhs(states["B"]["theta"], weights[1, 0] * states["A"]["theta"])
        expected = np.array(
            [
                [states["A"]["theta"] + dt * dtheta_a],
                [states["B"]["theta"] + dt * dtheta_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.25], [-0.5]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_linear(self, macro_mod_dir):
        dt = 0.1
        _configure_macro_runtime(macro_mod_dir, dt)

        states = {"A": {"x": 0.25}, "B": {"x": -0.5}}
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        rois = self._two_roi_network(
            "tvb_linear",
            ["x"],
            states,
            weights,
            "tvb_x_to_c_macro2macro",
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        dx_a = self._linear_rhs(states["A"]["x"], weights[0, 1] * states["B"]["x"])
        dx_b = self._linear_rhs(states["B"]["x"], weights[1, 0] * states["A"]["x"])
        expected = np.array(
            [
                [states["A"]["x"] + dt * dx_a],
                [states["B"]["x"] + dt * dx_b],
            ]
        )

        np.testing.assert_allclose(cube[0], [[0.25], [-0.5]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)

    def test_ww(self, macro_mod_dir):
        self.test_reduced_wong_wang(macro_mod_dir)
        self.test_reduced_wong_wang_exc_inh(macro_mod_dir)
