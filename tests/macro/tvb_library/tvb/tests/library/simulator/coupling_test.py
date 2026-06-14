from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

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


def _build_rois(model, outputs, states, labels, weights, delays=None, params=None, history_outputs=None):
    if delays is None:
        delays = np.zeros_like(np.array(weights))
    rois = ms.macro.load_rois(labels=labels, weights=weights, delays=delays)
    for roi in rois.rois():
        roi.use_macro(model, initial_state=states[roi.label], params=params or {})
        for output in outputs:
            roi.record(output)
    if history_outputs is None:
        history_outputs = outputs
    history = np.array(
        [[[states[label][output] for label in labels] for output in history_outputs]],
        dtype=float,
    )
    rois.initial_history(history, outputs=history_outputs, layout="time_output_roi")
    return rois


class TestCoupling:
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    state_1sv = np.array([1.0, 2.0])
    state_2sv = np.array([[1.0, 2.0], [1.0, 2.0]])

    @staticmethod
    def _insert_pair(rois, mechanism, params=None):
        rois.roi("A").insert("B", mechanism, params=params)
        rois.roi("B").insert("A", mechanism, params=params)

    @staticmethod
    def _weighted_sum(weights, values):
        return np.array(
            [
                weights[0, 0] * values[0] + weights[0, 1] * values[1],
                weights[1, 0] * values[0] + weights[1, 1] * values[1],
            ]
        )

    def test_linear_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        a = 0.00390625
        states = {"A": {"x": self.state_1sv[0]}, "B": {"x": self.state_1sv[1]}}
        rois = _build_rois(
            "tvb_sum1d",
            ["x"],
            states,
            labels=["A", "B"],
            weights=self.weights,
            delays=np.zeros_like(self.weights),
        )
        self._insert_pair(rois, "vep_x_macro2macro", {"a": a})

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        expected_coupling = a * self._weighted_sum(self.weights, self.state_1sv)

        np.testing.assert_allclose(cube[0, :, 0], self.state_1sv)
        np.testing.assert_allclose(cube[1, :, 0], self.state_1sv + expected_coupling)

    def test_scaling_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        a = 0.00390625
        states = {"A": {"x": self.state_1sv[0]}, "B": {"x": self.state_1sv[1]}}
        rois = _build_rois("tvb_sum1d", ["x"], states, ["A", "B"], self.weights)
        self._insert_pair(rois, "vep_x_macro2macro", {"a": a})

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        expected_coupling = a * self._weighted_sum(self.weights, self.state_1sv)

        np.testing.assert_allclose(cube[0, :, 0], self.state_1sv)
        np.testing.assert_allclose(cube[1, :, 0], self.state_1sv + expected_coupling)

    def test_difference_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        a = 0.1
        states = {"A": {"x": self.state_1sv[0]}, "B": {"x": self.state_1sv[1]}}
        rois = _build_rois("tvb_linear", ["x"], states, ["A", "B"], self.weights, params={"gamma": 0.0})
        self._insert_pair(rois, "tvb_difference_macro2macro", {"a": a})

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        expected_coupling = np.array(
            [
                a * (
                    self.weights[0, 0] * (self.state_1sv[0] - self.state_1sv[0])
                    + self.weights[0, 1] * (self.state_1sv[1] - self.state_1sv[0])
                ),
                a * (
                    self.weights[1, 0] * (self.state_1sv[0] - self.state_1sv[1])
                    + self.weights[1, 1] * (self.state_1sv[1] - self.state_1sv[1])
                ),
            ]
        )

        np.testing.assert_allclose(cube[0, :, 0], self.state_1sv)
        np.testing.assert_allclose(cube[1, :, 0], self.state_1sv + expected_coupling)

    def test_hyperbolic_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        a = 1.0
        b = 1.0
        midpoint = 0.0
        sigma = 1.0
        states = {"A": {"x": self.state_1sv[0]}, "B": {"x": self.state_1sv[1]}}
        rois = _build_rois("tvb_linear", ["x"], states, ["A", "B"], self.weights, params={"gamma": 0.0})
        self._insert_pair(
            rois,
            "tvb_hyperbolic_tangent_macro2macro",
            {"a": a, "b": b, "midpoint": midpoint, "sigma": sigma},
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        pre = a * (1.0 + np.tanh((b * self.state_1sv - midpoint) / sigma))
        expected_coupling = self._weighted_sum(self.weights, pre)

        np.testing.assert_allclose(cube[0, :, 0], self.state_1sv)
        np.testing.assert_allclose(cube[1, :, 0], self.state_1sv + expected_coupling)

    def test_kuramoto_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        a = 1.0
        theta = np.array([0.25, -0.5])
        states = {"A": {"theta": theta[0]}, "B": {"theta": theta[1]}}
        rois = _build_rois("tvb_kuramoto", ["theta"], states, ["A", "B"], self.weights, params={"omega": 0.0})
        self._insert_pair(rois, "tvb_kuramoto_macro2macro", {"a": a})

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        expected_coupling = np.array(
            [
                a * (
                    self.weights[0, 0] * np.sin(theta[0] - theta[0])
                    + self.weights[0, 1] * np.sin(theta[1] - theta[0])
                ),
                a * (
                    self.weights[1, 0] * np.sin(theta[0] - theta[1])
                    + self.weights[1, 1] * np.sin(theta[1] - theta[1])
                ),
            ]
        )

        np.testing.assert_allclose(cube[0, :, 0], theta)
        np.testing.assert_allclose(cube[1, :, 0], theta + expected_coupling)

    def test_sigmoidal_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        cmin = -1.0
        cmax = 1.0
        midpoint = 0.0
        sigma = 230.0
        a = 1.0
        states = {"A": {"x": self.state_1sv[0]}, "B": {"x": self.state_1sv[1]}}
        rois = _build_rois(
            "tvb_sigmoidal_sum1d",
            ["x"],
            states,
            ["A", "B"],
            self.weights,
            params={"cmin": cmin, "cmax": cmax, "midpoint": midpoint, "a": a, "sigma": sigma},
        )
        self._insert_pair(rois, "tvb_x_to_c_raw_macro2macro")

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        gx = self._weighted_sum(self.weights, self.state_1sv)
        expected_coupling = cmin + ((cmax - cmin) / (1.0 + np.exp(-a * ((gx - midpoint) / sigma))))

        np.testing.assert_allclose(cube[0, :, 0], self.state_1sv)
        np.testing.assert_allclose(cube[1, :, 0], self.state_1sv + expected_coupling)

    def test_sigmoidal_jr_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        cmin = 0.0
        cmax = 2.0 * 0.0025
        midpoint = 6.0
        r = 0.56
        a = 1.0
        states = {
            "A": {"y1": self.state_2sv[0, 0], "y2": self.state_2sv[1, 0], "x": 0.0},
            "B": {"y1": self.state_2sv[0, 1], "y2": self.state_2sv[1, 1], "x": 0.0},
        }
        rois = _build_rois(
            "tvb_two_state_sum1d",
            ["x"],
            states,
            ["A", "B"],
            self.weights,
            history_outputs=["y1", "y2", "x"],
        )
        self._insert_pair(
            rois,
            "tvb_y1_y2_sigmoidal_jr_macro2macro",
            {"cmin": cmin, "cmax": cmax, "midpoint": midpoint, "r": r, "a": a},
        )

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        pre = cmin + (cmax - cmin) / (1.0 + np.exp(r * (midpoint - (self.state_2sv[0] - self.state_2sv[1]))))
        expected_coupling = a * self._weighted_sum(self.weights, pre)

        np.testing.assert_allclose(cube[0, :, 0], [0.0, 0.0])
        np.testing.assert_allclose(cube[1, :, 0], expected_coupling)

    def test_pre_sigmoidal_coupling(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        H = 0.5
        Q = 1.0
        G = 60.0
        P = 1.0
        states = {
            "A": {"x": 0.2, "y": 0.1},
            "B": {"x": 0.4, "y": 0.3},
        }
        rois = _build_rois(
            "tvb_presigmoidal_sum2d",
            ["x", "y"],
            states,
            ["A", "B"],
            self.weights,
            params={"H": H, "Q": Q, "G": G, "P": P},
        )
        self._insert_pair(rois, "tvb_presigmoidal_macro2macro", {"H": H, "Q": Q, "G": G, "P": P})

        cube = _records_cube(ms.macro.Simulator(rois).run(n_steps=1))
        x = np.array([states["A"]["x"], states["B"]["x"]])
        y = np.array([states["A"]["y"], states["B"]["y"]])
        activity = H * (Q + np.tanh(G * (P * x - y)))
        expected_c0 = self._weighted_sum(self.weights, activity)
        expected = np.column_stack([x + expected_c0, y + activity])

        np.testing.assert_allclose(cube[0], [[0.2, 0.1], [0.4, 0.3]])
        np.testing.assert_allclose(cube[1], expected, rtol=0.0, atol=1.0e-12)
