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
        subprocess.run(["mind-nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def _configure_macro_runtime(macro_mod_dir, dt):
    ms.load_mech(macro_mod_dir)
    ms.macro.dt(dt)
    ms.macro.exchange_window(dt)


def _build_sum_rois(labels, weights, delays=None):
    if delays is None:
        delays = np.zeros_like(np.asarray(weights, dtype=float))
    rois = ms.macro.load_rois(labels=labels, weights=weights, delays=delays)
    for roi in rois:
        roi.use_macro("tvb_sum1d", initial_state={"x": 1.0})
        roi.record("x")
    return rois


def _build_chain_rois():
    rois = _build_sum_rois(
        labels=["A", "B"],
        weights=[[0.0, 1.0], [0.0, 0.0]],
        delays=[[0.0, 0.0], [0.0, 0.0]],
    )
    rois.roi("A").insert("B", "vep_x_macro2macro")
    for roi in rois:
        roi.initial_history(np.array([[1.0]]), outputs=["x"])
    return rois


class TestStep:
    dt = 0.1

    def _sim(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, self.dt)
        return _build_chain_rois()

    def test_simulation_length(self, macro_mod_dir):
        for t_stop in np.arange(0.1, 10.0, 0.1):
            result = ms.macro.Simulator(self._sim(macro_mod_dir)).run(float(t_stop))
            assert result.times[-1] > t_stop or np.isclose(result.times[-1], t_stop)

    def test_n_steps(self, macro_mod_dir):
        for n_steps in range(1, 10):
            result = ms.macro.Simulator(self._sim(macro_mod_dir)).run(n_steps=n_steps)
            for i, time in enumerate(result.times[1:]):
                assert np.isclose(time, (i + 1) * self.dt)
            assert i == n_steps - 1

    def test_n_steps_type(self, macro_mod_dir):
        with pytest.raises(TypeError) as context:
            ms.macro.Simulator(self._sim(macro_mod_dir)).run(n_steps=1.1)
        assert "n_steps" in str(context.value)

    def test_macro2macro_target_current_reads_snapshot(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        rois = ms.macro.load_rois(
            labels=["A", "B"],
            weights=[[0.0, 1.0], [0.0, 0.0]],
            delays=[[0.0, 0.0], [0.0, 0.0]],
        )
        rois.roi("A").use_macro("tvb_linear", initial_state={"x": 0.0}, params={"gamma": 0.0})
        rois.roi("B").use_macro("tvb_linear", initial_state={"x": 3.0}, params={"gamma": 0.0})
        rois.roi("A").insert("B", "mind_target_current_macro2macro")
        rois.roi("A").record("c")
        rois.roi("A").record("x")
        rois.roi("A").initial_history(np.array([[0.0, 2.0]]), outputs=["x", "c"])
        rois.roi("B").initial_history(np.array([[3.0, 0.0]]), outputs=["x", "c"])

        result = ms.macro.Simulator(rois).run(n_steps=1)
        values = np.asarray(result.records.values).reshape(
            result.records.sample_count,
            result.records.recorded_roi_count,
            result.records.output_count,
        )

        np.testing.assert_allclose(values[0, 0], [5.0, 0.0])
        np.testing.assert_allclose(values[1, 0], [5.0, 5.0])
