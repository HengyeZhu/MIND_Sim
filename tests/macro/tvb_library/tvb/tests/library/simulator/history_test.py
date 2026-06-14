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


def _build_sum_rois(labels, weights, delays=None):
    if delays is None:
        delays = np.zeros_like(np.asarray(weights, dtype=float))
    rois = ms.macro.load_rois(labels=labels, weights=weights, delays=delays)
    for roi in rois.rois():
        roi.use_macro("tvb_sum1d", initial_state={"x": 1.0})
        roi.record("x")
    return rois


class TestsExactPropagation:
    def build_simulator(self, macro_mod_dir, n=4):
        _configure_macro_runtime(macro_mod_dir, 1.0)

        weights = np.zeros((n, n))
        for target in range(n - 1):
            weights[target, target + 1] = 1.0
        delays = np.arange(n * n, dtype=float).reshape((n, n))
        rois = _build_sum_rois(labels=[f"R{i}" for i in range(n)], weights=weights, delays=delays)
        for target in range(n - 1):
            rois.roi(f"R{target}").insert(f"R{target + 1}", "vep_x_macro2macro")
        rois.initial_history(np.ones((n * n, 1, n)), outputs=["x"], layout="time_output_roi")
        return rois

    def test_propagation(self, macro_mod_dir):
        n = 4
        rois = self.build_simulator(macro_mod_dir, n=n)
        xs = _records_cube(ms.macro.Simulator(rois).run(10.0))[1:, :, 0]
        expected = np.array(
            [
                [2.0, 2.0, 2.0, 1.0],
                [3.0, 3.0, 3.0, 1.0],
                [5.0, 4.0, 4.0, 1.0],
                [8.0, 5.0, 5.0, 1.0],
                [12.0, 6.0, 6.0, 1.0],
                [17.0, 7.0, 7.0, 1.0],
                [23.0, 8.0, 8.0, 1.0],
                [30.0, 10.0, 9.0, 1.0],
                [38.0, 13.0, 10.0, 1.0],
                [48.0, 17.0, 11.0, 1.0],
            ]
        )
        assert np.allclose(xs, expected, rtol=0.0, atol=1.0e-12)
