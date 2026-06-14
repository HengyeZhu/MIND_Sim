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


class TestMonitors:
    def test_monitor_raw(self, macro_mod_dir):
        _configure_macro_runtime(macro_mod_dir, 0.1)

        rois = ms.macro.load_rois(
            labels=["A", "B"],
            weights=[[0.0, 0.0], [0.0, 0.0]],
            delays=[[0.0, 0.0], [0.0, 0.0]],
        )
        for roi in rois.rois():
            roi.use_macro("tvb_epileptor2d", initial_state={"x": -1.0, "z": 0.0})
            roi.record("x")
            roi.record("z")
        rois.initial_history(np.array([[[-1.0, -1.0], [0.0, 0.0]]]), outputs=["x", "z"])

        result = ms.macro.Simulator(rois).run(n_steps=2)
        cube = _records_cube(result)

        assert result.times == [0.0, 0.1, 0.2]
        assert cube.shape == (3, 2, 2)
        assert np.allclose(cube[0, :, 0], [-1.0, -1.0])
        assert np.allclose(cube[0, :, 1], [0.0, 0.0])
