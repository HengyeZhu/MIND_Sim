from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def finite_difference_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def test_finite_difference(finite_difference_mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.001)
    sim.load_mech(str(finite_difference_mod_dir))

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("finite_difference")

    sim.build_microcircuit()
    seg = group[0](0.5)
    x = ms.Vector().record(seg._ref_x_finite_difference)
    z = ms.Vector().record(seg._ref_z_finite_difference)
    t = ms.Vector().record(sim._ref_t)

    sim.finitialize(-65.0)
    sim.run(5.0)

    time = np.asarray(t.to_python())
    x_actual = np.asarray(x.to_python())
    z_actual = np.asarray(z.to_python())
    a = 0.1

    np.testing.assert_allclose(x_actual, 42.0 * np.exp(-a * time), rtol=2.0e-4)
    np.testing.assert_allclose(z_actual, 21.0 * np.exp(-2.0 * a * time), rtol=2.0e-4)
