from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def top_local_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def test_top_local(top_local_mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(2)
    sim.set_dt(0.025)
    sim.load_mech(str(top_local_mod_dir))

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 2, "sections": [soma]}])

    segments = []
    for cell in sim.population("CELL"):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("top_local")
        segments.append(group[0](0.5))

    sim.build_microcircuit()
    t = ms.Vector().record(sim._ref_t)
    y0 = ms.Vector().record(segments[0]._ref_y_top_local)
    y1 = ms.Vector().record(segments[1]._ref_y_top_local)

    sim.finitialize(-65.0)
    sim.run(1.0)

    time = np.asarray(t.to_python())
    values0 = np.asarray(y0.to_python())
    values1 = np.asarray(y1.to_python())

    early = time < 0.33
    late = 0.33 <= time

    assert np.all(values0[early] == 2.0)
    assert np.all(values0[late] == 3.0)
    assert np.all(values1[early] == 2.0)
    assert np.all(values1[late] == 3.0)
