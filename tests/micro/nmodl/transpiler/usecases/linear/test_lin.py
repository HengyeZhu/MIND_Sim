from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def lin_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind-nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def single_section_sim(mod_dir):
    ms.load_mech(mod_dir)
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])
    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    return sim, group, group[0](0.5)


def test_lin(lin_mod_dir):
    sim, group, seg = single_section_sim(lin_mod_dir)
    group.insert("lin")
    sim.build_microcircuit()
    sim.finitialize(-65.0)

    expected = np.linalg.solve(np.array([[2.0, 3.0], [4.0, 5.0]]), [0, 0])

    assert seg._ref_xx_lin.value() == expected[0]
    assert seg._ref_yy_lin.value() == expected[1]
