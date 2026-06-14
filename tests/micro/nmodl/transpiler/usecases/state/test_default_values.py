from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def default_values_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def single_section_sim(mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)
    sim.load_mech(str(mod_dir))

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


def test_default_values(default_values_mod_dir):
    sim, group, seg = single_section_sim(default_values_mod_dir)
    group.insert("default_values")
    sim.build_microcircuit()
    sim.finitialize(-65.0)

    assert seg._ref_X_default_values.value() == 2.0
    assert seg._ref_Y_default_values.value() == 0.0
    assert seg._ref_Z_default_values.value() == 7.0

    for index in range(3):
        assert seg._ref_A_default_values[index].value() == 4.0

    assert seg._ref_B_default_values[0].value() == 5.0
    assert seg._ref_B_default_values[1].value() == 8.0
