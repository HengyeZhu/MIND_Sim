from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parents[1] / "mod"


@pytest.fixture(scope="session")
def micro_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def test_version_macros(micro_mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)
    sim.load_mech(str(micro_mod_dir))

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("VersionMacros")

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    sim.fadvance()
    seg = group[0](0.5)

    assert seg._ref_eq8_2_0_result_VersionMacros.value() == 0.0
    assert seg._ref_ne9_0_1_result_VersionMacros.value() == 0.0
    assert seg._ref_gt9_0_0_result_VersionMacros.value() == 1.0
    assert seg._ref_lt42_1_2_result_VersionMacros.value() == 1.0
    assert seg._ref_gteq10_4_7_result_VersionMacros.value() == 0.0
    assert seg._ref_lteq8_1_0_result_VersionMacros.value() == 0.0
    assert seg._ref_explicit_gteq8_2_0_result_VersionMacros.value() == 1.0
