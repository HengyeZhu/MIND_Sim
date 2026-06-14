from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np
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


def _single_cell_sim(*, nseg: int = 1, length: float = 10.0, diam: float = 10.0):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.nseg = nseg
    soma.L_um = length
    soma.diam_um = diam
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    return sim, cell, group


def test_units(micro_mod_dir):
    sim, _, soma = _single_cell_sim()
    sim.load_mech(str(micro_mod_dir))
    pp = soma[0](0.5).insert("UnitsTest")

    sim.build_microcircuit()
    sim.finitialize(-65.0)

    assert np.allclose(
        [
            pp._ref_mole.value(),
            pp._ref_e.value(),
            pp._ref_faraday.value(),
            pp._ref_planck.value(),
            pp._ref_hbar.value(),
            pp._ref_gasconst.value(),
            pp._ref_gasconst_exact.value(),
            pp._ref_avogadro.value(),
            pp._ref_k.value(),
            pp._ref_erev.value(),
            pp._ref_ghk.value(),
        ],
        [
            6.02214076e23,
            1.602176634e-19,
            96485.33212331001,
            6.62607015e-34,
            1.0545718176461565e-34,
            8.31446261815324,
            8.313424,
            6.02214076e23,
            1.380649e-23,
            50.0,
            -8141.347452141084,
        ],
        rtol=0.0,
        atol=1.0e-12,
    )
