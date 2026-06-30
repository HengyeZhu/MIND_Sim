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
        subprocess.run(["mind-nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def _run_watch(micro_mod_dir, mechanism, thread_count, cell_count=4):
    ms.load_mech(micro_mod_dir)
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(thread_count)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": cell_count, "sections": [soma]}])

    processes = []
    for cell in sim.population("CELL"):
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        processes.append(group[0](0.5).insert(mechanism))

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    sim.run(5.0)
    return [
        [
            pp._ref_n_high.value(),
            pp._ref_n_mid.value(),
            pp._ref_n_low.value(),
            pp._ref_x.value(),
            pp._ref_r.value(),
            pp._ref_t1.value(),
        ]
        for pp in processes
    ]


def test_watchrange(micro_mod_dir):
    expected_random_watch = [4.0, 6.0, 4.0, 3.0, 0.8437286678599826, 4.900000000000036]
    for thread_count in (1, 2, 4):
        assert _run_watch(micro_mod_dir, "Bounce2", thread_count) == [expected_random_watch] * 4


def test_watchrange2(micro_mod_dir):
    assert _run_watch(micro_mod_dir, "Bounce", 1, cell_count=1) == [[0.0, 1.0, 0.0, 2.0, 0.5, 5.0]]
