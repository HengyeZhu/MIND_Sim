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


def test_array_variable_transfer(micro_mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.001)
    sim.load_mech(str(micro_mod_dir))

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    soma.nseg = 3
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("red")
    group.insert("green")

    sim.build_microcircuit()
    for tau in (0.25, 0.5, 0.75):
        group[0](tau)._ref_tau_green.set(tau)
    sim.finitialize(-65.0)

    t_vector = ms.Vector().record(sim._ref_t)
    record_index = [0, 15]
    taus = [0.25, 0.5, 0.75]
    cases = list(itertools.product(record_index, taus, [("green", 0.02), ("red", 1.0e-10)]))
    traces = []
    for array_index, tau, (mechanism, _) in cases:
        traces.append(ms.Vector().record(getattr(group[0](tau), f"_ref_upsilon_{mechanism}")[array_index]))

    sim.run(6.0)

    times = np.array(t_vector.to_python())
    assert times.size > 0

    for trace, (array_index, tau, (mechanism, tolerance)) in zip(traces, cases):
        observed = np.array(trace.to_python())
        if mechanism == "red":
            expected = np.sin((array_index + 1.0) * times)
        else:
            expected = 2.0 + 8.0 * tau + np.sin((array_index / 3.0 + 1.0) * times)
        abs_error = np.max(np.abs(observed - expected))
        rel_error = abs_error / np.max(np.abs(expected))
        assert rel_error < tolerance
