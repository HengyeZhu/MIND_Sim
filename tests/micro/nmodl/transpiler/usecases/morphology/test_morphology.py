from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent / "mod"


def mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind-nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def _run_single_section(diameter_um):
    tstop = 0.01
    ms.load_mech(mod_dir())
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.001)

    soma = ms.section("soma", "soma")
    soma.nseg = 1
    soma.pt3d = [
        (0.0, 0.0, 0.0, diameter_um),
        (1.0, 2.0, 0.0, diameter_um),
    ]
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("two_radii")

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    seg = group[0](0.5)
    inv = seg._ref_inv_two_radii.value()
    sim.run(tstop)
    return tstop, inv, seg._ref_v.value()


def simulate():
    for diameter_um in [1.1, 1.2, 1.3]:
        tstop, inv, voltage = _run_single_section(diameter_um)
        area_um2 = math.pi * diameter_um * math.sqrt(5.0)
        rate = diameter_um * diameter_um + area_um2
        voltage_expected = 20.0 + (-65.0 - 20.0) * math.exp(-rate * tstop)

        assert np.isclose(inv, 1.0 / rate, rtol=0.0, atol=6.0e-9)
        assert abs(voltage - voltage_expected) < 0.01 * abs(-65.0)


if __name__ == "__main__":
    simulate()
