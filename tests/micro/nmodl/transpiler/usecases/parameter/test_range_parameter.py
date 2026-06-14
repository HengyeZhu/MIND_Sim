from __future__ import annotations

import subprocess
from pathlib import Path

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


def mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def single_segment_sim(mod_dir):
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
    return sim, group[0](0.5)


sim, seg = single_segment_sim(mod_dir())
pp = seg.insert("range_parameter")

sim.build_microcircuit()
sim.finitialize(-65.0)

assert pp._ref_x.value() == 42.0
assert pp._ref_y.value() == 43.0

sim, seg = single_segment_sim(mod_dir())
pp = seg.insert("range_parameter", x=42.1)

sim.build_microcircuit()
sim.finitialize(-65.0)
sim.run(5.0)

assert pp._ref_x.value() == 42.1
assert pp._ref_y.value() == 43.0
