from __future__ import annotations

from pathlib import Path

import numpy as np

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


def mod_dir():
    import subprocess

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


def simulate(method, rtol, dt=None):
    sim, group, seg = single_section_sim(mod_dir())
    sim.set_dt(dt if dt is not None else 0.025)
    group.insert(f"{method}_scalar")
    sim.build_microcircuit()
    t = ms.Vector().record(sim._ref_t)
    x = ms.Vector().record(getattr(seg, f"_ref_x_{method}_scalar"))

    sim.finitialize(-65.0)
    sim.run(5.0)

    time = np.array(t.to_python())
    actual = np.array(x.to_python())
    expected = 42.0 * np.exp(-time)
    np.testing.assert_allclose(actual, expected, rtol=rtol)


if __name__ == "__main__":
    simulate("cnexp", rtol=1e-12)
    simulate("derivimplicit", rtol=1e-3, dt=1e-4)
