from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


def mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def run_simulation():
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)
    sim.load_mech(str(mod_dir()))

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("side_effects")

    sim.build_microcircuit()
    seg = group[0](0.5)
    t = ms.Vector().record(sim._ref_t)
    x = ms.Vector().record(seg._ref_x_side_effects)
    X = ms.Vector().record(seg._ref_X_side_effects)
    Y = ms.Vector().record(seg._ref_Y_side_effects)
    forward_flux = ms.Vector().record(seg._ref_forward_flux_side_effects)
    backward_flux = ms.Vector().record(seg._ref_backward_flux_side_effects)

    sim.finitialize(-65.0)
    sim.run(5.0)

    return {
        "t": np.asarray(t.to_python()),
        "x": np.asarray(x.to_python()),
        "X": np.asarray(X.to_python()),
        "Y": np.asarray(Y.to_python()),
        "forward_flux": np.asarray(forward_flux.to_python()),
        "backward_flux": np.asarray(backward_flux.to_python()),
    }


def check_assignment(x, X):
    np.testing.assert_array_equal(x[1:], X[1:])


def check_flux(actual_flux, expected_flux):
    np.testing.assert_array_almost_equal_nulp(actual_flux[1:], expected_flux[1:], nulp=8)


def check_forward_flux(X, actual_flux):
    check_flux(actual_flux, 0.4 * X)


def check_backward_flux(Y, actual_flux):
    check_flux(actual_flux, 0.5 * Y)


if __name__ == "__main__":
    timeseries = run_simulation()
    check_assignment(timeseries["x"], timeseries["X"])
    check_forward_flux(timeseries["X"], timeseries["forward_flux"])
    check_backward_flux(timeseries["Y"], timeseries["backward_flux"])
