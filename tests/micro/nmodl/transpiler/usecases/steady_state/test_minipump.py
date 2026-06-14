from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def minipump_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def run(steady_state, minipump_mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)
    sim.load_mech(str(minipump_mod_dir))
    sim.run_steady_state_minipump = 1.0 if steady_state else 0.0

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 1.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("minipump")

    sim.build_microcircuit()
    seg = group[0](0.5)
    t = ms.Vector().record(sim._ref_t)
    X = ms.Vector().record(seg._ref_X_minipump)
    Y = ms.Vector().record(seg._ref_Y_minipump)
    Z = ms.Vector().record(seg._ref_Z_minipump)

    sim.finitialize(-65.0)
    sim.run(1.0)

    return (
        np.asarray(t.to_python()),
        np.asarray(X.to_python()),
        np.asarray(Y.to_python()),
        np.asarray(Z.to_python()),
    )


def assert_trace_samples(t, X, Y, Z, expected):
    np.testing.assert_allclose([len(t), t[0], t[-1]], expected["time"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(X[:3], expected["X_first"], rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose(Y[:3], expected["Y_first"], rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose(Z[:3], expected["Z_first"], rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose([X[-1], Y[-1], Z[-1]], expected["last"], rtol=1.0e-8, atol=1.0e-12)


def test_steady_state(minipump_mod_dir):
    t, X, Y, Z = run(True, minipump_mod_dir)
    assert_trace_samples(
        t,
        X,
        Y,
        Z,
        {
            "time": [41, 0.0, 1.0],
            "X_first": [39.99999689300149, 39.99999689300149, 39.99999689300149],
            "Y_first": [7.9999968930014544, 7.9999968930014544, 7.9999968930014544],
            "Z_first": [239.99988814805974, 239.99988814805974, 239.99988814805974],
            "last": [39.99999689300149, 7.9999968930014544, 239.99988814805974],
        },
    )


def test_no_steady_state(minipump_mod_dir):
    t, X, Y, Z = run(False, minipump_mod_dir)
    assert_trace_samples(
        t,
        X,
        Y,
        Z,
        {
            "time": [41, 0.0, 1.0],
            "X_first": [40.0, 39.999999976282446, 39.99999995274594],
            "Y_first": [8.0, 7.999999976282443, 7.999999952745936],
            "Z_first": [1.0, 2.8244275018611913, 4.634927974634766],
            "last": [39.999999179775514, 7.9999991797754975, 64.09419252606067],
        },
    )
