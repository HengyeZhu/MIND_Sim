from __future__ import annotations

import numpy as np

import mind_sim as ms


def _model():
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("hh")
    seg = group[0](0.5)
    seg.insert("IClamp", **{"del": 0.5, "dur": 0.1, "amp": 0.3})

    sim.build_microcircuit()
    return sim, seg


def _run(method):
    sim, seg = _model()
    sim.finitialize(-65.0)
    t = ms.Vector().record(sim._ref_t)
    v = ms.Vector().record(seg._ref_v)

    if method == "fadvance":
        while sim.get_t() < 1.0:
            sim.fadvance()
    else:
        sim.continue_run(1.0)

    return np.array(t.to_python()), np.array(v.to_python())


def test_run():
    t_by_step, v_by_step = _run("fadvance")
    t_by_run, v_by_run = _run("continue_run")

    assert np.allclose(t_by_step, t_by_run[1:], rtol=0.0, atol=np.finfo(float).eps * len(t_by_step))
    assert np.array_equal(v_by_step, v_by_run[1:])
