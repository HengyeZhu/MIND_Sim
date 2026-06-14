from __future__ import annotations

import numpy as np

import mind_sim as ms


def _cell_with_clamp(clamp_kind: str):
    sim = ms.Sim()
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
    group.insert("hh")
    seg = group[0](0.5)

    if clamp_kind == "iclamp":
        clamp = seg.insert("IClamp", **{"del": 1.0, "dur": 0.1, "amp": 0.3})
    else:
        clamp = seg.insert("SEClamp", dur1=1.0, amp1=-65.0, dur2=1.0, amp2=-10.0, rs=10.0)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    t = ms.Vector().record(sim._ref_t)
    v = ms.Vector().record(seg._ref_v)
    i = ms.Vector().record(clamp._ref_i)
    sim.run(2.0)
    return np.array(t.to_python()), np.array(v.to_python()), np.array(i.to_python())


def test_imem():
    t_iclamp, v_iclamp, i_iclamp = _cell_with_clamp("iclamp")
    t_seclamp, v_seclamp, i_seclamp = _cell_with_clamp("seclamp")

    assert np.array_equal(t_iclamp, t_seclamp)
    assert np.allclose(np.diff(t_iclamp), 0.025)

    before = t_iclamp < 1.0
    during = (t_iclamp >= 1.025) & (t_iclamp <= 1.1)
    after = t_iclamp > 1.125
    assert np.allclose(i_iclamp[before], 0.0)
    assert np.allclose(i_iclamp[during], 0.3)
    assert np.allclose(i_iclamp[after], 0.0)

    assert i_seclamp[0] == 0.0
    assert i_seclamp[t_seclamp > 1.0].max() > np.max(np.abs(i_seclamp[t_seclamp <= 1.0]))
    assert v_seclamp[t_seclamp >= 1.5].mean() > v_iclamp[t_iclamp >= 1.5].mean()
