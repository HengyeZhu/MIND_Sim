from __future__ import annotations

import numpy as np

import mind_sim as ms


def test_ste():
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
    seg.insert("IClamp", **{"del": 0.0, "dur": 10.0, "amp": 0.1})
    sim.network().register_spike_source(0, seg._ref_v, -50.0)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    t = ms.Vector().record(sim._ref_t)
    v = ms.Vector().record(seg._ref_v)
    sim.run(10.0)

    times = np.array(t.to_python())
    voltage = np.array(v.to_python())
    spike_times = np.array(sim.spike_times())
    assert spike_times.size >= 1

    first_spike_index = int(np.argmin(np.abs(times - spike_times[0])))
    assert voltage[first_spike_index] >= -50.0
    assert voltage[first_spike_index - 1] <= -50.0
