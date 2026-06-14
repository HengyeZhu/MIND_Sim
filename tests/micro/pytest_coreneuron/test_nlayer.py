from __future__ import annotations

import numpy as np

import mind_sim as ms


def _run_cable_chain(nseg: int):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    cable = []
    for index in range(4):
        label = "soma" if index == 0 else f"cable{index}"
        sec = ms.section(f"cable{index}", label)
        sec.nseg = nseg
        sec.L_um = 100.0
        sec.diam_um = 1.0
        cable.append(sec)
    cable[1].connect(cable[0], 1.0)
    cable[2].connect(cable[1], 1.0)
    cable[3].connect(cable[0], 0.0)
    sim.build_morphology([{"name": "CABLE", "num_cells": 1, "sections": cable}])

    cell = sim.population("CABLE")[0]
    cell.v_init = -65.0
    recorded = []
    for index in range(4):
        group = cell.group("soma" if index == 0 else f"cable{index}")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")
        seg = group[0](1.0 if index == 3 else 0.5)
        if index == 3:
            seg.insert("IClamp", **{"del": 0.0, "dur": 1.0, "amp": 0.05})
        recorded.append(seg)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    traces = [ms.Vector().record(seg._ref_v) for seg in recorded]
    sim.run(10.0)
    return [np.array(trace.to_python()) for trace in traces]


def test_nlayer():
    nseg1 = _run_cable_chain(1)
    nseg3 = _run_cable_chain(3)

    assert all(np.isfinite(trace).all() for trace in nseg1 + nseg3)
    assert nseg1[3].max() > nseg1[0].max()
    assert nseg3[3].max() > nseg3[0].max()
    assert abs(nseg1[0][-1] - nseg3[0][-1]) < 1.0
