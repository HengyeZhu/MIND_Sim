from __future__ import annotations

import mind_sim as ms


def test_vector_record_uses_micro_runtime():
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
    group.insert("hh")
    seg = group[0](0.5)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    t = ms.Vector().record(sim._ref_t)
    v = ms.Vector().record(seg._ref_v)
    sim.run(0.1)

    assert t.to_python() == [0.0, 0.025, 0.05, 0.07500000000000001, 0.1]
    assert len(v) == 5
    assert v[0] == -65.0
