from __future__ import annotations

import math

import mind_sim as ms


def test_is_ion():
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(2)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": 2, "sections": [soma]}])

    for cell in sim.population("CELL"):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")

    sim.build_microcircuit()
    sim.finitialize(-65.0)

    assert sim.ion_charge("na_ion") == 1.0
    assert sim.ion_charge("k_ion") == 1.0
    for cell in sim.population("CELL"):
        seg = cell.group("soma")[0](0.5)
        assert math.isfinite(seg._ref_v.value())
        assert math.isfinite(seg._ref_ina.value())
        assert math.isfinite(seg._ref_ik.value())
        assert math.isfinite(seg._ref_m_hh.value())
