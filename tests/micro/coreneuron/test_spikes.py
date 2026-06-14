from __future__ import annotations

import itertools
import subprocess

import numpy as np
import pytest

import mind_sim as ms


def _single_cell_sim(*, nseg: int = 1, length: float = 10.0, diam: float = 10.0):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.nseg = nseg
    soma.L_um = length
    soma.diam_um = diam
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])

    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    return sim, cell, group


def test_spikes():
    sim, _, soma = _single_cell_sim(length=5.6419, diam=5.6419)
    soma.insert("hh")
    seg = soma[0](0.5)
    seg.insert("IClamp", **{"del": 0.5, "dur": 0.1, "amp": 0.3})
    sim.network().register_spike_source(11, seg._ref_v, None)
    with pytest.raises(RuntimeError, match="already registered"):
        sim.network().register_spike_source(12, seg._ref_v, None)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    sim.run(5.0)

    assert sim.spike_times() == [1.0500000000999983]
    assert sim.spike_gids() == [11]

    sim.clear_spikes()
    assert sim.spike_times() == []
    assert sim.spike_gids() == []

    sim.finitialize(-65.0)
    assert sim.spike_times() == []
    assert sim.spike_gids() == []
