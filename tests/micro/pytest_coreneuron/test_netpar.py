from __future__ import annotations

import itertools
import subprocess

import numpy as np
import pytest

import mind_sim as ms


def _sample(vec, sample_times, dt=0.025):
    return [vec[round(t / dt)] for t in sample_times]


def test_1():
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(2)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 4, "sections": [soma]}])

    target_segments = []
    synapses = []
    for cell in sim.population("CELL"):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")
        seg = group[0](0.5)
        target_segments.append(seg)
        synapses.append(seg.insert("ExpSyn", tau=3.0, e=0.0))

    stim = sim.insert("NetStim", start=1.0, noise=0.0, interval=10.0, number=1.0)
    network = sim.network()
    for index, synapse in enumerate(synapses):
        network.event_connect(stim, synapse, 0.05 * (index + 1), 0.1 * (index + 1))

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    traces = [ms.Vector().record(seg._ref_v) for seg in target_segments]
    sim.run(4.0)

    sample_times = [0.0, 1.0, 1.1, 1.2, 2.0, 4.0]
    expected = [
        [-65.0, -64.97554793406478, -64.97354825266615, -18.256535776018865, 23.83794958344851, -38.12973227033436],
        [-65.0, -64.97554793406478, -64.97354825266615, -64.97162368768419, 20.99607075301613, -27.950025626052618],
        [-65.0, -64.97554793406478, -64.97354825266615, -64.97162368768419, 18.717082124092567, -21.76147342762994],
        [-65.0, -64.97554793406478, -64.97354825266615, -64.97162368768419, 16.31601809789211, -17.26033309985128],
    ]
    for trace, reference in zip(traces, expected):
        assert np.allclose(_sample(trace, sample_times), reference, rtol=0.0, atol=2.0e-12)
