from __future__ import annotations

import numpy as np

import mind_sim as ms


def _run_layered_network(chunks):
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 4, "sections": [soma]}])

    synapses = []
    network = sim.network()
    for gid, cell in enumerate(sim.population("CELL")):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")
        seg = group[0](0.5)
        synapses.append(seg.insert("ExpSyn", tau=0.2, e=0.0))
        network.register_spike_source(gid, seg._ref_v, 0.0)

    stim = sim.insert("NetStim", start=0.9999, number=5.0, interval=0.1, noise=0.0)
    network.event_connect(stim, synapses[0], 0.08, 0.0)
    for gid in range(3):
        network.sid_connect(gid, synapses[gid + 1], 0.06 + 0.01 * gid, 0.025 + 0.01 * gid)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    for stop_time in chunks:
        sim.continue_run(stop_time)
    return np.array(sim.spike_times()), np.array(sim.spike_gids())


def test_bbss():
    full_t, full_gid = _run_layered_network([5.0])
    chunk_t, chunk_gid = _run_layered_network([1.1, 2.0, 3.5, 5.0])

    assert full_t.size > 0
    assert np.array_equal(full_t, chunk_t)
    assert np.array_equal(full_gid, chunk_gid)
