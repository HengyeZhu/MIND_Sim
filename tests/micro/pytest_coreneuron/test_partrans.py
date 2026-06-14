from __future__ import annotations

import numpy as np

import mind_sim as ms


def _run_transfer_network(thread_count: int):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(thread_count)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": 3, "sections": [soma]}])

    network = sim.network()
    synapses = []
    records = []
    for gid, cell in enumerate(sim.population("CELL")):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")
        seg = group[0](0.5)
        if gid == 0:
            seg.insert("IClamp", **{"del": 0.5, "dur": 0.1, "amp": 0.3})
        synapses.append(seg.insert("ExpSyn", tau=0.2, e=0.0))
        records.append(seg)
        network.register_spike_source(gid, seg._ref_v, 0.0)

    network.sid_connect(0, synapses[1], 0.04, 0.1)
    network.sid_connect(0, synapses[2], 0.08, 0.2)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    traces = [ms.Vector().record(seg._ref_v) for seg in records]
    sim.run(5.0)
    return [np.array(trace.to_python()) for trace in traces], sim.spike_times(), sim.spike_gids()


def test_partrans():
    traces_1, spikes_1, gids_1 = _run_transfer_network(1)
    traces_2, spikes_2, gids_2 = _run_transfer_network(2)

    assert spikes_1 == spikes_2
    assert gids_1 == gids_2
    for expected, actual in zip(traces_1, traces_2):
        assert np.array_equal(expected, actual)
    assert traces_1[1].max() > 0.0
    assert traces_1[2].max() > 0.0
    assert np.argmax(traces_1[1]) < np.argmax(traces_1[2])
