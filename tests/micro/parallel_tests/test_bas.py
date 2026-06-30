from __future__ import annotations

import itertools
import subprocess

import numpy as np
import pytest

import mind_sim as ms


def _build_ball_and_stick_network(cell_count, thread_count):
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(thread_count)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 12.6157
    soma.diam_um = 12.6157
    dend = ms.section("dend", "dend")
    dend.L_um = 200.0
    dend.diam_um = 1.0
    dend.connect(soma, 1.0)
    sim.build_morphology([{"name": "CELL", "num_cells": cell_count, "sections": [soma, dend]}])

    synapses = []
    for cell in sim.population("CELL"):
        cell.v_init = -65.0
        soma_group = cell.group("soma")
        soma_group.Ra = 100.0
        soma_group.cm = 1.0
        soma_group.insert("hh", gnabar=0.12, gkbar=0.036, gl=0.0003, el=-54.3)
        dend_group = cell.group("dend")
        dend_group.Ra = 100.0
        dend_group.cm = 1.0
        dend_group.insert("pas", g=0.001, e=-65.0)
        synapses.append(dend_group[0](0.5).insert("ExpSyn", tau=2.0))

    return sim, synapses


def test_bas():
    def run_ring(thread_count):
        sim, synapses = _build_ball_and_stick_network(5, thread_count)
        network = sim.network()
        stim = sim.insert("NetStim", start=9.0, interval=10.0, number=1.0, noise=0.0)
        network.event_connect(stim, synapses[0], 0.04, 1.0)
        for gid, cell in enumerate(sim.population("CELL")):
            network.register_spike_source(gid, cell.group("soma")[0](0.5)._ref_v, 10.0)
        for gid in range(len(synapses)):
            network.sid_connect(gid, synapses[(gid + 1) % len(synapses)], 0.01, 25.0)

        sim.build_microcircuit()
        sim.finitialize(-65.0)
        sim.run(200.0)

        spikes = {gid: [] for gid in range(5)}
        for spike_time, spike_gid in zip(sim.spike_times(), sim.spike_gids()):
            spikes[spike_gid].append(spike_time)
        return spikes

    expected = {
        0: [10.925000000099914, 143.3000000001066],
        1: [37.40000000009994, 169.7750000000825],
        2: [63.87500000010596, 196.25000000005844],
        3: [90.35000000011198],
        4: [116.825000000118],
    }

    assert run_ring(1) == expected
    assert run_ring(3) == expected

def test_starnet():
    gids = [0] + [100 + index for index in range(8)] + [200]
    gid_to_index = {gid: index for index, gid in enumerate(gids)}

    def run_starnet(thread_count):
        sim, synapses = _build_ball_and_stick_network(len(gids), thread_count)
        network = sim.network()
        for gid, index in gid_to_index.items():
            network.register_spike_source(gid, sim.population("CELL")[index].group("soma")[0](0.5)._ref_v, 10.0)

        stim = sim.insert("NetStim", start=6.0, interval=10.0, number=100.0, noise=0.0)
        network.event_connect(stim, synapses[gid_to_index[0]], 0.01, 2.0)

        for target_gid in gids:
            target_layer = target_gid // 100
            if target_layer == 1:
                network.sid_connect(0, synapses[gid_to_index[target_gid]], 0.01, 20.0)
            elif target_layer == 2:
                for source_index in range(8):
                    network.sid_connect(100 + source_index, synapses[gid_to_index[target_gid]], 0.01, 20.0)

        sim.build_microcircuit()
        sim.finitialize(-65.0)
        sim.run(100.0)

        spikes = {gid: [] for gid in gids}
        for spike_time, spike_gid in zip(sim.spike_times(), sim.spike_gids()):
            spikes[spike_gid].append(spike_time)
        return spikes

    expected = {
        0: [
            9.475000000099996,
            20.400000000099375,
            30.750000000098787,
            41.02500000010077,
            51.40000000010313,
            69.82500000010731,
            80.45000000010972,
            90.77500000011207,
        ],
        100: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        101: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        102: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        103: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        104: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        105: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        106: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        107: [
            30.950000000098775,
            42.4750000001011,
            53.12500000010352,
            63.575000000105895,
            73.95000000010825,
            91.3500000001122,
        ],
        200: [
            51.750000000103206,
            63.47500000010587,
            74.2000000001083,
            84.65000000011068,
            95.05000000011304,
        ],
    }

    assert run_starnet(1) == expected
    assert run_starnet(3) == expected
