from __future__ import annotations

import numpy as np

import mind_sim as ms


def _sample(vec, sample_times, dt=0.025):
    return [vec[round(t / dt)] for t in sample_times]


def test_netcvode_cover():
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": 2, "sections": [soma]}])

    source = sim.population("CELL")[0]
    target = sim.population("CELL")[1]
    for cell in (source, target):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")

    source_seg = source.group("soma")[0](0.5)
    target_seg = target.group("soma")[0](0.5)
    source_seg.insert("IClamp", **{"del": 0.5, "dur": 0.1, "amp": 0.3})
    syn = target_seg.insert("ExpSyn", tau=3.0, e=0.0)

    network = sim.network()
    network.register_spike_source(10, source_seg._ref_v, 0.0)
    netcon = network.sid_connect(10, syn, 0.5, 0.1)

    assert netcon.wcnt() == 1
    assert netcon.threshold == 0.0
    assert netcon.delay == 0.1
    assert netcon.weight[0] == 0.5

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    source_voltage = ms.Vector().record(source_seg._ref_v)
    target_voltage = ms.Vector().record(target_seg._ref_v)
    syn_g = ms.Vector().record(syn._ref_g)
    sim.run(2.0)

    spike_times = sim.spike_times()
    spike_gids = sim.spike_gids()
    assert spike_gids == [10]
    assert [round(time / 0.025) for time in spike_times] == [40]
    assert np.allclose(
        _sample(source_voltage, [0.0, 1.0, 1.025, 1.05, 2.0]),
        [-65.0, 1.8673835165202126, 9.116754628489247, 16.35818838334311, 10.349259178013854],
        rtol=0.0,
        atol=2.0e-11,
    )
    assert np.allclose(
        _sample(target_voltage, [1.0, 1.05, 1.125, 1.15, 2.0]),
        [-64.97554793406478, -64.97453872485134, -4.888213705525871, -0.436946718678219, 2.5752850459483407],
        rtol=0.0,
        atol=2.0e-11,
    )
    assert syn_g[round(1.1 / 0.025)] == 0.0
    assert syn_g[round(1.125 / 0.025)] > 0.0
