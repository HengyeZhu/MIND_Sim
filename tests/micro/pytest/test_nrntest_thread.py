from __future__ import annotations

import numpy as np

import mind_sim as ms


def _run_thread_model(thread_count: int):
    sim = ms.micro.sim()
    sim.set_device("cpu")
    sim.set_num_threads(thread_count)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 5.6419
    soma.diam_um = 5.6419
    sim.build_morphology([{"name": "CELL", "num_cells": 10, "sections": [soma]}])

    segments = []
    for index, cell in enumerate(sim.population("CELL")):
        cell.v_init = -65.0
        group = cell.group("soma")
        group.Ra = 35.4
        group.cm = 1.0
        group.insert("hh")
        seg = group[0](0.5)
        seg.insert("SEClamp", dur1=100.0, amp1=(100.0 * index / 10.0) - 50.0, rs=1.0)
        segments.append(seg)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    time = ms.Vector().record(sim._ref_t)
    voltages = [ms.Vector().record(seg._ref_v) for seg in segments]
    sim.run(5.0)
    return np.asarray(time.to_python()), [np.asarray(v.to_python()) for v in voltages]


def test_mcna():
    one_thread_time, one_thread_voltages = _run_thread_model(1)
    three_thread_time, three_thread_voltages = _run_thread_model(3)

    assert np.array_equal(one_thread_time, three_thread_time)
    for one_thread, three_thread in zip(one_thread_voltages, three_thread_voltages):
        assert np.array_equal(one_thread, three_thread)

    sample_indices = [0, 1, 40, 200]
    assert np.allclose(one_thread_time[sample_indices], [0.0, 0.025, 1.0, 5.0])
    assert np.allclose(
        one_thread_voltages[0][sample_indices],
        [-65.0, -50.5862810438723, -49.94298320924122, -49.99387963333887],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert np.allclose(
        one_thread_voltages[-1][sample_indices],
        [-65.0, 35.89585786202048, 39.16194836547582, 36.5392040816216],
        rtol=0.0,
        atol=2.0e-12,
    )
