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


def _continue_run_matches_single_run():
    def run_in_chunks(chunks):
        sim, _, soma = _single_cell_sim(nseg=3)
        soma.insert("hh")
        seg = soma[0](0.5)
        seg.insert("IClamp", **{"del": 1.0, "dur": 0.1, "amp": 0.3})

        sim.build_microcircuit()
        sim.finitialize(-65.0)
        v = ms.Vector().record(seg._ref_v)
        for chunk in chunks:
            sim.continue_run(chunk)
        return np.array(v.to_python())

    one_run = run_in_chunks([10.5])
    chunked = run_in_chunks([1.0, 1.0, 2.5, 6.0])

    assert np.array_equal(one_run, chunked)

def _fadvance_matches_continue_run_fixed_step():
    def run_with(method):
        sim, _, soma = _single_cell_sim(length=5.6419, diam=5.6419)
        soma.insert("hh")
        seg = soma[0](0.5)
        seg.insert("IClamp", **{"del": 0.5, "dur": 0.1, "amp": 0.3})

        sim.build_microcircuit()
        sim.finitialize(-65.0)
        t = ms.Vector().record(sim._ref_t)
        v = ms.Vector().record(seg._ref_v)
        if method == "fadvance":
            while sim.get_t() < 1.0:
                sim.fadvance()
        else:
            sim.continue_run(1.0)
        return np.array(t.to_python()), np.array(v.to_python())

    t_by_step, v_by_step = run_with("fadvance")
    t_by_chunk, v_by_chunk = run_with("continue_run")

    assert np.allclose(t_by_step, t_by_chunk[1:], rtol=0.0, atol=np.finfo(float).eps * len(t_by_step))
    assert np.array_equal(v_by_step, v_by_chunk[1:])


def test_direct_memory_transfer():
    _continue_run_matches_single_run()
    _fadvance_matches_continue_run_fixed_step()
