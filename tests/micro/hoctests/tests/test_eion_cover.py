from __future__ import annotations

from math import isclose

import pytest

import mind_sim as ms


def _single_cell_sim():
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
    return sim, group


def test_nernst():
    sim = ms.micro.sim()

    assert sim.nernst(1, 1, 0) == 0.0
    assert sim.nernst(-1, 1, 1) == 1e6
    assert sim.nernst(1, -1, 1) == -1e6


def test_ghk():
    sim = ms.micro.sim()

    assert isclose(sim.ghk(-10, 0.1, 10, 2), -2828.3285716150644)
    assert isclose(sim.ghk(1e-6, 0.1, 10, 2), -1910.40949510667)


def test_ion_style():
    _, soma = _single_cell_sim()

    with pytest.raises(RuntimeError):
        soma.ion_style("foo")


def test_second_order_cur():
    def run(secondorder):
        sim, soma = _single_cell_sim()
        soma.insert("hh")
        sim.secondorder = secondorder
        sim.build_microcircuit()
        sim.finitialize(-65.0)
        sim.fadvance()
        return soma[0](0.5)._ref_ina.value()

    assert isclose(run(2), -0.001220053188847315)
    assert isclose(run(0), -0.0012200571764654333)


def test_ion_charge():
    sim = ms.micro.sim()

    assert sim.ion_charge("na_ion") == 1

    with pytest.raises(RuntimeError, match="already defined"):
        sim.ion_register("na", 2.0)
    with pytest.raises(RuntimeError):
        sim.ion_charge("na")

    sim.ion_register("x", 4.0)
    assert sim.ion_charge("x_ion") == 4.0


def test_eion_cover():
    test_nernst()
    test_ghk()
    test_ion_style()
    test_second_order_cur()
    test_ion_charge()
