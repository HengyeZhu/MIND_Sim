from __future__ import annotations

import itertools
import subprocess

import numpy as np
import pytest

import mind_sim as ms





def test_datareturn():
    def run_model(thread_count, chunks):
        sim = ms.Sim()
        sim.set_device("cpu")
        sim.set_num_threads(thread_count)
        sim.set_dt(0.025)

        soma = ms.section("soma", "soma")
        soma.L_um = 10.0
        soma.diam_um = 10.0
        dend = ms.section("dend", "dend")
        dend.nseg = 3
        dend.L_um = 50.0
        dend.diam_um = 1.0
        dend.connect(soma, 1.0)
        axon = ms.section("axon", "axon")
        axon.nseg = 3
        axon.L_um = 60.0
        axon.diam_um = 0.8
        axon.connect(soma, 0.0)
        sim.build_morphology([{"name": "CELL", "num_cells": 4, "sections": [soma, dend, axon]}])

        network = sim.network()
        recorded_segments = []
        synapses = []
        for index, cell in enumerate(sim.population("CELL")):
            cell.v_init = -65.0
            soma_group = cell.group("soma")
            soma_group.Ra = 35.4
            soma_group.cm = 1.0
            soma_group.insert("hh", gnabar=0.11 + 0.005 * index, gkbar=0.034 + 0.001 * index)

            dend_group = cell.group("dend")
            dend_group.Ra = 80.0
            dend_group.cm = 1.0
            dend_group.insert("pas", e=-65.0, g=0.0008 + 0.0001 * index)
            dend_group.insert("hh", gnabar=0.02 + 0.002 * index, gkbar=0.01 + 0.001 * index)

            axon_group = cell.group("axon")
            axon_group.Ra = 35.4
            axon_group.cm = 1.0
            axon_group.insert("hh", gnabar=0.12, gkbar=0.036)

            source = soma_group[0](0.5)
            target = dend_group[0](0.5)
            synapses.append(target.insert("Exp2Syn", tau1=0.05, tau2=5.0, e=0.0))
            recorded_segments.extend([source, target, axon_group[0](0.5)])
            network.register_spike_source(index, source._ref_v, 0.0)

        for index, synapse in enumerate(synapses):
            stim = sim.insert(
                "NetStim",
                start=0.5 + 0.2 * index,
                interval=1.25 + 0.1 * index,
                number=4.0,
                noise=0.0,
            )
            network.event_connect(stim, synapse, 0.02 + 0.005 * index, 0.1)
            network.sid_connect((index - 1) % len(synapses), synapse, 0.01 + 0.002 * index, 0.2)

        sim.build_microcircuit()
        sim.finitialize(-65.0)
        for chunk in chunks:
            sim.continue_run(chunk)

        values = [sim.get_t()]
        for seg in recorded_segments:
            values.extend(
                [
                    seg._ref_v.value(),
                    seg._ref_ina.value(),
                    seg._ref_ik.value(),
                    seg._ref_m_hh.value(),
                    seg._ref_h_hh.value(),
                    seg._ref_n_hh.value(),
                ]
            )
        for synapse in synapses:
            values.extend([synapse._ref_g.value(), synapse._ref_i.value()])
        values.extend(sim.spike_times())
        values.extend(float(gid) for gid in sim.spike_gids())
        return np.array(values)

    one_run = run_model(1, [5.0])
    chunked = run_model(1, [1.0, 0.5, 1.5, 2.0])
    threaded = run_model(3, [1.0, 0.5, 1.5, 2.0])

    assert np.array_equal(one_run, chunked)
    assert np.array_equal(one_run, threaded)
    assert one_run.size > 80
