from __future__ import annotations

import subprocess
from pathlib import Path

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


def useion_mod_dir():
    lib = MOD_DIR / "x86_64" / "libcorenrnmech.so"
    mod_mtime = max(path.stat().st_mtime for path in MOD_DIR.glob("*.mod"))
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind-nrnivmodl", str(MOD_DIR)], check=True)
    return MOD_DIR


def single_section_sim(mod_dir):
    ms.load_mech(mod_dir)
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
    return sim, group, group[0](0.5)


def nernst(sim, ca):
    return sim.nernst(ca._ref_cai.value(), ca._ref_cao.value(), sim.ion_charge("ca_ion"))


def simulate(mech_name, compute_eca=None):
    sim, group, ca = single_section_sim(useion_mod_dir())
    group.insert(mech_name)
    sim.build_microcircuit()
    sim.finitialize(-65.0)

    eca_expected = nernst(sim, ca) if compute_eca is None else compute_eca(ca)
    eca = ca._ref_eca.value()

    assert abs(eca - eca_expected) < 1e-10, f"{eca} - {eca_expected} = {eca - eca_expected}"
