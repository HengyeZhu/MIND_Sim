from __future__ import annotations

import subprocess
from pathlib import Path
import shutil

import numpy as np
import pytest

import mind_sim as ms


MOD_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def read_only_mod_dir(tmp_path_factory):
    mod_dir = tmp_path_factory.mktemp("read_only_mod")
    shutil.copy2(MOD_DIR / "read_only.mod", mod_dir / "read_only.mod")
    lib = mod_dir / "x86_64" / "libcorenrnmech.so"
    mod_mtime = (mod_dir / "read_only.mod").stat().st_mtime
    if not lib.exists() or lib.stat().st_mtime < mod_mtime:
        subprocess.run(["mind_nrnivmodl", str(mod_dir)], check=True)
    return mod_dir


def single_section_sim(mod_dir):
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)
    sim.load_mech(str(mod_dir))

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


def test_read_only(read_only_mod_dir):
    sim, group, seg = single_section_sim(read_only_mod_dir)
    group.insert("read_only")
    sim.build_microcircuit()
    x_hoc = ms.Vector().record(seg._ref_x_read_only)
    t_hoc = ms.Vector().record(sim._ref_t)

    sim.finitialize(-65.0)
    sim.run(1.0)

    x = np.array(x_hoc.to_python())
    t = np.array(t_hoc.to_python())

    x_exact = 2.0 * np.ones_like(t)
    x_exact[0] = 42
    abs_err = np.abs(x - x_exact)

    assert np.all(abs_err < 1e-12), f"{abs_err=}"
