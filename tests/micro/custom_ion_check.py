from __future__ import annotations

import math
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    sim = ms.Sim()
    sim.set_dt(0.025)
    sim.load_mech_metadata(str(Path(__file__).resolve().parent / "mod"))

    soma = ms.section("soma", "soma")
    soma.L_um = 18.8
    soma.diam_um = 18.8
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": 1, "sections": [soma]}])

    group = sim.population("E")[0].group("soma")
    group.Ra = 123.0
    group.cm = 1.0
    group.insert("xchan", gbar=0.002)

    mid = group[0](0.5)
    sim.build_microcircuit()
    sim.finitialize(-65.0)

    require(abs(mid.ref("ex", "x_ion").value()) < 1e-12, "custom x_ion ex default is wrong")
    require(abs(mid.ref("xi", "x_ion").value() - 1.0) < 1e-12, "custom x_ion xi default is wrong")
    require(abs(mid.ref("xo", "x_ion").value() - 1.0) < 1e-12, "custom x_ion xo default is wrong")

    ix0 = mid.ref("ix", "x_ion").value()
    dix0 = mid.ref("dix_dv_", "x_ion").value()
    require(abs(ix0 + 0.13) < 1e-12, f"xchan did not write ix into x_ion: {ix0}")
    require(abs(dix0 - 0.002) < 1e-12, f"xchan did not write dix_dv_ into x_ion: {dix0}")

    v0 = mid._ref_v.value()
    sim.run(0.025)
    v1 = mid._ref_v.value()
    ix1 = mid.ref("ix", "x_ion").value()
    dix1 = mid.ref("dix_dv_", "x_ion").value()

    require(math.isfinite(v1), "voltage is not finite after custom ion advance")
    require(v1 > v0, "custom x current did not depolarize the cell")
    require(abs(ix1 + 0.13) < 1e-12, f"x_ion ix was cleared after advance: {ix1}")
    require(abs(dix1 - 0.002) < 1e-12, f"x_ion dix_dv_ was cleared after advance: {dix1}")


if __name__ == "__main__":
    main()
