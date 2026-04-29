from __future__ import annotations

import math

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    sim = ms.Sim()
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 18.8
    soma.diam_um = 18.8
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": 1, "sections": [soma]}])

    group = sim.population("E")[0].group("soma")
    group.Ra = 123.0
    group.cm = 1.0
    group.insert("hh", gnabar=0.12, gkbar=0.036, gl=0.0003, el=-54.3)

    mid = group[0](0.5)
    mid.insert("IClamp", **{"del": 0.0, "dur": 1.0, "amp": 0.05})

    sim.build_microcircuit()
    sim.finitialize(-65.0)

    ena = mid.ref("ena", "na_ion").value()
    ek = mid.ref("ek", "k_ion").value()
    require(abs(ena - 50.0) < 1e-9, f"hh ena did not bind to na_ion default: {ena}")
    require(abs(ek + 77.0) < 1e-9, f"hh ek did not bind to k_ion default: {ek}")

    v0 = mid._ref_v.value()
    sim.run(0.5)
    v1 = mid._ref_v.value()
    ina = mid.ref("ina", "na_ion").value()
    ik = mid.ref("ik", "k_ion").value()

    require(math.isfinite(v1), "voltage is not finite after hh advance")
    require(math.isfinite(ina), "hh ina is not finite after advance")
    require(math.isfinite(ik), "hh ik is not finite after advance")
    require(abs(v1 - v0) > 1e-12, "hh + IClamp did not advance voltage")


if __name__ == "__main__":
    main()
