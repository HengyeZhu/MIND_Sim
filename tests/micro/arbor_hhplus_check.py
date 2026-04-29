from __future__ import annotations

import math
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    sim = ms.Sim()
    sim.set_dt(0.01)
    sim.ion_register("cl", -1.0)
    sim.load_mech_metadata(str(Path(__file__).resolve().parents[2] / "examples" / "cosim_arbor_hhplus" / "mod"))
    sim.cli0_cl_ion = 5.0
    sim.clo0_cl_ion = 112.0

    volume_um3 = 2160.0
    radius_um = (0.75 * volume_um3 / math.pi) ** (1.0 / 3.0)
    area_um2 = 4.0 * math.pi * radius_um * radius_um
    length_um = math.sqrt(0.5 * area_um2 / math.pi)

    soma = ms.section("soma", "soma")
    soma.L_um = length_um
    soma.diam_um = 2.0 * length_um
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": 1, "sections": [soma]}])

    group = sim.population("E")[0].group("soma")
    group.Ra = 100.0
    group.cm = 0.115
    group.insert("hhplus", kbath=17.0)
    group.ecl = -26.64 * math.log(112.0 / 5.0)

    mid = group[0](0.5)
    sim.build_microcircuit()
    sim.finitialize(-78.0)

    ecl = mid.ref("ecl", "cl_ion").value()
    require(abs(ecl - group.ecl) < 1e-12, f"hhplus did not read migrated cl_ion ecl: {ecl}")
    require(math.isfinite(mid.ref("ina", "na_ion").value()), "hhplus ina is not finite")
    require(math.isfinite(mid.ref("ik", "k_ion").value()), "hhplus ik is not finite")
    require(math.isfinite(mid.ref("icl", "cl_ion").value()), "hhplus icl is not finite")

    v0 = mid._ref_v.value()
    sim.run(1.0)
    v1 = mid._ref_v.value()
    require(math.isfinite(v1), "hhplus voltage is not finite after run")
    require(abs(v1 - v0) > 1e-12, "hhplus did not advance voltage")


if __name__ == "__main__":
    main()
