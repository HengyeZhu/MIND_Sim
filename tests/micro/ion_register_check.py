from __future__ import annotations

import math
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def nernst_mV(ci: float, co: float, charge: float, celsius: float) -> float:
    gas_constant = 8.31446261815324
    faraday = 96485.33212
    return 1000.0 * gas_constant * (273.15 + celsius) / faraday / charge * math.log(co / ci)


def main() -> None:
    sim = ms.Sim()
    sim.set_dt(0.025)
    sim.ion_register("cl", -1.0)
    sim.load_mech_metadata(str(Path(__file__).resolve().parent / "mod_ion_register"))

    require(sim.ion_charge("cl_ion") == -1.0, "cl_ion charge was not registered")
    sim.cli0_cl_ion = 5.0
    sim.clo0_cl_ion = 112.0

    soma = ms.section("soma", "soma")
    soma.L_um = 18.8
    soma.diam_um = 18.8
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": 1, "sections": [soma]}])

    group = sim.population("E")[0].group("soma")
    group.Ra = 123.0
    group.cm = 1.0
    group.insert("clchan", gbar=0.002)
    require(group.ion_style("cl_ion", 1, 2, 1, 0, 1) == -1, "unexpected pre-build ion style")

    mid = group[0](0.5)
    sim.build_microcircuit()
    sim.finitialize(-65.0)

    expected_ecl = nernst_mV(5.0, 112.0, -1.0, sim.celsius)
    ecl = mid.ref("ecl", "cl_ion").value()
    cli = mid.ref("cli", "cl_ion").value()
    clo = mid.ref("clo", "cl_ion").value()
    icl = mid.ref("icl", "cl_ion").value()

    require(abs(cli - 5.0) < 1e-12, f"cli0_cl_ion did not initialize cli: {cli}")
    require(abs(clo - 112.0) < 1e-12, f"clo0_cl_ion did not initialize clo: {clo}")
    require(abs(ecl - expected_ecl) < 1e-5, f"ecl did not follow NRN Nernst style: {ecl} vs {expected_ecl}")
    require(math.isfinite(icl), "clchan did not write finite icl")

    v0 = mid._ref_v.value()
    sim.run(0.025)
    require(math.isfinite(mid._ref_v.value()), "voltage is not finite after clchan advance")
    require(abs(mid._ref_v.value() - v0) > 1e-12, "clchan current did not advance voltage")


if __name__ == "__main__":
    main()
