from __future__ import annotations

import math
from pathlib import Path

import mind_sim as ms


ROOT = Path(__file__).resolve().parents[2]


def make_hhplus_sim(cell_count: int) -> tuple[ms.Sim, list[object]]:
    sim = ms.Sim()
    sim.set_dt(0.01)
    sim.set_spike_output_enabled(True)
    sim.ion_register("cl", -1.0)
    sim.load_mech_metadata(str(ROOT / "examples" / "cosim_arbor_hhplus" / "mod"))
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
    sim.build_morphology([{"name": "E", "num_cells": cell_count, "sections": [soma]}])

    population = sim.population("E")
    somata = []
    for gid in range(cell_count):
        group = population[gid].group("soma")
        group.Ra = 100.0
        group.cm = 0.115
        group.insert("hhplus", kbath=17.0)
        group.ecl = -26.64 * math.log(112.0 / 5.0)
        midpoint = group[0](0.5)
        somata.append(midpoint)

    return sim, somata


def require_close_sequence(actual: list[float], expected: list[float], label: str) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: expected {len(expected)} spikes, got {len(actual)}: {actual}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if abs(got - want) > 1e-9:
            raise RuntimeError(f"{label}: spike {index} expected {want}, got {got}")


def check_single_cell_autapse_matches_core() -> None:
    expected_by_weight = {
        0.5: [78.96000000010068, 79.47000000010021, 119.60000000006372, 120.0500000000633],
        49.5: [78.96000000010068, 79.47000000010021, 138.56000000004647, 139.01000000004606],
    }

    for weight, expected_spikes in expected_by_weight.items():
        sim, somata = make_hhplus_sim(1)
        net = sim.network()
        midpoint = somata[0]
        net.register_gid_source(0, midpoint._ref_v, -25.0)
        synapse = midpoint.insert("ExpSyn", tau=2.0, e=0.0)
        net.gid_connect(0, synapse, weight, 0.5)

        sim.build_microcircuit()
        sim.finitialize(-78.0)
        sim.run(150.0)

        require_close_sequence(
            list(sim.get_spk_by_gid(0)),
            expected_spikes,
            f"single-cell hhplus autapse w={weight}",
        )


def check_recurrent_network_is_finite() -> None:
    cell_count = 10
    sim, somata = make_hhplus_sim(cell_count)
    net = sim.network()
    for gid, midpoint in enumerate(somata):
        net.register_gid_source(gid, midpoint._ref_v, -25.0)

    for post, midpoint in enumerate(somata):
        synapse = midpoint.insert("ExpSyn", tau=2.0, e=0.0)
        for pre in range(cell_count):
            if pre != post:
                net.gid_connect(pre, synapse, 0.5, 0.5)

    sim.build_microcircuit()
    sim.finitialize(-78.0)
    sim.run(100.0)

    for midpoint in somata:
        value = midpoint._ref_v.value()
        if not math.isfinite(value):
            raise RuntimeError("recurrent migrated hhplus voltage is not finite")


def main() -> None:
    check_single_cell_autapse_matches_core()
    check_recurrent_network_is_finite()


if __name__ == "__main__":
    main()
