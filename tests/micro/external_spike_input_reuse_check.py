from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    micro = ms.Sim()
    micro.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 20.0
    soma.diam_um = 20.0
    soma.nseg = 1
    micro.build_morphology([{"name": "E", "num_cells": 2, "sections": [soma]}])

    population = micro.population("E")
    synapses = []
    voltages = []
    for gid in range(2):
        group = population[gid].group("soma")
        group.Ra = 100.0
        group.cm = 1.0
        group.insert("pas", g=0.0001, e=-65.0)
        midpoint = group[0](0.5)
        synapses.append(midpoint.insert("ExpSyn", tau=2.0, e=0.0))
        voltages.append(midpoint._ref_v)

    net = micro.network()
    source = net.spike_input()
    weight = 0.01
    net.spike_connect(source, synapses[0], weight, 0.0)
    net.spike_connect(source, synapses[0], weight, 0.0)
    net.spike_connect(source, synapses[1], weight, 0.0)

    micro.build_microcircuit()
    micro.finitialize(-65.0)
    initial_v = [ref.value() for ref in voltages]

    network = ms.Network(
        labels=["micro"],
        weights=[[0.0]],
        delays=[[1.0]],
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "external_spike_reuse.mod"
        input_path.write_text(
            """
MIND {
  MICRO_INPUT external_spike_reuse
  EMIT source
}
STATE {
  sent = 0.0
}
INPUT {
if (sent < 0.5) {
    source(t, 0);
    sent = 1.0;
}
}
""",
            encoding="utf-8",
        )
        output_path = tmpdir / "zero_output.mod"
        output_path.write_text(
            """
MIND {
  MICRO_OUTPUT zero_output
  WRITE S
}
BREAKPOINT {
  S = 0.0;
}
""",
            encoding="utf-8",
        )
        micro_owner = ms.MicroCircuit(micro).bind_roi(
            0,
            gid_ranges=population,
            ports={"source": source},
        )
        network.use_micro(micro_owner)
        network.roi(0).connect(network.roi(0), input_path)
        network.roi(0).connect(network.roi(0), output_path)

    ms.Simulator(network, dt_micro=0.025, dt_macro=0.1, batch_window=0.1).run(5.0)

    final_v = [ref.value() for ref in voltages]
    require(all(math.isfinite(value) for value in final_v), "external spike fanout produced non-finite voltage")
    depol0 = final_v[0] - initial_v[0]
    depol1 = final_v[1] - initial_v[1]
    require(depol0 > 1e-5, f"source fanout did not deliver to reused first target: {depol0}")
    require(depol1 > 1e-5, f"source fanout did not deliver to second target: {depol1}")
    require(depol0 > depol1 * 1.25, f"duplicate NetCons to one target were not both delivered: {depol0} vs {depol1}")


if __name__ == "__main__":
    main()
