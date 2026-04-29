from __future__ import annotations

from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_rejected(func, needle: str) -> None:
    try:
        func()
    except Exception as exc:
        require(needle in str(exc), f"unexpected rejection message: {exc}")
        return
    raise RuntimeError("operation unexpectedly succeeded")


def main() -> None:
    sim = ms.Sim()
    sim.set_dt(0.025)
    sim.load_mech_metadata(str(Path(__file__).resolve().parent / "mod_global"))

    soma = ms.section("soma", "soma")
    soma.L_um = 20.0
    soma.diam_um = 20.0
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": 1, "sections": [soma]}])

    group = sim.population("E")[0].group("soma")
    group.Ra = 100.0
    group.cm = 1.0

    require_rejected(
        lambda: group.insert("globalparam", g_globalparam=0.002),
        "g_globalparam",
    )
    require_rejected(
        lambda: group.insert("globalparam", shift_globalparam=5.0),
        "shift_globalparam",
    )

    group.insert("globalparam", g=0.002, shift=7.0)
    require(abs(sim.shift_globalparam - 7.0) < 1e-12, "non-RANGE PARAMETER was not set globally")

    mid = group[0](0.5)
    sim.build_microcircuit()
    sim.finitialize(-65.0)
    require(abs(mid.ref("g", "globalparam").value() - 0.002) < 1e-12, "RANGE parameter g was not local")
    require(abs(sim.shift_globalparam - 7.0) < 1e-12, "global parameter changed during build")


if __name__ == "__main__":
    main()
