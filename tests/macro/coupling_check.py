from __future__ import annotations

import math

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_raises(callable_obj, message_fragment: str) -> None:
    try:
        callable_obj()
    except RuntimeError as exc:
        require(message_fragment in str(exc), f"unexpected error: {exc}")
        return
    raise RuntimeError(f"expected error containing: {message_fragment}")


def main() -> None:
    network = ms.Network(
        labels=["a", "b", "c"],
        weights=[
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        delays=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        inputs=["drive"],
        exposures=["S"],
    )

    region = ms.RegionRule(
        name="copy_input",
        step="out.S = in.drive;",
    )
    for roi in network.rois():
        roi.use(region)

    coupling = ms.CouplingRule(
        name="count_coupling",
        edge="in.drive += 1.0;",
    )
    network.couple_all(coupling, delays=False)

    result = ms.MacroSimulator(network, dt_macro=0.1).run(0.1)
    values = result.exposures.values
    sample_width = result.exposures.recorded_roi_count * result.exposures.exposure_count
    final_values = values[sample_width : sample_width * 2]

    expected = [1.0, 2.0, 0.0]
    require(len(final_values) == len(expected), "unexpected final exposure width")
    for got, want in zip(final_values, expected):
        require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-7), f"coupling value {got} != {want}")

    strict_network = ms.Network(
        labels=["strict"],
        weights=[[0.0]],
        delays=[[1.0]],
        inputs=["drive"],
        exposures=["S"],
    )
    require_raises(
        lambda: strict_network.roi(0).use(ms.RegionRule(
            name="unused_region_param",
            params={"unused": 1.0},
            step="out.S = in.drive;",
        )),
        "declared param is unused: unused",
    )
    require_raises(
        lambda: strict_network.roi(0).use(ms.RegionRule(
            name="unused_region_state",
            state={"x": 0.0},
            step="out.S = in.drive;",
        )),
        "declared state is unused: x",
    )
    require_raises(
        lambda: strict_network.couple_all(ms.CouplingRule(
            name="unused_coupling_param",
            params={"G": 1.0},
            edge="in.drive += src.S;",
        )),
        "declared param is unused: G",
    )
    require_raises(
        lambda: ms.MicroInputRule(
            name="unused_input_port",
            ports=["afferent"],
            code="double x = in.drive;",
        )._native_for(["drive"]),
        "declared input port is unused: afferent",
    )
    require_raises(
        lambda: ms.MicroOutputRule(
            name="unused_output_state",
            state={"count": 0.0},
            spike="",
            finish="out.S = 0.0;",
        )._native_for(["S"]),
        "declared state is unused: count",
    )


if __name__ == "__main__":
    main()
