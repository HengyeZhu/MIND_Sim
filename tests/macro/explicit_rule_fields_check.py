from __future__ import annotations

import math

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_raises(fn, text: str) -> None:
    try:
        fn()
    except Exception as exc:
        if text not in str(exc):
            raise RuntimeError(f"expected error containing {text!r}, got {exc!r}") from exc
        return
    raise RuntimeError(f"expected error containing {text!r}")


def main() -> None:
    require(not hasattr(ms, "RegionRule"), "RegionRule must not be public API")
    require(not hasattr(ms, "NeuralFieldRule"), "NeuralFieldRule must not be public API")

    require_raises(
        lambda: ms.Network(labels=["r0"], weights=[[0.0]], delays=[[0.0]]).roi(0).use("S = 0.0"),
        "missing 1 required keyword-only argument: 'exposures'",
    )
    require_raises(
        lambda: ms.Network(labels=["r0"], weights=[[0.0]], delays=[[0.0]]).roi(0).use(
            "dS/dt = drive",
            inputs={"drive": 0.0},
            exposures="S",
        ),
        "derivative references undeclared state: S",
    )
    require_raises(
        lambda: ms.Network(labels=["r0"], weights=[[0.0]], delays=[[0.0]]).use_neural_field(
            ms.NeuralField(
                "bad_field",
                "x = x + dt * drive",
                exposures="x",
                state={"x": 0.0},
            ),
            node_map=ms.NodeToRoiMap([0]),
        ),
        "MIND_Sim code generation failed",
    )

    region = """
  tmp = drive * 2.0;
  S = tmp + 1.0;
"""
    network = ms.Network(labels=["r0"], weights=[[0.0]], delays=[[0.0]])
    network.roi(0).use(region, inputs={"drive": 0.0}, exposures="S").dc_input({"drive": 3.0})
    result = ms.MacroSimulator(network, dt_macro=0.1).run(0.1)

    require(result.exposures.exposure_count == 1, "WRITE should define the ROI exposure schema")
    final = result.exposures.values[-1]
    require(math.isclose(final, 7.0, rel_tol=0.0, abs_tol=1e-12), f"unexpected exposure value: {final}")


if __name__ == "__main__":
    main()
