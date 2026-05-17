from __future__ import annotations

import math

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    local_rule = """
  S = S + dt * (drive + local(S));
"""
    network = ms.Network(
        labels=["a", "b"],
        weights=[
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        delays=[
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    )
    field = ms.NeuralField(
        "surface",
        local_rule,
        inputs={"drive": 0.0},
        exposures="S",
        local=ms.LocalConnectivity.from_edges(
            4,
            [
                (0, 1, 1.0),
                (1, 0, 1.0),
                (2, 3, 1.0),
                (3, 2, 1.0),
            ],
        ),
        state={"S": [1.0, 3.0, 10.0, 14.0]},
    )
    network.use_neural_field(
        field,
        node_map=ms.NodeToRoiMap([0, 0, 1, 1]),
    )
    network.roi("a").dc_input({"drive": 1.0})
    network.roi("b").dc_input({"drive": -2.0})

    result = ms.MacroSimulator(network, dt_macro=0.1).run(0.1)
    sample_width = result.exposures.recorded_roi_count * result.exposures.exposure_count
    initial = result.exposures.values[:sample_width]
    final = result.exposures.values[sample_width : 2 * sample_width]

    for got, want in zip(initial, [2.0, 12.0]):
        require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12), f"initial field mean {got} != {want}")
    for got, want in zip(final, [2.3, 13.0]):
        require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12), f"final field mean {got} != {want}")

    weighted_rule = """
  x = x + dt * 0.0;
"""
    weighted = ms.Network(
        labels=["a", "b"],
        weights=[
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        delays=[
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    )
    weighted_field = ms.NeuralField(
        "weighted_surface",
        weighted_rule,
        exposures="x",
        state={"x": [1.0, 2.0, 10.0, -3.0, 4.0]},
    )
    weighted.use_neural_field(
        weighted_field,
        node_map=ms.NodeToRoiMap([0, 0, 0, 1, 1], [2.0, 3.0, 5.0, 7.0, 11.0]),
    )
    weighted_result = ms.MacroSimulator(weighted, dt_macro=0.1).run(0.0)
    for got, want in zip(weighted_result.exposures.values, [5.8, 23.0 / 18.0]):
        require(
            math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12),
            f"weighted field mean {got} != {want}",
        )


if __name__ == "__main__":
    main()
