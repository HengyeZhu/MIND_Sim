from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mod_dir = Path(tmp)
        (mod_dir / "field_drive.mod").write_text(
            """
MIND {
  COUPLING field_drive
  READ x
  WRITE drive
}

EDGE {
  drive += weight * x;
}
""",
            encoding="utf-8",
        )
        field_rule = "x = x + dt * drive"

        network = ms.Network(
            labels=["a", "b"],
            weights=[
                [0.0, 0.5],
                [0.0, 0.0],
            ],
            delays=[
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )
        field = ms.NeuralField(
            "surface",
            field_rule,
            inputs={"drive": 0.0},
            exposures="x",
            state={"x": [1.0, 3.0, 10.0, 14.0]},
        )
        network.use_neural_field(field, node_map=ms.NodeToRoiMap([0, 0, 1, 1]))
        network.roi("a").connect(network.roi("b"), mod_dir / "field_drive.mod")

        result = ms.MacroSimulator(network, dt_macro=0.1).run(0.1)
        sample_width = result.exposures.recorded_roi_count * result.exposures.exposure_count
        initial = result.exposures.values[:sample_width]
        final = result.exposures.values[sample_width : 2 * sample_width]

        for got, want in zip(initial, [2.0, 12.0]):
            require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12), f"initial mean {got} != {want}")
        for got, want in zip(final, [2.6, 12.0]):
            require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12), f"global-coupled mean {got} != {want}")


if __name__ == "__main__":
    main()
