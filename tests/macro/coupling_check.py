from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_raises(callable_obj, message_fragment: str) -> None:
    try:
        callable_obj()
    except Exception as exc:
        require(message_fragment in str(exc), f"unexpected error: {exc}")
        return
    raise RuntimeError(f"expected error containing: {message_fragment}")


def write_mod(directory: Path, name: str, source: str):
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        count_coupling = write_mod(
            tmpdir,
            "count_coupling.mod",
            """
MIND {
  COUPLING count_coupling
  READ S
  WRITE drive
}

EDGE {
  drive += (S * 0.0) + 1.0;
}
""",
        )
        copy_region = "S = drive"
        network = ms.Network(
            labels=["a", "b", "c"],
            weights=[
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ],
            delays=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        for roi in network.rois():
            roi.use(copy_region, inputs={"drive": 0.0}, exposures="S")

        for target in network.rois():
            for source in network.rois():
                target.connect(source, count_coupling)

        require_raises(
            lambda: network.roi("a").connect("b", count_coupling),
            "ROI.connect source must be a ROI handle",
        )
        require_raises(
            lambda: network.roi("a").connect(1, count_coupling),
            "ROI.connect source must be a ROI handle",
        )
        require_raises(
            lambda: network.roi("a").use(count_coupling, inputs={"drive": 0.0}, exposures="S"),
            "owner equations must be a string",
        )

        result = ms.MacroSimulator(network, dt_macro=0.1).run(0.1)
        values = result.exposures.values
        sample_width = result.exposures.recorded_roi_count * result.exposures.exposure_count
        final_values = values[sample_width : sample_width * 2]

        expected = [1.0, 2.0, 0.0]
        require(len(final_values) == len(expected), "unexpected final exposure width")
        for got, want in zip(final_values, expected):
            require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-7), f"coupling value {got} != {want}")

        one_roi = ms.Network(
            labels=["roi"],
            weights=[[0.0]],
            delays=[[1.0]],
        )
        one_roi.roi(0).use("S = drive", inputs={"drive": 0.0}, exposures="S")
        def load_unused_param():
            path = write_mod(
                tmpdir,
                "unused_param.mod",
                """
MIND {
  COUPLING unused_param
  READ S
  WRITE drive
}
PARAMETER {
  G = 1.0
}
EDGE {
  drive += S;
}
""",
            )
            return one_roi.roi(0).connect(one_roi.roi(0), path)
        require_raises(
            load_unused_param,
            "declared param is unused: G",
        )
        def load_unused_port():
            path = write_mod(
                tmpdir,
                "unused_port.mod",
                """
MIND {
  MICRO_INPUT unused_port
  READ drive
  EMIT afferent
}
INPUT {
  x = drive;
}
""",
            )
            return one_roi.roi(0).connect(one_roi.roi(0), path)
        require_raises(
            load_unused_port,
            "declared EMIT port is unused: afferent",
        )
        def load_unused_output_state():
            path = write_mod(
                tmpdir,
                "unused_output_state.mod",
                """
MIND {
  MICRO_OUTPUT unused_output_state
  WRITE S
}
STATE {
  count = 0.0
}
BREAKPOINT {
  S = 0.0;
}
""",
            )
            return one_roi.roi(0).connect(one_roi.roi(0), path)
        require_raises(
            load_unused_output_state,
            "declared state is unused: count",
        )


if __name__ == "__main__":
    main()
