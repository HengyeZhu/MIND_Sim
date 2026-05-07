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
    except RuntimeError as exc:
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
        write_mod(
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
        network.load_mod_metadata(tmpdir)

        region = ms.RegionRule(
            name="copy_input",
            step="S = drive;",
        )
        for roi in network.rois():
            roi.use(region)

        for target in network.rois():
            for source in network.rois():
                target.connect(source, "count_coupling")

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
        )
        require_raises(
            lambda: strict_network.roi(0).use(ms.RegionRule(
                name="unused_region_param",
                params={"unused": 1.0},
                step="S = drive;",
            )),
            "declared param is unused: unused",
        )
        require_raises(
            lambda: strict_network.roi(0).use(ms.RegionRule(
                name="unused_region_state",
                state={"x": 0.0},
                step="S = drive;",
            )),
            "declared state is unused: x",
        )
        strict_network.roi(0).use(ms.RegionRule(
            name="strict_copy",
            step="S = drive;",
        ))
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
            return strict_network.load_mod_metadata(path)
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
            return strict_network.load_mod_metadata(path)
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
            return strict_network.load_mod_metadata(path)
        require_raises(
            load_unused_output_state,
            "declared state is unused: count",
        )


if __name__ == "__main__":
    main()
