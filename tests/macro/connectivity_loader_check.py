from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_close(got: float, want: float, message: str) -> None:
    require(math.isclose(got, want, rel_tol=0.0, abs_tol=1e-6), message)


def write_connectivity_dir(path: Path) -> None:
    path.mkdir()
    (path / "region_labels.txt").write_text("a b c\n", encoding="utf-8")
    (path / "weights.txt").write_text(
        "0.0 2.0 0.0\n"
        "1.0 0.0 3.0\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (path / "delays.txt").write_text(
        "0.0 0.2 0.0\n"
        "0.1 0.0 0.3\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )


def write_connectivity_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("region_labels.txt", "x,y\n")
        archive.writestr("weights.txt", "0,4\n5,0\n")
        archive.writestr("tract_lengths.txt", "0,8\n2,0\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        connectivity_dir = root / "connectivity"
        write_connectivity_dir(connectivity_dir)
        conn = ms.load_connectivity(connectivity_dir)
        require(conn.roi_count() == 3, "directory ROI count mismatch")
        require(conn.roi_index("b") == 1, "directory ROI label lookup mismatch")
        require_close(conn.weight_at(1, 2), 3.0, "directory weight mismatch")
        require_close(conn.delay_at(1, 2), 0.3, "directory delay mismatch")

        connectivity_zip = root / "connectivity.zip"
        write_connectivity_zip(connectivity_zip)
        conn = ms.load_connectivity(
            connectivity_zip,
            conduction_speed=2.0,
            min_tract_length=1.0,
        )
        require(conn.roi_count() == 2, "zip ROI count mismatch")
        require_close(conn.delay_at(0, 1), 4.0, "zip tract delay mismatch")
        require_close(conn.delay_at(0, 0), 0.5, "zip minimum tract delay mismatch")

        network = ms.load_network(
            connectivity_zip,
            inputs=["drive"],
            exposures=["S"],
            conduction_speed=2.0,
            min_tract_length=1.0,
        )
        roi = network.roi("x")
        require(hasattr(roi, "use"), "loaded Network ROI is not a frontend ROI handle")
        require(roi.label == "x", "loaded Network ROI label mismatch")


if __name__ == "__main__":
    main()
