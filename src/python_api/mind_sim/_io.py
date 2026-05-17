from __future__ import annotations

import math
import zipfile
from collections.abc import Sequence
from pathlib import Path

from . import _native
from ._codegen import Network


def _read_connectivity_files(path: str | Path) -> dict[str, str]:
    root = Path(path)
    filenames = ("region_labels.txt", "weights.txt", "delays.txt", "tract_lengths.txt")
    if root.is_dir():
        return {
            name: (root / name).read_text(encoding="utf-8")
            for name in filenames
            if (root / name).is_file()
        }
    with zipfile.ZipFile(root) as archive:
        entries = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        return {
            name: archive.read(entries[name]).decode("utf-8")
            for name in filenames
            if name in entries
        }


def _parse_connectivity_labels(text: str) -> list[str]:
    labels = text.replace(",", " ").split()
    if not labels:
        raise ValueError("connectivity region_labels.txt is empty")
    return labels


def _parse_numeric_matrix(text: str, name: str) -> list[list[float]]:
    rows = [
        [float(item) for item in line.strip().replace(",", " ").split()]
        for line in text.splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"connectivity {name} is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"connectivity {name} rows must have equal length")
    return rows


def load_connectivity(
    path: str | Path,
    *,
    conduction_speed: float | None = None,
    min_tract_length: float = 0.0,
):
    files = _read_connectivity_files(path)
    for name in ("region_labels.txt", "weights.txt"):
        if name not in files:
            raise FileNotFoundError(f"connectivity file is missing {name}: {path}")
    labels = _parse_connectivity_labels(files["region_labels.txt"])
    weights = _parse_numeric_matrix(files["weights.txt"], "weights.txt")
    if "delays.txt" in files:
        delays = _parse_numeric_matrix(files["delays.txt"], "delays.txt")
    elif "tract_lengths.txt" in files:
        if conduction_speed is None:
            raise ValueError("tract_lengths.txt requires conduction_speed")
        speed = float(conduction_speed)
        if speed <= 0.0 or not math.isfinite(speed):
            raise ValueError("conduction_speed must be positive and finite")
        floor = float(min_tract_length)
        tract_lengths = _parse_numeric_matrix(files["tract_lengths.txt"], "tract_lengths.txt")
        delays = [[max(length, floor) / speed for length in row] for row in tract_lengths]
    else:
        raise FileNotFoundError(f"connectivity file is missing delays.txt or tract_lengths.txt: {path}")
    return _native.Connectivity(labels, weights, delays)


def load_network(
    path: str | Path,
    *,
    inputs: Sequence[str] | str | None = None,
    exposures: Sequence[str] | str | None = None,
    recorded_rois="all",
    conduction_speed: float | None = None,
    min_tract_length: float = 0.0,
) -> Network:
    network = Network(
        connectivity=load_connectivity(
            path,
            conduction_speed=conduction_speed,
            min_tract_length=min_tract_length,
        ),
        inputs=inputs,
        exposures=exposures,
    )
    if recorded_rois is not None:
        network.record(rois=recorded_rois)
    return network
