#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_labels(path: Path) -> list[str]:
    labels: list[str] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if "name" not in header:
            raise SystemExit(f"{path} must be a TSV with a 'name' column")
        name_idx = header.index("name")
        for raw in f:
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) <= name_idx:
                raise SystemExit(f"malformed label row: {raw!r}")
            labels.append(parts[name_idx])
    if not labels:
        raise SystemExit(f"no labels read from {path}")
    return labels


def read_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path.expanduser(), delimiter=",")
    except ValueError:
        matrix = np.loadtxt(path.expanduser())
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SystemExit(f"{path} must contain a square matrix; got shape {matrix.shape}")
    matrix[~np.isfinite(matrix)] = 0.0
    matrix[matrix < 0.0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def normalize_weights(weights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return weights
    out = weights.astype(float, copy=True)
    if mode == "global":
        peak = float(out.max(initial=0.0))
        if peak > 0.0:
            out /= peak
        return out
    if mode == "row":
        row_sum = out.sum(axis=1, keepdims=True)
        np.divide(out, row_sum, out=out, where=row_sum > 0.0)
        return out
    raise ValueError(mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an MRtrix tck2connectome CSV matrix into run_vep_ca3_cosim.py connectivity.npz format."
    )
    parser.add_argument("--weights-csv", type=Path, required=True)
    parser.add_argument("--labels-tsv", type=Path, required=True, help="TSV from build_hybrid_ca3_parcellation.py")
    parser.add_argument("--lengths-csv", type=Path, default=None, help="Optional streamline length matrix from tck2connectome -scale_length")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ca3-label", default="Left-CA3")
    parser.add_argument("--normalize", choices=("none", "global", "row"), default="global")
    parser.add_argument("--conduction-speed-mm-per-ms", type=float, default=5.0, help="5 mm/ms equals 5 m/s")
    parser.add_argument("--default-delay-ms", type=float, default=10.0)
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels_tsv)
    weights = read_matrix(args.weights_csv)
    if weights.shape != (len(labels), len(labels)):
        raise SystemExit(
            f"weights matrix shape {weights.shape} does not match {len(labels)} labels from {args.labels_tsv}"
        )
    weights = normalize_weights(weights, args.normalize)

    if args.lengths_csv is not None:
        lengths = read_matrix(args.lengths_csv)
        if lengths.shape != weights.shape:
            raise SystemExit(f"length matrix shape {lengths.shape} does not match weights shape {weights.shape}")
        if args.conduction_speed_mm_per_ms <= 0.0:
            raise SystemExit("--conduction-speed-mm-per-ms must be positive")
        delays = np.where(weights > 0.0, lengths / float(args.conduction_speed_mm_per_ms), 0.0)
    else:
        delays = np.where(weights > 0.0, float(args.default_delay_ms), 0.0)

    np.fill_diagonal(delays, 0.0)
    if args.ca3_label not in labels:
        matches = [label for label in labels if "CA3" in label.upper()]
        raise SystemExit(f"{args.ca3_label!r} not found in labels; CA3-like labels: {matches}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "MRtrix connectome converted for MIND Sim VEP-CA3 cosim",
        "weights_csv": str(args.weights_csv.expanduser().resolve()),
        "lengths_csv": str(args.lengths_csv.expanduser().resolve()) if args.lengths_csv else None,
        "labels_tsv": str(args.labels_tsv.expanduser().resolve()),
        "label_count": len(labels),
        "ca3_label": args.ca3_label,
        "normalize": args.normalize,
        "conduction_speed_mm_per_ms": float(args.conduction_speed_mm_per_ms),
        "default_delay_ms": float(args.default_delay_ms),
    }
    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )

    if args.metadata_json is not None:
        meta_path = args.metadata_json.expanduser().resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"output={output}")
    print(f"labels={len(labels)}")
    print(f"ca3_index={labels.index(args.ca3_label)}")
    print(f"positive_edges={int(np.count_nonzero(weights))}")
    print(f"weight_max={float(weights.max(initial=0.0)):.6g}")
    print(f"delay_max_ms={float(delays.max(initial=0.0)):.6g}")


if __name__ == "__main__":
    main()
