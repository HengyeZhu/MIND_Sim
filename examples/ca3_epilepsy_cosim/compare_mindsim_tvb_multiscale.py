#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NUMERIC_KEYS = [
    "time_ms",
    "macro_x",
    "macro_z",
    "ca3_macro_input",
    "ca3_drive_hz",
    "ca3_macro_output_x",
    "ca3_macro_output_z",
    "ca3_micro_rate_proxy_hz",
    "ca3_micro_population_rate_hz",
]
SPIKE_KEYS = ["spike_times_ms", "spike_gids"]


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "shape": list(a.shape),
        "shape_match": True,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "exact_equal": bool(np.array_equal(a, b)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MIND Sim VEP-CA3 cosim with the TVB-multiscale NetPyNE migration.")
    parser.add_argument("mindsim", type=Path)
    parser.add_argument("tvb_multiscale", type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=Path("outputs/mindsim_vs_tvb_multiscale.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mind = np.load(args.mindsim, allow_pickle=True)
    tvbm = np.load(args.tvb_multiscale, allow_pickle=True)
    report = {
        "mindsim": str(args.mindsim.resolve()),
        "tvb_multiscale": str(args.tvb_multiscale.resolve()),
        "atol": float(args.atol),
        "numeric": {},
        "spikes": {},
    }
    numeric_pass = True
    for key in NUMERIC_KEYS:
        item = metrics(mind[key], tvbm[key])
        if not item.get("shape_match", False) or item.get("max_abs", float("inf")) > float(args.atol):
            numeric_pass = False
        report["numeric"][key] = item

    spike_pass = True
    for key in SPIKE_KEYS:
        equal = bool(np.array_equal(mind[key], tvbm[key]))
        spike_pass = spike_pass and equal
        report["spikes"][key] = {
            "shape_mindsim": list(mind[key].shape),
            "shape_tvb_multiscale": list(tvbm[key].shape),
            "exact_equal": equal,
        }
        if key == "spike_times_ms" and mind[key].shape == tvbm[key].shape:
            diff = np.asarray(mind[key], dtype=float) - np.asarray(tvbm[key], dtype=float)
            report["spikes"][key]["max_abs_ms"] = float(np.max(np.abs(diff))) if diff.size else 0.0

    report["passed"] = bool(numeric_pass and spike_pass)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
