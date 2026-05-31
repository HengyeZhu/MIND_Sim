#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json_field(data, key: str) -> dict:
    if key not in data.files:
        return {}
    return json.loads(str(data[key].tolist()))


def binned_counts(times_ms: np.ndarray, duration_ms: float, bin_ms: float) -> np.ndarray:
    if bin_ms <= 0.0:
        raise ValueError("bin_ms must be positive")
    edges = np.arange(0.0, float(duration_ms) + float(bin_ms), float(bin_ms), dtype=float)
    if edges[-1] < float(duration_ms):
        edges = np.append(edges, float(duration_ms))
    counts, _ = np.histogram(np.asarray(times_ms, dtype=float), bins=edges)
    return counts.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original NEURON and MIND_Sim CA3 validation outputs.")
    parser.add_argument("original", type=Path)
    parser.add_argument("mind_sim", type=Path)
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    original = np.load(args.original, allow_pickle=True)
    mind = np.load(args.mind_sim, allow_pickle=True)
    if list(original["record_labels"]) != list(mind["record_labels"]):
        raise SystemExit("record_labels differ")
    labels = [str(x) for x in original["record_labels"]]
    original_v = np.asarray(original["voltages_mv"], dtype=float)
    mind_v = np.asarray(mind["voltages_mv"], dtype=float)
    per_trace = []
    voltage_shape_match = original_v.shape == mind_v.shape
    if voltage_shape_match:
        diff = mind_v - original_v
        global_max_abs = float(np.abs(diff).max(initial=0.0))
        global_rms = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
        for i, label in enumerate(labels):
            abs_diff = np.abs(diff[i])
            per_trace.append(
                {
                    "label": label,
                    "max_abs_mv": float(abs_diff.max(initial=0.0)),
                    "rms_mv": float(np.sqrt(np.mean(abs_diff**2))) if abs_diff.size else 0.0,
                    "final_original_mv": float(original_v[i, -1]),
                    "final_mind_sim_mv": float(mind_v[i, -1]),
                }
            )
    else:
        original_final = np.asarray(original["record_final_mv"], dtype=float) if "record_final_mv" in original.files else original_v[:, -1]
        mind_final = np.asarray(mind["record_final_mv"], dtype=float) if "record_final_mv" in mind.files else mind_v[:, -1]
        final_diff = mind_final - original_final
        global_max_abs = None
        global_rms = None
        for i, label in enumerate(labels):
            per_trace.append(
                {
                    "label": label,
                    "trajectory_shape_original": list(original_v.shape),
                    "trajectory_shape_mind_sim": list(mind_v.shape),
                    "final_abs_mv": float(abs(final_diff[i])),
                    "final_original_mv": float(original_final[i]),
                    "final_mind_sim_mv": float(mind_final[i]),
                }
            )

    report = {
        "original": str(args.original),
        "mind_sim": str(args.mind_sim),
        "samples": int(original_v.shape[1]),
        "voltage_shape_match": bool(voltage_shape_match),
        "global_max_abs_mv": global_max_abs,
        "global_rms_mv": global_rms,
        "original_spikes": int(original["spike_times_ms"].shape[0]),
        "mind_sim_spikes": int(mind["spike_times_ms"].shape[0]),
        "original_population_spike_counts": load_json_field(original, "population_spike_counts_json"),
        "mind_sim_population_spike_counts": load_json_field(mind, "population_spike_counts_json"),
        "original_population_rate_hz": load_json_field(original, "population_rate_hz_json"),
        "mind_sim_population_rate_hz": load_json_field(mind, "population_rate_hz_json"),
        "per_trace": per_trace,
    }
    if original["spike_times_ms"].shape == mind["spike_times_ms"].shape:
        spike_dt = np.asarray(mind["spike_times_ms"], dtype=float) - np.asarray(original["spike_times_ms"], dtype=float)
        report["spike_time_max_abs_ms"] = float(np.abs(spike_dt).max(initial=0.0))
        report["spike_gid_equal"] = bool(np.array_equal(original["spike_gids"], mind["spike_gids"]))
    else:
        report["spike_time_max_abs_ms"] = None
        report["spike_gid_equal"] = False
    duration_ms = max(float(original["duration_ms"]), float(mind["duration_ms"]))
    original_counts = binned_counts(original["spike_times_ms"], duration_ms, float(args.bin_ms))
    mind_counts = binned_counts(mind["spike_times_ms"], duration_ms, float(args.bin_ms))
    if original_counts.shape == mind_counts.shape:
        diff_counts = mind_counts - original_counts
        report["binned_spike_count_bin_ms"] = float(args.bin_ms)
        report["binned_spike_count_max_abs"] = float(np.abs(diff_counts).max(initial=0.0))
        report["binned_spike_count_rms"] = float(np.sqrt(np.mean(diff_counts**2))) if diff_counts.size else 0.0
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
