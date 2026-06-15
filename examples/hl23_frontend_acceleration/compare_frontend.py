#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def scalar(data: Any, key: str) -> float:
    if key == "pre_run_s" and key not in data:
        build = scalar(data, "build_s")
        init = scalar(data, "finitialize_s")
        if np.isfinite(build) and np.isfinite(init):
            return build + init
        return float("nan")
    if key not in data:
        return float("nan")
    return float(np.asarray(data[key]).reshape(-1)[0])


def numeric_trace_keys(data: Any) -> set[str]:
    keys: set[str] = set()
    for key in data.files:
        if key.isdigit():
            keys.add(key)
    return keys


def array(data: Any, key: str, *, dtype: Any = float) -> np.ndarray:
    if key not in data:
        return np.asarray([], dtype=dtype)
    return np.asarray(data[key], dtype=dtype)


def cell_type_for_gid(data: Any, gid: int) -> str:
    offset = 0
    for name in ("PYR", "SST", "PV", "VIP"):
        count = int(scalar(data, f"num_cells_{name.lower()}"))
        if gid < offset + count:
            return name
        offset += count
    return "unknown"


def spike_summary(data: Any) -> dict[str, Any]:
    times = array(data, "spike_times", dtype=float)
    gids = array(data, "spike_gids", dtype=int)
    by_type = {name: 0 for name in ("PYR", "SST", "PV", "VIP")}
    by_type["unknown"] = 0
    for gid in gids:
        by_type[cell_type_for_gid(data, int(gid))] += 1
    first_ms = float(times[0]) if times.size else float("nan")
    last_ms = float(times[-1]) if times.size else float("nan")
    return {
        "count": int(times.size),
        "first_ms": first_ms,
        "last_ms": last_ms,
        "by_type": by_type,
    }


def compare_spikes(mind: Any, coreneuron: Any) -> dict[str, Any]:
    mind_times = array(mind, "spike_times", dtype=float)
    core_times = array(coreneuron, "spike_times", dtype=float)
    mind_gids = array(mind, "spike_gids", dtype=int)
    core_gids = array(coreneuron, "spike_gids", dtype=int)
    n = min(mind_times.size, core_times.size)
    max_time_delta_ms = float("nan")
    gid_prefix_equal = False
    if n > 0:
        max_time_delta_ms = float(np.max(np.abs(mind_times[:n] - core_times[:n])))
        gid_prefix_equal = bool(np.array_equal(mind_gids[:n], core_gids[:n]))
    return {
        "mind": spike_summary(mind),
        "coreneuron": spike_summary(coreneuron),
        "counts_equal": bool(mind_times.size == core_times.size),
        "gid_prefix_equal": gid_prefix_equal,
        "max_time_delta_ms_over_common_prefix": max_time_delta_ms,
    }


def compare_traces(left: Any, right: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in sorted(numeric_trace_keys(left) & numeric_trace_keys(right), key=int):
        a = np.asarray(left[key], dtype=float)
        b = np.asarray(right[key], dtype=float)
        n = min(a.size, b.size)
        if n == 0:
            continue
        diff = a[:n] - b[:n]
        out[key] = {
            "points": int(n),
            "max_abs_mV": float(np.max(np.abs(diff))),
            "rms_mV": float(np.sqrt(np.mean(diff * diff))),
            "mind_first_mV": float(a[0]),
            "coreneuron_first_mV": float(b[0]),
            "mind_last_mV": float(a[n - 1]),
            "coreneuron_last_mV": float(b[n - 1]),
        }
    return out


def timing_summary(mind: Any, coreneuron: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in ("build_s", "finitialize_s", "pre_run_s", "run_s", "sum_s"):
        m = scalar(mind, key)
        c = scalar(coreneuron, key)
        out[key] = {
            "mind": m,
            "coreneuron": c,
            "speedup_vs_coreneuron": c / m if m > 0.0 and np.isfinite(c) else float("nan"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MIND_Sim and CoreNEURON HL23 example outputs.")
    parser.add_argument("--mind", type=Path, required=True)
    parser.add_argument("--coreneuron", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.mind.is_file():
        raise SystemExit(f"missing MIND_Sim output: {args.mind}")
    if not args.coreneuron.is_file():
        raise SystemExit(f"missing CoreNEURON output: {args.coreneuron}")

    mind = np.load(args.mind, allow_pickle=True)
    coreneuron = np.load(args.coreneuron, allow_pickle=True)
    result = {
        "outputs": {"mind": str(args.mind), "coreneuron": str(args.coreneuron)},
        "timing": timing_summary(mind, coreneuron),
        "spikes": compare_spikes(mind, coreneuron),
        "voltage": compare_traces(mind, coreneuron),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("Timing summary (speedup is CoreNEURON seconds / MIND_Sim seconds):")
    for key, values in result["timing"].items():
        print(
            f"  {key}: mind={values['mind']:.6f}s "
            f"coreneuron={values['coreneuron']:.6f}s speedup={values['speedup_vs_coreneuron']:.3f}"
        )
    print("Voltage differences:")
    for gid, values in result["voltage"].items():
        print(
            f"  gid {gid}: points={values['points']} "
            f"max_abs={values['max_abs_mV']:.6g}mV rms={values['rms_mV']:.6g}mV"
        )
    spikes = result["spikes"]
    print(
        "Spikes: "
        f"mind={spikes['mind']['count']} coreneuron={spikes['coreneuron']['count']} "
        f"counts_equal={spikes['counts_equal']} "
        f"max_dt_common={spikes['max_time_delta_ms_over_common_prefix']:.6g}ms"
    )
    print(f"Saved comparison JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
