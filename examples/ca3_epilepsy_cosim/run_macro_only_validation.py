#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from ca3_mind_sim_api import DT_MS, HERE, ensure_mind_sim_import
from run_vep_ca3_cosim import (
    EXPOSURE_NAMES,
    MIND_MOD_DIR,
    default_connectivity,
    exposure_array,
    initial_x0,
    load_connectivity,
)


ensure_mind_sim_import()
import mind_sim as ms  # noqa: E402


def build_initial_state(labels: list[str], ca3_label: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(args.seed))
    x0 = np.asarray(
        [
            initial_x0(label, ca3_label, args.ca3_x0, args.propagation_x0, args.healthy_x0)
            for label in labels
        ],
        dtype=float,
    )
    x = x0 + 0.02 * rng.standard_normal(len(labels))
    z = np.zeros(len(labels), dtype=float)
    return x0, x, z


def build_mind_macro_network(
    labels: list[str],
    weights: np.ndarray,
    delays: np.ndarray,
    x0: np.ndarray,
    initial_x: np.ndarray,
    initial_z: np.ndarray,
    args: argparse.Namespace,
):
    network = ms.Network(
        labels=labels,
        weights=weights.tolist(),
        delays=delays.tolist(),
        inputs=["coupled_x"],
        exposures=EXPOSURE_NAMES,
    )
    network.record(rois="all")
    vep_owner = """
  dx/dt = (1.0 - x*x*x - 2.0*x*x - z + i_ext + global_coupling * coupled_x) / tau_x_ms;
  dz/dt = (4.0 * (x - x0) - z) / tau_z_ms;
"""
    for index, roi in enumerate(network.rois()):
        roi.initial_output({"x": float(initial_x[index]), "z": float(initial_z[index])})
        roi.use(
            vep_owner,
            inputs={"coupled_x": 0.0},
            exposures=["x", "z"],
            state={"x": float(initial_x[index]), "z": float(initial_z[index])},
            params={
                "x0": float(x0[index]),
                "global_coupling": float(args.global_coupling),
                "tau_x_ms": float(args.tau_x_ms),
                "tau_z_ms": float(args.tau_z_ms),
                "i_ext": float(args.i_ext),
            },
            name="vep_reduced",
        )

    coupling_rule = MIND_MOD_DIR / "vep_x_coupling.mod"
    for target in network.rois():
        for source in network.rois():
            target.connect(source, coupling_rule)
    return network


def numpy_reference(
    *,
    weights: np.ndarray,
    delays: np.ndarray,
    x0: np.ndarray,
    initial_x: np.ndarray,
    initial_z: np.ndarray,
    duration_ms: float,
    dt_ms: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roi_count = len(initial_x)
    step_count = int(math.ceil(float(duration_ms) / float(dt_ms) - 1e-12))
    times = np.asarray([min(float(duration_ms), step * float(dt_ms)) for step in range(step_count + 1)], dtype=float)
    delay_steps = np.rint(delays / float(dt_ms)).astype(np.int64)
    max_delay_steps = int(delay_steps[weights != 0.0].max(initial=0))
    history_capacity = max(1, max_delay_steps + 1)

    x = np.asarray(initial_x, dtype=float).copy()
    z = np.asarray(initial_z, dtype=float).copy()
    history_x = np.zeros((history_capacity, roi_count), dtype=float)
    history_x[:] = x

    out_x = np.zeros((step_count + 1, roi_count), dtype=float)
    out_z = np.zeros((step_count + 1, roi_count), dtype=float)
    out_coupled = np.zeros((step_count + 1, roi_count), dtype=float)
    out_x[0] = x
    out_z[0] = z

    def coupling_for(step: int) -> np.ndarray:
        coupled = np.zeros(roi_count, dtype=float)
        current_slot = int(step) % history_capacity
        for target in range(roi_count):
            for source in range(roi_count):
                weight = float(weights[target, source])
                if weight == 0.0:
                    continue
                delay_slot = (history_capacity - int(delay_steps[target, source]) % history_capacity) % history_capacity
                source_slot = current_slot + delay_slot
                if source_slot >= history_capacity:
                    source_slot -= history_capacity
                coupled[target] += weight * history_x[source_slot, source]
        return coupled

    coupled_x = coupling_for(0)
    out_coupled[0] = coupled_x
    for step in range(step_count):
        start_time = step * float(dt_ms)
        dt = min(float(duration_ms), start_time + float(dt_ms)) - start_time
        new_x = x + dt * (
            1.0
            - x * x * x
            - 2.0 * x * x
            - z
            + float(args.i_ext)
            + float(args.global_coupling) * coupled_x
        ) / float(args.tau_x_ms)
        # MIND Sim owner code is sequential: the z equation sees the just-updated x.
        new_z = z + dt * (4.0 * (new_x - x0) - z) / float(args.tau_z_ms)
        x = new_x
        z = new_z
        out_x[step + 1] = x
        out_z[step + 1] = z
        history_x[(step + 1) % history_capacity] = x
        coupled_x = coupling_for(step + 1)
        out_coupled[step + 1] = coupled_x
    return times, out_x, out_z, out_coupled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MIND Sim reduced VEP macro-only dynamics against a NumPy reference.")
    parser.add_argument("--connectivity-npz", type=Path, default=None)
    parser.add_argument("--ca3-label", default="Left-CA3")
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--macro-dt-ms", type=float, default=DT_MS)
    parser.add_argument("--exchange-window-ms", type=float, default=10.0, help="Default delay for the built-in scaffold or connectivity files without delays.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ca3-x0", type=float, default=-1.6)
    parser.add_argument("--propagation-x0", type=float, default=-1.9)
    parser.add_argument("--healthy-x0", type=float, default=-2.4)
    parser.add_argument("--global-coupling", type=float, default=0.35)
    parser.add_argument("--tau-x-ms", type=float, default=50.0)
    parser.add_argument("--tau-z-ms", type=float, default=2857.0)
    parser.add_argument("--i-ext", type=float, default=3.1)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "macro_only_validation.npz")
    parser.add_argument("--report", type=Path, default=HERE / "outputs" / "macro_only_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    if args.macro_dt_ms <= 0.0:
        raise SystemExit("--macro-dt-ms must be positive")

    labels, weights, delays = load_connectivity(args.connectivity_npz, args.exchange_window_ms)
    if args.connectivity_npz is None:
        labels, weights, delays = default_connectivity(args.exchange_window_ms)
    if args.ca3_label not in labels:
        raise SystemExit(f"{args.ca3_label!r} not found in connectivity labels")
    x0, initial_x, initial_z = build_initial_state(labels, args.ca3_label, args)

    build_start = time.perf_counter()
    network = build_mind_macro_network(labels, weights, delays, x0, initial_x, initial_z, args)
    build_s = time.perf_counter() - build_start

    run_start = time.perf_counter()
    mind_result = ms.MacroSimulator(network, dt_macro=float(args.macro_dt_ms)).run(float(args.duration_ms))
    mind_run_s = time.perf_counter() - run_start
    mind_cube = exposure_array(mind_result)
    mind_x = mind_cube[:, :, 0]
    mind_z = mind_cube[:, :, 1]
    mind_times = np.asarray(mind_result.times, dtype=float)

    ref_start = time.perf_counter()
    ref_times, ref_x, ref_z, ref_coupled_x = numpy_reference(
        weights=weights,
        delays=delays,
        x0=x0,
        initial_x=initial_x,
        initial_z=initial_z,
        duration_ms=float(args.duration_ms),
        dt_ms=float(args.macro_dt_ms),
        args=args,
    )
    ref_run_s = time.perf_counter() - ref_start

    if mind_x.shape != ref_x.shape:
        raise SystemExit(f"shape mismatch: MIND {mind_x.shape} vs reference {ref_x.shape}")
    time_diff = mind_times - ref_times
    x_diff = mind_x - ref_x
    z_diff = mind_z - ref_z
    report = {
        "samples": int(mind_x.shape[0]),
        "roi_count": int(mind_x.shape[1]),
        "time_max_abs_ms": float(np.abs(time_diff).max(initial=0.0)),
        "x_max_abs": float(np.abs(x_diff).max(initial=0.0)),
        "x_rms": float(np.sqrt(np.mean(x_diff**2))) if x_diff.size else 0.0,
        "z_max_abs": float(np.abs(z_diff).max(initial=0.0)),
        "z_rms": float(np.sqrt(np.mean(z_diff**2))) if z_diff.size else 0.0,
        "passed": bool(
            np.abs(time_diff).max(initial=0.0) <= args.atol
            and np.abs(x_diff).max(initial=0.0) <= args.atol
            and np.abs(z_diff).max(initial=0.0) <= args.atol
        ),
        "mind_build_s": float(build_s),
        "mind_run_s": float(mind_run_s),
        "reference_run_s": float(ref_run_s),
    }

    output = args.output.expanduser()
    if not output.is_absolute():
        output = (HERE / output).resolve()
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = (HERE / report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        time_ms=mind_times,
        mind_x=mind_x,
        mind_z=mind_z,
        reference_time_ms=ref_times,
        reference_x=ref_x,
        reference_z=ref_z,
        reference_coupled_x=ref_coupled_x,
        x0=x0,
        initial_x=initial_x,
        initial_z=initial_z,
        report_json=json.dumps(report, sort_keys=True),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
