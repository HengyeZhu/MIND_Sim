#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ca3_mind_sim_api import (
    BAS_COUNT,
    DT_MS,
    HERE,
    OLM_COUNT,
    PYR_COUNT,
    V_INIT_MV,
    build_mind_ca3,
    ensure_mind_sim_import,
)


ensure_mind_sim_import()
import mind_sim as ms  # noqa: E402


DEFAULT_LABELS = [
    "Left-entorhinal",
    "Left-CA3",
    "Left-CA1",
    "Left-subiculum",
    "Right-CA3",
]
EXPOSURE_NAMES = ["x", "z", "rate"]
MIND_MOD_DIR = HERE / "mind_mod"
CA3_CELL_COUNT = PYR_COUNT + BAS_COUNT + OLM_COUNT


@contextlib.contextmanager
def suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def default_connectivity(default_delay_ms: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    weights = np.array(
        [
            [0.00, 0.35, 0.10, 0.20, 0.03],
            [0.20, 0.00, 0.45, 0.10, 0.08],
            [0.05, 0.35, 0.00, 0.30, 0.05],
            [0.20, 0.10, 0.35, 0.00, 0.03],
            [0.03, 0.08, 0.05, 0.03, 0.00],
        ],
        dtype=float,
    )
    delays = np.where(weights > 0.0, float(default_delay_ms), 0.0)
    return list(DEFAULT_LABELS), weights, delays


def load_connectivity(path: Path | None, default_delay_ms: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    if path is None:
        return default_connectivity(default_delay_ms)
    data = np.load(path, allow_pickle=True)
    labels = [str(x) for x in data["labels"].tolist()]
    weights = np.asarray(data["weights"], dtype=float)
    if weights.shape != (len(labels), len(labels)):
        raise ValueError("connectivity weights must have shape len(labels) x len(labels)")
    if "delays" in data:
        delays = np.asarray(data["delays"], dtype=float)
    else:
        delays = np.where(weights > 0.0, float(default_delay_ms), 0.0)
    if delays.shape != weights.shape:
        raise ValueError("connectivity delays must have the same shape as weights")
    np.fill_diagonal(weights, 0.0)
    np.fill_diagonal(delays, 0.0)
    return labels, weights, delays


def initial_x0(label: str, ca3_label: str, ca3_x0: float, propagation_x0: float, healthy_x0: float) -> float:
    if label == ca3_label:
        return ca3_x0
    if any(token in label for token in ("CA1", "subiculum", "entorhinal", "CA3")):
        return propagation_x0
    return healthy_x0


def exposure_array(result) -> np.ndarray:
    record = result.exposures
    values = np.asarray(record.values, dtype=float)
    return values.reshape(int(record.sample_count), int(record.recorded_roi_count), int(record.exposure_count))


def sorted_spikes(table) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(table.time, dtype=float)
    gids = np.asarray(table.gid, dtype=np.int32)
    if times.size == 0:
        return times, gids
    order = np.lexsort((gids, times))
    return times[order], gids[order]


def delayed_ca3_input(x: np.ndarray, weights: np.ndarray, delays: np.ndarray, ca3_index: int, dt_macro_ms: float) -> np.ndarray:
    out = np.zeros(x.shape[0], dtype=float)
    for source in range(x.shape[1]):
        weight = float(weights[ca3_index, source])
        if weight == 0.0:
            continue
        delay_steps = int(round(float(delays[ca3_index, source]) / float(dt_macro_ms)))
        for sample in range(x.shape[0]):
            source_sample = max(0, sample - delay_steps)
            out[sample] += weight * x[source_sample, source]
    return out


def drive_from_input(
    ca3_input: np.ndarray,
    *,
    base_hz: float,
    gain_hz: float,
    max_hz: float,
    threshold: float,
    slope: float,
) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-float(slope) * (ca3_input - float(threshold))))
    return np.clip(float(base_hz) + float(gain_hz) * sigmoid, 0.0, float(max_hz))


def binned_rate(times_ms: np.ndarray, spike_times_ms: np.ndarray, cells: int) -> np.ndarray:
    if times_ms.size < 2:
        return np.zeros(0, dtype=float)
    counts, _ = np.histogram(spike_times_ms, bins=times_ms)
    widths_s = np.diff(times_ms) / 1000.0
    return counts.astype(float) / np.maximum(widths_s, 1e-12) / float(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIND Sim VEP neural-mass plus ModelDB 186768 CA3 micro cosimulation.")
    parser.add_argument("--connectivity-npz", type=Path, default=None, help="Optional npz with labels, weights, and optional delays arrays.")
    parser.add_argument("--ca3-label", default="Left-CA3")
    parser.add_argument("--duration-ms", type=float, default=300.0)
    parser.add_argument("--macro-dt-ms", type=float, default=DT_MS)
    parser.add_argument("--exchange-window-ms", type=float, default=10.0)
    parser.add_argument("--connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=os.environ.get("MIND_SIM_DEVICE", "cpu"))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MIND_SIM_NUM_THREADS", "1") or "1"))
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "vep_ca3_mindsim_cosim.npz")
    parser.add_argument("--ca3-x0", type=float, default=-1.6)
    parser.add_argument("--propagation-x0", type=float, default=-1.9)
    parser.add_argument("--healthy-x0", type=float, default=-2.4)
    parser.add_argument("--global-coupling", type=float, default=0.35)
    parser.add_argument("--tau-x-ms", type=float, default=50.0)
    parser.add_argument("--tau-z-ms", type=float, default=2857.0)
    parser.add_argument("--i-ext", type=float, default=3.1)
    parser.add_argument("--drive-base-hz", type=float, default=1.0)
    parser.add_argument("--drive-gain-hz", type=float, default=45.0)
    parser.add_argument("--drive-max-hz", type=float, default=120.0)
    parser.add_argument("--drive-threshold", type=float, default=-0.35)
    parser.add_argument("--drive-slope", type=float, default=4.0)
    parser.add_argument("--drive-weight", type=float, default=0.02e-3)
    parser.add_argument("--drive-delay-ms", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    if args.exchange_window_ms <= 0.0:
        raise SystemExit("--exchange-window-ms must be positive")
    if args.macro_dt_ms <= 0.0:
        raise SystemExit("--macro-dt-ms must be positive")

    labels, weights, delays = load_connectivity(args.connectivity_npz, args.exchange_window_ms)
    if args.ca3_label not in labels:
        raise SystemExit(f"{args.ca3_label!r} not found in connectivity labels")
    ca3_index = labels.index(args.ca3_label)
    min_positive_delay = float(np.min(delays[delays > 0.0])) if np.any(delays > 0.0) else 0.0
    if min_positive_delay <= 0.0:
        raise SystemExit("connectivity delays must contain at least one positive delay")
    if args.exchange_window_ms > min_positive_delay + 1e-9:
        raise SystemExit("--exchange-window-ms must not exceed the minimum positive connectivity delay")

    output = args.output.expanduser()
    if not output.is_absolute():
        output = (HERE / output).resolve()
    else:
        output = output.resolve()

    pre_start = time.perf_counter()
    with suppress_native_output(args.quiet):
        build = build_mind_ca3(
            connections=bool(args.connections),
            include_psr=False,
            wseed=int(args.wseed),
            device=args.device,
            num_threads=int(args.num_threads),
        )
        micro = build.sim
        micro_net = micro.network()
        afferent = micro_net.spike_inputs(PYR_COUNT)
        for gid in range(PYR_COUNT):
            micro_net.spike_connect(
                afferent[gid],
                build.synapses[(gid, "Adend3AMPAf")],
                float(args.drive_weight),
                float(args.drive_delay_ms),
            )
        micro.build_microcircuit()
        micro.finitialize(V_INIT_MV)

        network = ms.Network(
            labels=labels,
            weights=weights.tolist(),
            delays=delays.tolist(),
            inputs=["coupled_x", "ca3_input"],
            exposures=EXPOSURE_NAMES,
        )
        network.record(rois="all")
        ca3_gid_ranges = [
            (build.populations[name].gid_begin, build.populations[name].gid_end)
            for name in ("PYR", "BAS", "OLM")
        ]
        micro_owner = ms.MicroCircuit(micro).bind_roi(
            args.ca3_label,
            gid_ranges=ca3_gid_ranges,
            ports={"afferent": afferent},
        )
        network.use_micro(micro_owner)

        vep_coupling = MIND_MOD_DIR / "vep_x_coupling.mod"
        ca3_input_coupling = MIND_MOD_DIR / "ca3_input_coupling.mod"
        ca3_input_to_spikes = MIND_MOD_DIR / "ca3_input_to_spikes.mod"
        ca3_spikes_to_vep = MIND_MOD_DIR / "ca3_spikes_to_vep.mod"
        vep_owner = """
  dx/dt = (1.0 - x*x*x - 2.0*x*x - z + i_ext + global_coupling * coupled_x) / tau_x_ms;
  dz/dt = (4.0 * (x - x0) - z) / tau_z_ms;
"""
        rng = np.random.default_rng(int(args.seed))
        for roi in network.rois():
            x0 = initial_x0(roi.label, args.ca3_label, args.ca3_x0, args.propagation_x0, args.healthy_x0)
            initial_x = x0 + 0.02 * float(rng.standard_normal())
            if roi.index == ca3_index:
                roi.initial_output({"x": initial_x, "z": 0.0, "rate": 0.0})
                continue
            roi.initial_output({"x": initial_x, "z": 0.0})
            roi.use(
                vep_owner,
                inputs={"coupled_x": 0.0},
                exposures=["x", "z"],
                state={"x": initial_x, "z": 0.0},
                params={
                    "x0": x0,
                    "global_coupling": float(args.global_coupling),
                    "tau_x_ms": float(args.tau_x_ms),
                    "tau_z_ms": float(args.tau_z_ms),
                    "i_ext": float(args.i_ext),
                },
                name="vep_reduced",
            )

        ca3_roi = network.roi(args.ca3_label)
        for target in network.rois():
            if target.index == ca3_index:
                for source in network.rois():
                    target.connect(source, ca3_input_coupling)
                continue
            for source in network.rois():
                target.connect(source, vep_coupling)

        ca3_roi.connect(
            ca3_roi,
            ca3_input_to_spikes,
            params={
                "cells": float(PYR_COUNT),
                "base_hz": float(args.drive_base_hz),
                "gain_hz": float(args.drive_gain_hz),
                "max_rate_hz": float(args.drive_max_hz),
                "threshold": float(args.drive_threshold),
                "slope": float(args.drive_slope),
            },
            random={
                "rng": {
                    "state": {"seed": float(int(args.seed) + 10_000)},
                    "uniform": r"""
    std::uint64_t rng = (std::uint64_t) seed;
    rng = rng * 2862933555777941757ULL + 3037000493ULL;
    seed = (double) (rng & 9007199254740991ULL);
    return seed / 9007199254740992.0;
""",
                }
            },
        )
        ca3_roi.connect(
            ca3_roi,
            ca3_spikes_to_vep,
            params={
                "cells": float(CA3_CELL_COUNT),
                "tau_activity_ms": 50.0,
                "tau_z_ms": float(args.tau_z_ms),
                "x_baseline": -1.8,
                "x_gain": 2.0,
                "z_gain": 1.0,
            },
        )
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    with suppress_native_output(args.quiet):
        try:
            simulator = ms.Simulator(
                network,
                dt_micro=DT_MS,
                dt_macro=float(args.macro_dt_ms),
                exchange_window=float(args.exchange_window_ms),
            )
        except TypeError as exc:
            if "exchange_window" not in str(exc):
                raise
            simulator = ms.Simulator(
                network,
                dt_micro=DT_MS,
                dt_macro=float(args.macro_dt_ms),
                batch_window=float(args.exchange_window_ms),
            )
        result = simulator.run(float(args.duration_ms))
    run_s = time.perf_counter() - run_start

    times_ms = np.asarray(result.times, dtype=float)
    cube = exposure_array(result)
    x = cube[:, :, 0]
    z = cube[:, :, 1]
    rate = cube[:, :, 2]
    ca3_macro_input = delayed_ca3_input(x, weights, delays, ca3_index, float(args.macro_dt_ms))
    ca3_drive_hz = drive_from_input(
        ca3_macro_input,
        base_hz=float(args.drive_base_hz),
        gain_hz=float(args.drive_gain_hz),
        max_hz=float(args.drive_max_hz),
        threshold=float(args.drive_threshold),
        slope=float(args.drive_slope),
    )
    spike_times_ms, spike_gids = sorted_spikes(result.micro_spikes_by_roi[ca3_index])
    ca3_rate_hz = binned_rate(times_ms, spike_times_ms, CA3_CELL_COUNT)

    metadata = {
        "source": "MIND Sim reduced VEP neural mass + MIND Sim API rewrite of ModelDB 186768 CA3",
        "ca3_label": args.ca3_label,
        "ca3_micro_model": "ModelDB 186768 CA3, MIND Sim API/CoreNEURON rewrite",
        "macro_model": "reduced VEP/Epileptor-style x-z neural mass",
        "duration_ms": float(args.duration_ms),
        "dt_micro_ms": float(DT_MS),
        "dt_macro_ms": float(args.macro_dt_ms),
        "exchange_window_ms": float(args.exchange_window_ms),
        "connections": bool(args.connections),
        "wseed": int(args.wseed),
        "seed": int(args.seed),
        "notes": "CA3 ROI is replaced by aggregate PYR+BAS+OLM spike-derived micro output; macro input to CA3 is transformed into external AMPA events on PYR Adend3 synapses.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        exposure_names=np.asarray(EXPOSURE_NAMES, dtype=object),
        time_ms=times_ms,
        macro_x=x,
        macro_z=z,
        ca3_micro_rate_proxy_hz=rate[:, ca3_index],
        ca3_drive_hz=ca3_drive_hz,
        ca3_macro_input=ca3_macro_input,
        ca3_macro_output_x=x[:, ca3_index],
        ca3_macro_output_z=z[:, ca3_index],
        rate_time_ms=times_ms[1:],
        ca3_micro_population_rate_hz=ca3_rate_hz,
        spike_times_ms=spike_times_ms,
        spike_gids=spike_gids,
        timing_s=np.asarray([pre_run_s, run_s, pre_run_s + run_s], dtype=float),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"output={output}")
    print("backend=mind_sim")
    print(f"device={args.device}")
    print(f"num_threads={args.num_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")
    print(f"spikes={len(spike_times_ms)}")


if __name__ == "__main__":
    main()
