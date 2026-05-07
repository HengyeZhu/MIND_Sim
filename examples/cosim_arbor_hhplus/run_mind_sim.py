#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import mind_sim as ms


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
CONNECTIVITY_FILE = Path.home() / "arbor-tvb-cosim" / "connectivity_mouse.zip"
RESULT_DIR = ROOT / "result" / "cosim_arbor_hhplus"

DEFAULT_CELLS = 100
DEFAULT_DURATION_MS = 2000.0
DT_MS = 0.01
MACRO_DT_MS = 0.01
DEFAULT_BATCH_WINDOW_MS = 0.25
MICRO_ROI = 72
SEED = 1234

PATHOLOGICAL_FRACTION = 1.0
K_BATH_OK = 9.5
K_BATH_BAD = 17.0
ARBOR_RECURRENT_WEIGHT = 0.5
ARBOR_RECURRENT_DELAY_MS = 0.5

MACRO_EVENT_WEIGHT = 0.01
MACRO_EVENT_DELAY_MS = 1.0
INPUT_SPIKE_SCALE = 5.0
MAX_INPUT_RATE_HZ = 150.0
COUPLING_A = 0.096
EXPOSURE_GAIN = 10.0
EXPOSURE_TAU_MS = 100.0
MIN_TRACT_LENGTH_MS = 1.0
CONDUCTION_SPEED = 3.0


@contextlib.contextmanager
def suppress_native_output():
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


parser = argparse.ArgumentParser(description="MIND_Sim Arbor-HHPlus cosim benchmark.")
parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
parser.add_argument("--duration-ms", type=float, default=DEFAULT_DURATION_MS)
parser.add_argument("--batch-window-ms", type=float, default=DEFAULT_BATCH_WINDOW_MS)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--device",
    choices=("cpu", "gpu"),
    default=os.environ.get("MIND_SIM_COSIM_DEVICE", "cpu").strip() or "cpu",
)
args = parser.parse_args()
if args.cells <= 0:
    raise SystemExit("--cells must be positive")
if args.duration_ms <= 0.0:
    raise SystemExit("--duration-ms must be positive")

cells = args.cells
duration_ms = float(args.duration_ms)
batch_window_ms = float(args.batch_window_ms)
duration_label = f"{duration_ms / 1000.0:g}s".replace(".", "p")
output_file = args.output or RESULT_DIR / f"mind_sim_{cells}cells_{duration_label}_{args.device}.h5"
output_file.parent.mkdir(parents=True, exist_ok=True)
if round(MACRO_DT_MS / DT_MS) < 1 or not np.isclose(MACRO_DT_MS / DT_MS, round(MACRO_DT_MS / DT_MS), rtol=0.0, atol=1e-9):
    raise RuntimeError(f"MACRO_DT_MS must be an integer multiple of {DT_MS}")

with zipfile.ZipFile(CONNECTIVITY_FILE) as archive:
    labels = archive.read("region_labels.txt").decode("utf-8").split()
    weights = np.loadtxt(io.BytesIO(archive.read("weights.txt")), dtype=float)
    tract_lengths = np.loadtxt(io.BytesIO(archive.read("tract_lengths.txt")), dtype=float)

np.fill_diagonal(weights, 0.0)
delays = np.maximum(tract_lengths, MIN_TRACT_LENGTH_MS) / CONDUCTION_SPEED
min_positive_delay = float(np.min(delays[delays > 0.0]))
if round(batch_window_ms / MACRO_DT_MS) < 1 or not np.isclose(
    batch_window_ms / MACRO_DT_MS, round(batch_window_ms / MACRO_DT_MS), rtol=0.0, atol=1e-9
):
    raise RuntimeError(f"batch_window_ms must be an integer multiple of {MACRO_DT_MS}")
if batch_window_ms > min_positive_delay:
    raise RuntimeError("batch_window_ms must not exceed the minimum positive connectivity delay")

roi_count = len(labels)
exposure_names = ["S", "H"]
pre_start = time.perf_counter()
with suppress_native_output():
    # Build the micro circuit first. Its ion channel mechanism comes from mod/hhplus.mod.
    micro = ms.Sim()
    micro.set_device(args.device)
    micro.set_dt(DT_MS)
    micro.set_spike_output_enabled(True)
    micro.ion_register("cl", -1.0)
    micro.load_mech_metadata(str(EXAMPLE_DIR / "mod"))
    micro.cli0_cl_ion = 5.0
    micro.clo0_cl_ion = 112.0

    soma = ms.section("soma", "soma")
    volume_um3 = 2160.0
    radius_um = (0.75 * volume_um3 / math.pi) ** (1.0 / 3.0)
    area_um2 = 4.0 * math.pi * radius_um * radius_um
    length_um = math.sqrt(0.5 * area_um2 / math.pi)
    soma.L_um = length_um
    soma.diam_um = 2.0 * length_um
    soma.nseg = 1
    micro.build_morphology([{"name": "E", "num_cells": cells, "sections": [soma]}])

    population = micro.population("E")
    net = micro.network()
    somata = []
    healthy_cells = cells - int(PATHOLOGICAL_FRACTION * cells)
    for index in range(cells):
        group = population[index].group("soma")
        group.Ra = 100.0
        group.cm = 0.115
        kbath = K_BATH_OK if index < healthy_cells else K_BATH_BAD
        group.insert("hhplus", kbath=kbath)
        group.ecl = -26.64 * math.log(112.0 / 5.0)
        soma_mid = group[0](0.5)
        somata.append(soma_mid)
    spike_inputs = net.spike_inputs(cells)
    for index, soma_mid in enumerate(somata):
        synapse = soma_mid.insert("ExpSyn", tau=2.0, e=0.0)
        net.spike_connect(spike_inputs[index], synapse, MACRO_EVENT_WEIGHT, MACRO_EVENT_DELAY_MS)

    for index, soma_mid in enumerate(somata):
        net.register_gid_source(index, soma_mid._ref_v, -25.0)

    for post, soma_mid in enumerate(somata):
        synapse = soma_mid.insert("ExpSyn", tau=2.0, e=0.0)
        for pre in range(cells):
            if pre != post:
                net.gid_connect(pre, synapse, ARBOR_RECURRENT_WEIGHT, ARBOR_RECURRENT_DELAY_MS)

    micro.build_microcircuit()
    micro.finitialize(-78.0)

    # Load macro connectivity and MindMod rules, then replace one ROI by the micro circuit.
    network = ms.Network(
        labels=labels,
        weights=weights.tolist(),
        delays=delays.tolist(),
    )
    network.load_mod_metadata(EXAMPLE_DIR / "mind_mod")
    network.record(rois="all")

    micro_roi = network.roi(MICRO_ROI)
    all_rois = network.rois()
    macro_rois = [roi for roi in all_rois if roi.index != micro_roi.index]

    network.use_micro(micro).bind_roi(
        micro_roi,
        gid_ranges=population,
        ports={"afferent": spike_inputs},
    )

    # Macro ROI dynamics stays in RegionRule; all exchange between ROIs is in .mod rules.
    region = ms.RegionRule(
        name="rww",
        state={"S": 0.0},
        params={
            "a": 0.27,
            "b": 0.108,
            "d": 154.0,
            "gamma": 0.641,
            "tau_s": 100.0,
            "w": 1.0,
            "J_N": 0.2609,
            "I_o": 0.3,
        },
        step=r"""
current = w * J_N * S + I_o + J_N * coupled_S;
numerator = a * current - b;
denominator = 1.0 - exp(-d * numerator);
H = fabs(denominator) < 1e-12 ? (1.0 / d) : (numerator / denominator);
S += dt * (-(S / tau_s) + (1.0 - S) * H * gamma);
S = clamp(S, 0.0, 1.0);
""",
    )

    rng = np.random.default_rng(SEED)
    for roi in all_rois:
        initial_s = float(rng.uniform(0.0, 1.0))
        current = 1.0 * 0.2609 * initial_s + 0.3
        numerator = 0.27 * current - 0.108
        denominator = 1.0 - np.exp(-154.0 * numerator)
        initial_h = 1.0 / 154.0 if abs(denominator) < 1e-12 else float(numerator / denominator)
        if roi.index == MICRO_ROI:
            roi.initial_output({"S": 0.0, "H": 0.0})
            continue
        roi.initial_output({"S": initial_s, "H": initial_h})
        roi.use(region, state={"S": initial_s})

    for source_roi in macro_rois:
        micro_roi.connect(
            source_roi,
            "mean_h_input",
            params={"inv_source_count": 1.0 / float(len(macro_rois))},
        )
    micro_roi.connect(
        micro_roi,
        "h_to_netcon",
        params={
            "cells": float(cells),
            "scale": INPUT_SPIKE_SCALE,
            "max_rate_hz": MAX_INPUT_RATE_HZ,
        },
        random={
            "rng": {
                "state": {"seed": float(SEED + 10_000)},
                "uniform": r"""
    std::uint64_t rng = (std::uint64_t) seed;
    rng = rng * 2862933555777941757ULL + 3037000493ULL;
    seed = (double) (rng & 9007199254740991ULL);
    return seed / 9007199254740992.0;
""",
            }
        },
    )
    micro_roi.connect(
        micro_roi,
        "spikes_to_exposure",
        state={"ca": 0.0},
        params={
            "cells": float(cells),
            "tau_ms": EXPOSURE_TAU_MS,
            "exposure_gain": EXPOSURE_GAIN,
        },
    )

    for target_roi in macro_rois:
        for source_roi in all_rois:
            target_roi.connect(source_roi, "rww_s_coupling", params={"a": COUPLING_A})

pre_run_s = time.perf_counter() - pre_start

run_start = time.perf_counter()
with suppress_native_output():
    result = ms.Simulator(
        network,
        dt_micro=DT_MS,
        dt_macro=MACRO_DT_MS,
        batch_window=batch_window_ms,
    ).run(duration_ms)
run_s = time.perf_counter() - run_start

metadata = [
    float(cells),
    float(duration_ms),
    float(DT_MS),
    float(MACRO_DT_MS),
    float(batch_window_ms),
    float(roi_count),
    float(MICRO_ROI),
]
timing = [pre_run_s, run_s, pre_run_s + run_s]
result.save_h5(
    str(output_file),
    exposure_names,
    labels,
    MICRO_ROI,
    timing,
    metadata,
)

spikes = result.micro_spikes_by_roi[MICRO_ROI]
print(f"output={output_file}")
print(f"device={args.device}")
print(f"pre_run_s={pre_run_s:.6f}")
print(f"run_s={run_s:.6f}")
print(f"total_s={pre_run_s + run_s:.6f}")
print(f"spikes={len(spikes)}")
