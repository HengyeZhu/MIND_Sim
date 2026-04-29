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
CONNECTIVITY_FILE = Path.home() / "arbor-tvb-cosim" / "connectivity_mouse.zip"
RESULT_DIR = ROOT / "result" / "cosim_arbor_hhplus"

DEFAULT_CELLS = 100
DURATION_MS = 2000.0
DT_MS = 0.01
MACRO_DT_MS = 0.01
BATCH_WINDOW_MS = 0.25
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


def rww_initial_rate(s_value: float) -> float:
    a = 0.27
    b = 0.108
    d = 154.0
    j_n = 0.2609
    i_o = 0.3
    w = 1.0
    current = w * j_n * s_value + i_o
    numerator = a * current - b
    denominator = 1.0 - np.exp(-d * numerator)
    if abs(denominator) < 1e-12:
        return 1.0 / d
    return float(numerator / denominator)


def arbor_point_cell_dimensions() -> tuple[float, float]:
    volume_um3 = 2160.0
    radius_um = (0.75 * volume_um3 / math.pi) ** (1.0 / 3.0)
    area_um2 = 4.0 * math.pi * radius_um * radius_um
    length_um = math.sqrt(0.5 * area_um2 / math.pi)
    return length_um, 2.0 * length_um


def require_integer_multiple(value: float, unit: float, label: str) -> None:
    ratio = value / unit
    if round(ratio) < 1 or not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{label} must be an integer multiple of {unit}")


parser = argparse.ArgumentParser(description="MIND_Sim Arbor-HHPlus cosim benchmark.")
parser.add_argument("--cells", type=int, default=DEFAULT_CELLS, choices=(100, 1000))
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()

CELLS = args.cells
OUTPUT_FILE = args.output or RESULT_DIR / f"mind_sim_{CELLS}cells_2s.h5"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
require_integer_multiple(MACRO_DT_MS, DT_MS, "MACRO_DT_MS")
require_integer_multiple(BATCH_WINDOW_MS, MACRO_DT_MS, "BATCH_WINDOW_MS")

with zipfile.ZipFile(CONNECTIVITY_FILE) as archive:
    labels = archive.read("region_labels.txt").decode("utf-8").split()
    weights = np.loadtxt(io.BytesIO(archive.read("weights.txt")), dtype=float)
    tract_lengths = np.loadtxt(io.BytesIO(archive.read("tract_lengths.txt")), dtype=float)

np.fill_diagonal(weights, 0.0)
delays = np.maximum(tract_lengths, MIN_TRACT_LENGTH_MS) / CONDUCTION_SPEED
if BATCH_WINDOW_MS > float(np.min(delays[delays > 0.0])):
    raise RuntimeError("BATCH_WINDOW_MS must not exceed the minimum positive connectivity delay")

roi_count = len(labels)
input_names = ["coupled_S", "mean_H"]
exposure_names = ["S", "H"]
rng = np.random.default_rng(SEED)
initial_s = rng.uniform(0.0, 1.0, size=roi_count).astype(float)
initial_h = np.asarray([rww_initial_rate(float(value)) for value in initial_s], dtype=float)
initial_s[MICRO_ROI] = 0.0
initial_h[MICRO_ROI] = 0.0

rww_update = r"""
double current = p.w * p.J_N * s.S + p.I_o + p.J_N * in.coupled_S;
double numerator = p.a * current - p.b;
double denominator = 1.0 - exp(-p.d * numerator);
double H_value = fabs(denominator) < 1e-12 ? (1.0 / p.d) : (numerator / denominator);
s.S += dt * (-(s.S / p.tau_s) + (1.0 - s.S) * H_value * p.gamma);
s.S = clamp(s.S, 0.0, 1.0);
out.S = s.S;
out.H = H_value;
"""
rww_coupling_edge = r"""
in.coupled_S += p.a * static_cast<double>(static_cast<float>(edge.weight)) * static_cast<double>(static_cast<float>(src.S));
"""
h_input_coupling_edge = r"""
in.mean_H += p.inv_source_count * src.H;
"""
input_emit = r"""
const int cell_count = static_cast<int>(p.cells);
double event_rate = in.mean_H * p.scale;
event_rate = clamp(event_rate, 0.0, p.max_rate_hz);
const double probability = clamp(event_rate * window.duration / 1000.0, 0.0, 1.0);
for (int cell = 0; cell < cell_count; ++cell) {
    uint64_t rng = static_cast<uint64_t>(s.seed);
    rng = rng * 2862933555777941757ULL + 3037000493ULL;
    s.seed = static_cast<double>(rng & 9007199254740991ULL);
    const double u = s.seed / 9007199254740992.0;
    rng = rng * 2862933555777941757ULL + 3037000493ULL;
    s.seed = static_cast<double>(rng & 9007199254740991ULL);
    const double offset = (s.seed / 9007199254740992.0) * window.duration;
    if (u < probability) {
        emit.afferent(window.start + offset, cell);
    }
}
"""
exposure_spike = r"""
if (spike.t >= window.start && spike.t < window.stop) {
    s.ca += 1.0 / p.cells;
}
"""
exposure_finish = r"""
s.ca *= exp(-window.duration / p.tau_ms);
const double activity = s.ca * p.exposure_gain;
out.S = activity;
out.H = activity;
"""

pre_start = time.perf_counter()
with suppress_native_output():
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
        step=rww_update,
    )
    rww_coupling = ms.CouplingRule(
        name="rww_s",
        params={"a": COUPLING_A},
        edge=rww_coupling_edge,
        finish="",
    )
    h_input_coupling = ms.CouplingRule(
        name="mean_h_input",
        params={"inv_source_count": 1.0},
        edge=h_input_coupling_edge,
        finish="",
    )
    input_rule = ms.MicroInputRule(
        name="h_to_netcon",
        ports=["afferent"],
        state={"seed": float(SEED + 10_000)},
        params={
            "cells": float(CELLS),
            "scale": INPUT_SPIKE_SCALE,
            "max_rate_hz": MAX_INPUT_RATE_HZ,
        },
        code=input_emit,
    )
    exposure_rule = ms.MicroOutputRule(
        name="spikes_to_exposure",
        state={"ca": 0.0},
        params={
            "cells": float(CELLS),
            "tau_ms": EXPOSURE_TAU_MS,
            "exposure_gain": EXPOSURE_GAIN,
        },
        spike=exposure_spike,
        finish=exposure_finish,
    )

    network = ms.Network(
        labels=labels,
        weights=weights.tolist(),
        delays=delays.tolist(),
        inputs=input_names,
        exposures=exposure_names,
    )
    network.record(rois="all", exposures="all")
    network.couple_all(rww_coupling)

    micro = ms.Sim()
    micro.set_dt(DT_MS)
    micro.set_spike_output_enabled(True)
    micro.ion_register("cl", -1.0)
    micro.load_mech_metadata(str(Path(__file__).resolve().parent / "mod"))
    micro.cli0_cl_ion = 5.0
    micro.clo0_cl_ion = 112.0

    soma = ms.section("soma", "soma")
    soma.L_um, soma.diam_um = arbor_point_cell_dimensions()
    soma.nseg = 1
    micro.build_morphology([{"name": "E", "num_cells": CELLS, "sections": [soma]}])

    population = micro.population("E")
    net = micro.network()
    somata = []
    healthy_cells = CELLS - int(PATHOLOGICAL_FRACTION * CELLS)
    for index in range(CELLS):
        group = population[index].group("soma")
        group.Ra = 100.0
        group.cm = 0.115
        kbath = K_BATH_OK if index < healthy_cells else K_BATH_BAD
        group.insert("hhplus", kbath=kbath)
        group.ecl = -26.64 * math.log(112.0 / 5.0)
        soma_mid = group[0](0.5)
        somata.append(soma_mid)
    spike_inputs = net.spike_inputs(CELLS)
    for index, soma_mid in enumerate(somata):
        synapse = soma_mid.insert("ExpSyn", tau=2.0, e=0.0)
        net.spike_connect(spike_inputs[index], synapse, MACRO_EVENT_WEIGHT, MACRO_EVENT_DELAY_MS)

    for index, soma_mid in enumerate(somata):
        net.register_gid_source(index, soma_mid._ref_v, -25.0)

    for post, soma_mid in enumerate(somata):
        synapse = soma_mid.insert("ExpSyn", tau=2.0, e=0.0)
        for pre in range(CELLS):
            if pre != post:
                net.gid_connect(pre, synapse, ARBOR_RECURRENT_WEIGHT, ARBOR_RECURRENT_DELAY_MS)

    micro.build_microcircuit()
    micro.finitialize(-78.0)

    micro_roi = network.roi(MICRO_ROI)
    micro_roi.initial_output({"S": float(initial_s[MICRO_ROI]), "H": float(initial_h[MICRO_ROI])})
    network.use_micro("micro", micro).bind_roi(
        micro_roi,
        gid_ranges=population,
        input=input_rule,
        input_ports={"afferent": spike_inputs},
        output=exposure_rule,
    )

    macro_roi_indices = [roi_index for roi_index in range(roi_count) if roi_index != MICRO_ROI]
    network.couple(
        sources=macro_roi_indices,
        targets=[MICRO_ROI],
        rule=h_input_coupling,
        params={"inv_source_count": 1.0 / float(len(macro_roi_indices))},
        delays=False,
    )

    for roi_index in range(roi_count):
        if roi_index == MICRO_ROI:
            continue
        roi = network.roi(roi_index)
        roi.initial_output({"S": float(initial_s[roi_index]), "H": float(initial_h[roi_index])})
        roi.use(region, state={"S": float(initial_s[roi_index])})

pre_run_s = time.perf_counter() - pre_start

run_start = time.perf_counter()
with suppress_native_output():
    result = ms.Simulator(
        network,
        dt_micro=DT_MS,
        dt_macro=MACRO_DT_MS,
        batch_window=BATCH_WINDOW_MS,
    ).run(DURATION_MS)
run_s = time.perf_counter() - run_start

metadata = [
    float(CELLS),
    float(DURATION_MS),
    float(DT_MS),
    float(MACRO_DT_MS),
    float(BATCH_WINDOW_MS),
    float(roi_count),
    float(MICRO_ROI),
]
timing = [pre_run_s, run_s, pre_run_s + run_s]
result.save_h5(
    str(OUTPUT_FILE),
    exposure_names,
    labels,
    MICRO_ROI,
    timing,
    metadata,
)

spikes = result.micro_spikes_by_roi[MICRO_ROI]
print(f"output={OUTPUT_FILE}")
print(f"pre_run_s={pre_run_s:.6f}")
print(f"run_s={run_s:.6f}")
print(f"total_s={pre_run_s + run_s:.6f}")
print(f"spikes={len(spikes)}")
