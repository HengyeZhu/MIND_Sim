#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import mind_sim as ms


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result" / "cosim_arbor_hhplus"

DEFAULT_CELLS = 100
DEFAULT_DURATION_MS = 2000.0
DT_MS = 0.01
V_INIT = -78.0
PATHOLOGICAL_FRACTION = 1.0
K_BATH_OK = 9.5
K_BATH_BAD = 17.0
RECURRENT_WEIGHT = 0.5
RECURRENT_DELAY_MS = 0.5
MACRO_EVENT_WEIGHT = 0.01
MACRO_EVENT_DELAY_MS = 1.0


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
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def arbor_point_cell_dimensions() -> tuple[float, float]:
    volume_um3 = 2160.0
    radius_um = (0.75 * volume_um3 / math.pi) ** (1.0 / 3.0)
    area_um2 = 4.0 * math.pi * radius_um * radius_um
    length_um = math.sqrt(0.5 * area_um2 / math.pi)
    return length_um, 2.0 * length_um


def parse_record_gids(raw: str, cells: int) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        gid = int(item)
        if gid < 0 or gid >= cells:
            raise SystemExit(f"record gid out of range: {gid}")
        if gid not in values:
            values.append(gid)
    return values


def duration_label(duration_ms: float) -> str:
    if abs(duration_ms - round(duration_ms)) < 1e-9:
        return f"{int(round(duration_ms))}ms"
    return f"{duration_ms:g}ms".replace(".", "p")


def collect_spikes(sim: ms.Sim, cells: int) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    gids: list[int] = []
    for gid in range(cells):
        gid_spikes = sim.get_spk_by_gid(gid)
        times.extend(float(t) for t in gid_spikes)
        gids.extend([gid] * len(gid_spikes))
    spike_times = np.asarray(times, dtype=np.float64)
    spike_gids = np.asarray(gids, dtype=np.int32)
    order = np.lexsort((spike_gids, spike_times))
    return spike_times[order], spike_gids[order]


parser = argparse.ArgumentParser(description="MIND_Sim micro-only HHPlus network from cosim_arbor_hhplus.")
parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
parser.add_argument("--duration-ms", type=float, default=DEFAULT_DURATION_MS)
parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MIND_SIM_MICRO_NUM_THREADS", "1") or "1"))
parser.add_argument("--record-gids", default="0,1,2,3,4,5,6,7,8,9")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

if args.cells <= 0:
    raise SystemExit("--cells must be positive")
if args.duration_ms <= 0.0:
    raise SystemExit("--duration-ms must be positive")

cells = int(args.cells)
duration_ms = float(args.duration_ms)
record_gids = parse_record_gids(args.record_gids, cells)
output = args.output or RESULT_DIR / (
    f"micro_only_mind_sim_{cells}cells_{duration_label(duration_ms)}_{args.device}.npz"
)
output.parent.mkdir(parents=True, exist_ok=True)

pre_start = time.perf_counter()
with suppress_native_output(args.quiet):
    sim = ms.Sim()
    sim.set_device(args.device)
    sim.set_num_threads(args.num_threads)
    sim.set_dt(DT_MS)
    sim.set_spike_output_enabled(True)
    sim.ion_register("cl", -1.0)
    sim.load_mech_metadata(str(HERE / "mod"))
    sim.cli0_cl_ion = 5.0
    sim.clo0_cl_ion = 112.0

    soma = ms.section("soma", "soma")
    soma.L_um, soma.diam_um = arbor_point_cell_dimensions()
    soma.nseg = 1
    sim.build_morphology([{"name": "E", "num_cells": cells, "sections": [soma]}])

    population = sim.population("E")
    net = sim.network()
    somata = []
    healthy_cells = cells - int(PATHOLOGICAL_FRACTION * cells)
    ecl = -26.64 * math.log(112.0 / 5.0)
    for gid in range(cells):
        group = population[gid].group("soma")
        group.Ra = 100.0
        group.cm = 0.115
        kbath = K_BATH_OK if gid < healthy_cells else K_BATH_BAD
        group.insert("hhplus", kbath=kbath)
        group.ecl = ecl
        somata.append(group[0](0.5))

    spike_inputs = net.spike_inputs(cells)
    for gid, midpoint in enumerate(somata):
        synapse = midpoint.insert("ExpSyn", tau=2.0, e=0.0)
        net.spike_connect(spike_inputs[gid], synapse, MACRO_EVENT_WEIGHT, MACRO_EVENT_DELAY_MS)

    for gid, midpoint in enumerate(somata):
        net.register_gid_source(gid, midpoint._ref_v, -25.0)

    for post, midpoint in enumerate(somata):
        synapse = midpoint.insert("ExpSyn", tau=2.0, e=0.0)
        for pre in range(cells):
            if pre != post:
                net.gid_connect(pre, synapse, RECURRENT_WEIGHT, RECURRENT_DELAY_MS)

    voltage_vectors = [ms.Vector().record(somata[gid]._ref_v) for gid in record_gids]
    time_vector = ms.Vector().record(sim._ref_t)

    sim.build_microcircuit()
    sim.finitialize(V_INIT)
pre_run_s = time.perf_counter() - pre_start

run_start = time.perf_counter()
with suppress_native_output(args.quiet):
    sim.run(duration_ms)
run_s = time.perf_counter() - run_start

spike_times, spike_gids = collect_spikes(sim, cells)
times_ms = np.asarray(time_vector.to_python(), dtype=np.float64)
voltages_mv = np.vstack([np.asarray(vec.to_python(), dtype=np.float64) for vec in voltage_vectors])

np.savez_compressed(
    output,
    backend=np.asarray("mind_sim", dtype="S"),
    device=np.asarray(args.device, dtype="S"),
    num_threads=np.asarray(args.num_threads, dtype=np.int32),
    cells=np.asarray(cells, dtype=np.int32),
    duration_ms=np.asarray(duration_ms, dtype=np.float64),
    dt_ms=np.asarray(DT_MS, dtype=np.float64),
    record_gids=np.asarray(record_gids, dtype=np.int32),
    times_ms=times_ms,
    voltages_mv=voltages_mv,
    spike_times_ms=spike_times,
    spike_gids=spike_gids,
    timing_s=np.asarray([pre_run_s, run_s, pre_run_s + run_s], dtype=np.float64),
)

print(f"output={output}")
print("backend=mind_sim")
print(f"device={args.device}")
print(f"num_threads={args.num_threads}")
print(f"cells={cells}")
print(f"duration_ms={duration_ms:g}")
print(f"pre_run_s={pre_run_s:.6f}")
print(f"run_s={run_s:.6f}")
print(f"total_s={pre_run_s + run_s:.6f}")
print(f"spikes={len(spike_times)}")
print(f"record_gids={','.join(str(gid) for gid in record_gids)}")
