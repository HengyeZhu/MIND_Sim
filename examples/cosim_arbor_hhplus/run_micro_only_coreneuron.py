#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result" / "cosim_arbor_hhplus"
DEFAULT_MOD_DIR = HERE / "mod_gpu"

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

_CORENEURON_LIBRARY_HANDLE = None


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


def preload_coreneuron_library(path: Path) -> None:
    global _CORENEURON_LIBRARY_HANDLE
    mode = os.RTLD_GLOBAL | os.RTLD_NOW
    if hasattr(os, "RTLD_NODELETE"):
        mode |= os.RTLD_NODELETE
    _CORENEURON_LIBRARY_HANDLE = ctypes.CDLL(str(path), mode=mode)


def vector_to_numpy(vec, dtype=float) -> np.ndarray:
    return np.asarray([vec.x[i] for i in range(int(vec.size()))], dtype=dtype)


parser = argparse.ArgumentParser(description="CoreNEURON GPU micro-only HHPlus network from cosim_arbor_hhplus.")
parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
parser.add_argument("--duration-ms", type=float, default=DEFAULT_DURATION_MS)
parser.add_argument("--record-gids", default="0,1,2,3,4,5,6,7,8,9")
parser.add_argument("--mod-dir", type=Path, default=DEFAULT_MOD_DIR)
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
mod_dir = args.mod_dir.resolve()
coreneuron_library = mod_dir / "x86_64" / "libcorenrnmech.so"
if not coreneuron_library.is_file():
    raise SystemExit(f"missing CoreNEURON mechanism library: {coreneuron_library}")

output = args.output or RESULT_DIR / f"micro_only_coreneuron_{cells}cells_{duration_label(duration_ms)}_gpu.npz"
output.parent.mkdir(parents=True, exist_ok=True)

os.environ["CORENEURONLIB"] = str(coreneuron_library)
preload_coreneuron_library(coreneuron_library)

from neuron import coreneuron, h, load_mechanisms  # noqa: E402

pre_start = time.perf_counter()
with suppress_native_output(args.quiet):
    h.load_file("stdrun.hoc")
    load_mechanisms(str(mod_dir))
    pc = h.ParallelContext()
    pc.set_maxstep(10.0)

    h.dt = DT_MS
    h.tstop = duration_ms
    h.cli0_cl_ion = 5.0
    h.clo0_cl_ion = 112.0

    length_um, diam_um = arbor_point_cell_dimensions()
    healthy_cells = cells - int(PATHOLOGICAL_FRACTION * cells)
    ecl = -26.64 * math.log(112.0 / 5.0)

    sections = []
    gid_sources = []
    macro_input_synapses = []
    macro_input_netcons = []
    recurrent_synapses = []
    recurrent_netcons = []
    for gid in range(cells):
        section = h.Section(name=f"soma_{gid}")
        section.L = length_um
        section.diam = diam_um
        section.nseg = 1
        section.Ra = 100.0
        section.cm = 0.115
        section.insert("hhplus")
        for segment in section:
            segment.hhplus.kbath = K_BATH_OK if gid < healthy_cells else K_BATH_BAD
            segment.ecl = ecl
        source = h.NetCon(section(0.5)._ref_v, None, sec=section)
        source.threshold = -25.0
        pc.set_gid2node(gid, int(pc.id()))
        pc.cell(gid, source)
        sections.append(section)
        gid_sources.append(source)

    for gid, section in enumerate(sections):
        macro_synapse = h.ExpSyn(0.5, sec=section)
        macro_synapse.tau = 2.0
        macro_synapse.e = 0.0
        macro_input_synapses.append(macro_synapse)
        netcon = h.NetCon(None, macro_synapse)
        netcon.weight[0] = MACRO_EVENT_WEIGHT
        netcon.delay = MACRO_EVENT_DELAY_MS
        macro_input_netcons.append(netcon)

    for post_gid, section in enumerate(sections):
        synapse = h.ExpSyn(0.5, sec=section)
        synapse.tau = 2.0
        synapse.e = 0.0
        recurrent_synapses.append(synapse)
        for pre_gid in range(cells):
            if pre_gid == post_gid:
                continue
            netcon = pc.gid_connect(pre_gid, synapse)
            netcon.weight[0] = RECURRENT_WEIGHT
            netcon.delay = RECURRENT_DELAY_MS
            recurrent_netcons.append(netcon)

    spike_times_vector = h.Vector()
    spike_gids_vector = h.Vector()
    pc.spike_record(-1, spike_times_vector, spike_gids_vector)
    voltage_vectors = [h.Vector().record(sections[gid](0.5)._ref_v) for gid in record_gids]
    time_vector = h.Vector().record(h._ref_t)

    coreneuron.enable = True
    coreneuron.gpu = True
    coreneuron.num_gpus = 1
    coreneuron.verbose = 0
    coreneuron.cell_permute = 2
    h.finitialize(V_INIT)
pre_run_s = time.perf_counter() - pre_start

run_start = time.perf_counter()
with suppress_native_output(args.quiet):
    pc.psolve(duration_ms)
run_s = time.perf_counter() - run_start

spike_times = vector_to_numpy(spike_times_vector, np.float64)
spike_gids = vector_to_numpy(spike_gids_vector, np.int32)
order = np.lexsort((spike_gids, spike_times))
spike_times = spike_times[order]
spike_gids = spike_gids[order]
times_ms = vector_to_numpy(time_vector, np.float64)
voltages_mv = np.vstack([vector_to_numpy(vec, np.float64) for vec in voltage_vectors])

np.savez_compressed(
    output,
    backend=np.asarray("coreneuron", dtype="S"),
    device=np.asarray("gpu", dtype="S"),
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
print("backend=coreneuron")
print("device=gpu")
print(f"cells={cells}")
print(f"duration_ms={duration_ms:g}")
print(f"pre_run_s={pre_run_s:.6f}")
print(f"run_s={run_s:.6f}")
print(f"total_s={pre_run_s + run_s:.6f}")
print(f"spikes={len(spike_times)}")
print(f"record_gids={','.join(str(gid) for gid in record_gids)}")
