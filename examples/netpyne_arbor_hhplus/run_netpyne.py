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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MOD_DIR = HERE / "mod"
CONNECTIVITY_FILE = Path.home() / "arbor-tvb-cosim" / "connectivity_mouse.zip"
RESULT_DIR = ROOT / "result" / "netpyne_arbor_hhplus"

DEFAULT_CELLS = 100
DURATION_MS = 2000.0
DT_MS = 0.01
MACRO_DT_MS = 0.01
EXCHANGE_WINDOW_MS = 0.25
MICRO_ROI = 72
SEED = 1234
V_INIT = -78.0
CORENEURON_CPU = True

PATHOLOGICAL_FRACTION = 1.0
K_BATH_OK = 9.5
K_BATH_BAD = 17.0
RECURRENT_WEIGHT = 0.5
RECURRENT_DELAY_MS = 0.5

MACRO_EVENT_WEIGHT = 0.01
MACRO_EVENT_DELAY_MS = 1.0
INPUT_SPIKE_SCALE = 5.0
MAX_INPUT_RATE_HZ = 150.0
COUPLING_A = 0.096
EXPOSURE_GAIN = 10.0
EXPOSURE_TAU_MS = 100.0
MIN_TRACT_LENGTH_MS = 1.0
CONDUCTION_SPEED = 3.0

EXPOSURE_NAMES = ["S", "H"]
S_INDEX = 0
H_INDEX = 1


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


def require_integer_multiple(value: float, unit: float, label: str) -> int:
    ratio = value / unit
    rounded = round(ratio)
    if rounded < 1 or not np.isclose(ratio, rounded, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{label} must be an integer multiple of {unit}")
    return int(rounded)


def arbor_point_cell_dimensions() -> tuple[float, float]:
    volume_um3 = 2160.0
    radius_um = (0.75 * volume_um3 / math.pi) ** (1.0 / 3.0)
    area_um2 = 4.0 * math.pi * radius_um * radius_um
    length_um = math.sqrt(0.5 * area_um2 / math.pi)
    return length_um, 2.0 * length_um


def rww_initial_rate(s_value: float) -> float:
    a = 0.27
    b = 0.108
    d = 154.0
    j_n = 0.2609
    i_o = 0.3
    current = j_n * s_value + i_o
    numerator = a * current - b
    denominator = 1.0 - np.exp(-d * numerator)
    if abs(denominator) < 1e-12:
        return 1.0 / d
    return float(numerator / denominator)


def load_connectivity() -> tuple[list[str], np.ndarray, np.ndarray]:
    with zipfile.ZipFile(CONNECTIVITY_FILE) as archive:
        labels = archive.read("region_labels.txt").decode("utf-8").split()
        weights = np.loadtxt(io.BytesIO(archive.read("weights.txt")), dtype=np.float32)
        tract_lengths = np.loadtxt(io.BytesIO(archive.read("tract_lengths.txt")), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    delays = np.maximum(tract_lengths, MIN_TRACT_LENGTH_MS) / CONDUCTION_SPEED
    return labels, weights, delays


def make_netpyne_params(cells: int, duration_ms: float):
    from netpyne import specs

    length_um, diam_um = arbor_point_cell_dimensions()
    ecl = -26.64 * math.log(112.0 / 5.0)
    healthy_cells = cells - int(PATHOLOGICAL_FRACTION * cells)

    net_params = specs.NetParams()
    net_params.defaultThreshold = -25.0
    net_params.scale = 1.0
    net_params.sizeX = 1.0
    net_params.sizeY = 1.0
    net_params.sizeZ = 1.0

    def cell_rule(kbath: float) -> dict:
        return {
            "conds": {"cellType": f"hhplus_{kbath:g}", "cellModel": "hhplus_point"},
            "secs": {
                "soma": {
                    "geom": {
                        "L": length_um,
                        "diam": diam_um,
                        "nseg": 1,
                        "Ra": 100.0,
                        "cm": 0.115,
                    },
                    "mechs": {"hhplus": {"kbath": kbath}},
                    "ions": {"cl": {"e": ecl, "i": 5.0, "o": 112.0}},
                    "threshold": -25.0,
                }
            },
        }

    if healthy_cells:
        net_params.cellParams["healthy"] = cell_rule(K_BATH_OK)
        net_params.popParams["healthy"] = {
            "cellType": f"hhplus_{K_BATH_OK:g}",
            "cellModel": "hhplus_point",
            "numCells": healthy_cells,
        }
    pathological_cells = cells - healthy_cells
    if pathological_cells:
        net_params.cellParams["pathological"] = cell_rule(K_BATH_BAD)
        net_params.popParams["pathological"] = {
            "cellType": f"hhplus_{K_BATH_BAD:g}",
            "cellModel": "hhplus_point",
            "numCells": pathological_cells,
        }

    sim_config = specs.SimConfig()
    sim_config.duration = duration_ms
    sim_config.dt = DT_MS
    sim_config.hParams["v_init"] = V_INIT
    sim_config.createNEURONObj = True
    sim_config.createPyStruct = True
    sim_config.addSynMechs = False
    sim_config.oneSynPerNetcon = False
    sim_config.recordCells = []
    sim_config.recordTraces = {}
    sim_config.recordCellsSpikes = -1
    sim_config.recordTime = False
    sim_config.recordStim = False
    sim_config.saveCellSecs = False
    sim_config.saveCellConns = False
    sim_config.gatherOnlySimData = True
    sim_config.timing = False
    sim_config.verbose = False
    sim_config.progressBar = 0
    sim_config.printRunTime = False
    sim_config.printPopAvgRates = False
    return net_params, sim_config


def make_initial_exposures(roi_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    initial_s = rng.uniform(0.0, 1.0, size=roi_count).astype(np.float64)
    initial_h = np.asarray([rww_initial_rate(float(value)) for value in initial_s], dtype=np.float64)
    initial_s[MICRO_ROI] = 0.0
    initial_h[MICRO_ROI] = 0.0
    exposures = np.zeros((roi_count, len(EXPOSURE_NAMES)), dtype=np.float32)
    exposures[:, S_INDEX] = initial_s.astype(np.float32)
    exposures[:, H_INDEX] = initial_h.astype(np.float32)
    return exposures, initial_s, initial_h


def fill_region_coupling(
    inputs: np.ndarray,
    target_rois: np.ndarray,
    target_weights: np.ndarray,
    target_delay_offsets: np.ndarray,
    source_indices: np.ndarray,
    history: np.ndarray,
    step: int,
    history_capacity: int,
) -> None:
    current_slot = step % history_capacity
    slots = current_slot + target_delay_offsets
    slots = np.where(slots >= history_capacity, slots - history_capacity, slots)
    delayed_s = history[slots, source_indices, S_INDEX]
    inputs.fill(0.0)
    inputs[target_rois, S_INDEX] = np.sum(target_weights * delayed_s, axis=1, dtype=np.float32) * np.float32(COUPLING_A)


def apply_micro_coupling(
    macro_rois: np.ndarray,
    micro_roi: int,
    history: np.ndarray,
    step: int,
    history_capacity: int,
) -> np.ndarray:
    current_slot = step % history_capacity
    input_values = np.zeros(len(EXPOSURE_NAMES), dtype=np.float32)
    input_values[H_INDEX] = np.float32(np.sum(history[current_slot, macro_rois, H_INDEX], dtype=np.float32) / len(macro_rois))
    return input_values


def step_regions(
    macro_rois: np.ndarray,
    states_s: np.ndarray,
    inputs: np.ndarray,
    exposures: np.ndarray,
) -> None:
    s_values = states_s[macro_rois]
    currents = 0.2609 * s_values + 0.3 + 0.2609 * inputs[macro_rois, S_INDEX].astype(np.float64)
    numerators = 0.27 * currents - 0.108
    denominators = 1.0 - np.exp(-154.0 * numerators)
    h_values = np.where(np.abs(denominators) < 1e-12, 1.0 / 154.0, numerators / denominators)
    s_values = s_values + MACRO_DT_MS * (-(s_values / 100.0) + (1.0 - s_values) * h_values * 0.641)
    s_values = np.clip(s_values, 0.0, 1.0)
    states_s[macro_rois] = s_values
    exposures[macro_rois, S_INDEX] = s_values.astype(np.float32)
    exposures[macro_rois, H_INDEX] = h_values.astype(np.float32)


def lcg_uniform_pair(seed_state: float) -> tuple[float, float, float]:
    rng = int(seed_state) & ((1 << 64) - 1)
    rng = (rng * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    seed_state = float(rng & 9007199254740991)
    u = seed_state / 9007199254740992.0
    rng = (rng * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    seed_state = float(rng & 9007199254740991)
    offset_fraction = seed_state / 9007199254740992.0
    return seed_state, u, offset_fraction


def save_h5(
    path: Path,
    times_ms: np.ndarray,
    roi_exposures: np.ndarray,
    labels: list[str],
    spike_times_ms: np.ndarray,
    spike_gids: np.ndarray,
    timing_s: list[float],
    metadata: list[float],
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times_ms", data=times_ms[1:].astype(np.float64))
        h5.create_dataset("exposure_names", data=np.asarray(EXPOSURE_NAMES, dtype="S"))
        h5.create_dataset("roi_labels", data=np.asarray(labels, dtype="S"))
        h5.create_dataset("recorded_rois", data=np.arange(len(labels), dtype=np.int32))
        h5.create_dataset("roi_exposures", data=roi_exposures[1:].astype(np.float32))
        h5.create_dataset("spike_times_ms", data=spike_times_ms.astype(np.float64))
        h5.create_dataset("spike_gids", data=spike_gids.astype(np.int32))
        h5.create_dataset("timing_s", data=np.asarray(timing_s, dtype=np.float64))
        h5.create_dataset("metadata", data=np.asarray(metadata, dtype=np.float64))
        h5.attrs["micro_backend"] = "coreneuron_cpu" if CORENEURON_CPU else "neuron"
        h5.attrs["roi_count"] = len(labels)
        h5.attrs["exposure_count"] = len(EXPOSURE_NAMES)


parser = argparse.ArgumentParser(description="Fixed NetPyNE baseline for the MIND_Sim Arbor-HHPlus cosim benchmark.")
parser.add_argument("--cells", type=int, default=DEFAULT_CELLS, choices=(100, 1000))
parser.add_argument("--duration-ms", type=float, default=DURATION_MS)
parser.add_argument("--check-config", action="store_true")
parser.add_argument("--build-only", action="store_true")
parser.add_argument("--skip-core-transfer-after-first", action="store_true")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()
duration_ms = args.duration_ms

require_integer_multiple(MACRO_DT_MS, DT_MS, "MACRO_DT_MS")
exchange_step_count = require_integer_multiple(EXCHANGE_WINDOW_MS, MACRO_DT_MS, "EXCHANGE_WINDOW_MS")

labels, weights, delays = load_connectivity()
positive_delays = delays[delays > 0.0]
if EXCHANGE_WINDOW_MS > float(np.min(positive_delays)):
    raise RuntimeError("EXCHANGE_WINDOW_MS must not exceed the minimum positive connectivity delay")

roi_count = len(labels)
macro_rois = np.asarray([roi for roi in range(roi_count) if roi != MICRO_ROI], dtype=np.int32)
output_suffix = "2s" if duration_ms == DURATION_MS else f"{duration_ms:g}ms"
output_file = args.output or RESULT_DIR / f"netpyne_{args.cells}cells_{output_suffix}.h5"

net_params, sim_config = make_netpyne_params(args.cells, duration_ms)

if args.check_config:
    print(f"cells={args.cells}")
    print(f"duration_ms={duration_ms}")
    print(f"dt_ms={DT_MS}")
    print(f"macro_dt_ms={MACRO_DT_MS}")
    print(f"exchange_window_ms={EXCHANGE_WINDOW_MS}")
    print(f"micro_backend={'coreneuron_cpu' if CORENEURON_CPU else 'neuron'}")
    print(f"roi_count={roi_count}")
    print(f"micro_roi={MICRO_ROI}")
    print(f"output={output_file}")
    print(f"manual_recurrent_gid_connects={args.cells * (args.cells - 1)}")
    print(f"manual_macro_input_netcons={args.cells}")
    print(f"mod_dir={MOD_DIR}")
    raise SystemExit(0)

pre_start = time.perf_counter()
with suppress_native_output(args.quiet):
    if CORENEURON_CPU:
        os.environ["CORENEURONLIB"] = str(MOD_DIR / "x86_64" / "libcorenrnmech.so")
    from neuron import coreneuron, h, load_mechanisms
    from netpyne import sim

    load_mechanisms(str(MOD_DIR))
    sim.create(net_params, sim_config, clearAll=True)

    h.cli0_cl_ion = 5.0
    h.clo0_cl_ion = 112.0

    cells_by_gid = {int(cell.gid): cell for cell in sim.net.cells}
    input_netcons = []
    recurrent_netcons = []
    macro_input_synapses = []
    recurrent_synapses = []
    for gid in range(args.cells):
        soma = cells_by_gid[gid].secs["soma"]["hObj"]
        macro_synapse = h.ExpSyn(0.5, sec=soma)
        macro_synapse.tau = 2.0
        macro_synapse.e = 0.0
        macro_input_synapses.append(macro_synapse)
        netcon = h.NetCon(None, macro_synapse)
        netcon.weight[0] = MACRO_EVENT_WEIGHT
        netcon.delay = MACRO_EVENT_DELAY_MS
        input_netcons.append(netcon)

    for post_gid in range(args.cells):
        soma = cells_by_gid[post_gid].secs["soma"]["hObj"]
        synapse = h.ExpSyn(0.5, sec=soma)
        synapse.tau = 2.0
        synapse.e = 0.0
        recurrent_synapses.append(synapse)
        for pre_gid in range(args.cells):
            if pre_gid == post_gid:
                continue
            netcon = sim.pc.gid_connect(pre_gid, synapse)
            netcon.weight[0] = RECURRENT_WEIGHT
            netcon.delay = RECURRENT_DELAY_MS
            recurrent_netcons.append(netcon)

    if CORENEURON_CPU:
        coreneuron.enable = True
        coreneuron.gpu = False
        coreneuron.verbose = 0
        coreneuron.cell_permute = 1
        if args.skip_core_transfer_after_first:
            original_nrncore_arg = coreneuron.nrncore_arg
            nrncore_arg_calls = {"count": 0}

            def nrncore_arg_with_skip(tstop):
                arg = original_nrncore_arg(tstop)
                nrncore_arg_calls["count"] += 1
                if nrncore_arg_calls["count"] > 1 and "--skip-write-model-to-disk" not in arg:
                    arg += " --skip-write-model-to-disk"
                return arg

            coreneuron.nrncore_arg = nrncore_arg_with_skip
    sim.preRun()
    sim.pc.set_maxstep(EXCHANGE_WINDOW_MS)
    spike_times_vector = h.Vector()
    spike_gids_vector = h.Vector()
    sim.pc.spike_record(-1, spike_times_vector, spike_gids_vector)
    h.finitialize(V_INIT)
pre_run_s = time.perf_counter() - pre_start

if args.build_only:
    print(f"cells={args.cells}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"macro_input_netcons={len(input_netcons)}")
    print(f"recurrent_gid_connects={len(recurrent_netcons)}")
    print(f"macro_input_synapses={len(macro_input_synapses)}")
    print(f"recurrent_synapses={len(recurrent_synapses)}")
    raise SystemExit(0)

delay_steps = np.rint(delays / MACRO_DT_MS).astype(np.int32)
history_capacity = int(np.max(delay_steps)) + 1
delay_offsets = (history_capacity - (delay_steps % history_capacity)) % history_capacity
source_indices = np.arange(roi_count, dtype=np.intp)[None, :]
target_weights = weights[macro_rois]
target_delay_offsets = delay_offsets[macro_rois]
current_exposure, initial_s, _initial_h = make_initial_exposures(roi_count)
region_states_s = initial_s.copy()
history = np.empty((history_capacity, roi_count, len(EXPOSURE_NAMES)), dtype=np.float32)
history[:] = current_exposure
current_input = np.zeros((roi_count, len(EXPOSURE_NAMES)), dtype=np.float32)
fill_region_coupling(
    current_input,
    macro_rois,
    target_weights,
    target_delay_offsets,
    source_indices,
    history,
    0,
    history_capacity,
)

step_count = int(round(duration_ms / MACRO_DT_MS))
times_ms = np.arange(step_count + 1, dtype=np.float64) * MACRO_DT_MS
roi_exposures = np.empty((step_count + 1, roi_count, len(EXPOSURE_NAMES)), dtype=np.float32)
roi_exposures[0] = current_exposure

micro_ca = 0.0
input_seed = float(SEED + 10_000)
last_spike_index = 0
all_spike_times = []
all_spike_gids = []

run_start = time.perf_counter()
with suppress_native_output(args.quiet):
    for exchange_start in range(0, step_count, exchange_step_count):
        exchange_stop = min(step_count, exchange_start + exchange_step_count)
        exchange_start_time = exchange_start * MACRO_DT_MS
        exchange_stop_time = exchange_stop * MACRO_DT_MS

        sim.pc.psolve(exchange_stop_time)
        spike_count = int(spike_times_vector.size())
        exchange_spikes = 0
        spike_index = last_spike_index
        while spike_index < spike_count:
            spike_time = float(spike_times_vector.x[spike_index])
            if spike_time >= exchange_stop_time:
                break
            if exchange_start_time <= spike_time < exchange_stop_time:
                spike_gid = int(spike_gids_vector.x[spike_index])
                exchange_spikes += 1
                all_spike_times.append(spike_time)
                all_spike_gids.append(spike_gid)
            spike_index += 1
        last_spike_index = spike_index

        for step in range(exchange_start, exchange_stop):
            step_regions(macro_rois, region_states_s, current_input, current_exposure)

            if step + 1 == exchange_stop:
                micro_ca += exchange_spikes / float(args.cells)
                micro_ca *= math.exp(-EXCHANGE_WINDOW_MS / EXPOSURE_TAU_MS)
                activity = np.float32(micro_ca * EXPOSURE_GAIN)
                current_exposure[MICRO_ROI, S_INDEX] = activity
                current_exposure[MICRO_ROI, H_INDEX] = activity

            history[(step + 1) % history_capacity] = current_exposure
            roi_exposures[step + 1] = current_exposure
            fill_region_coupling(
                current_input,
                macro_rois,
                target_weights,
                target_delay_offsets,
                source_indices,
                history,
                step + 1,
                history_capacity,
            )

        if exchange_stop < step_count:
            micro_input = apply_micro_coupling(macro_rois, MICRO_ROI, history, exchange_stop, history_capacity)
            event_rate = float(micro_input[H_INDEX]) * INPUT_SPIKE_SCALE
            event_rate = min(max(event_rate, 0.0), MAX_INPUT_RATE_HZ)
            probability = min(max(event_rate * EXCHANGE_WINDOW_MS / 1000.0, 0.0), 1.0)
            for cell in range(args.cells):
                input_seed, u, offset_fraction = lcg_uniform_pair(input_seed)
                if u < probability:
                    event_time = exchange_start_time + offset_fraction * EXCHANGE_WINDOW_MS + MACRO_EVENT_DELAY_MS
                    input_netcons[cell].event(event_time)
run_s = time.perf_counter() - run_start

spike_times = np.asarray(all_spike_times, dtype=np.float64)
spike_gids = np.asarray(all_spike_gids, dtype=np.int32)
order = np.argsort(spike_times, kind="stable")
spike_times = spike_times[order]
spike_gids = spike_gids[order]

metadata = [
    float(args.cells),
    float(duration_ms),
    float(DT_MS),
    float(MACRO_DT_MS),
    float(EXCHANGE_WINDOW_MS),
    float(roi_count),
    float(MICRO_ROI),
]
timing = [pre_run_s, run_s, pre_run_s + run_s]
save_h5(output_file, times_ms, roi_exposures, labels, spike_times, spike_gids, timing, metadata)

print(f"output={output_file}")
print(f"pre_run_s={pre_run_s:.6f}")
print(f"run_s={run_s:.6f}")
print(f"total_s={pre_run_s + run_s:.6f}")
print(f"spikes={len(spike_times)}")
