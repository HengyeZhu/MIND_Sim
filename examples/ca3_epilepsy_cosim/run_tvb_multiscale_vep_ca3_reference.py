#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
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
)

DEFAULT_TVB_ROOT = Path.home() / "tvb-root"
DEFAULT_TVB_MULTISCALE_ROOT = Path.home() / "tvb-multiscale"
MODELDB_DIR = HERE / "modeldb_186768"
DEFAULT_LABELS = [
    "Left-entorhinal",
    "Left-CA3",
    "Left-CA1",
    "Left-subiculum",
    "Right-CA3",
]
EXPOSURE_NAMES = ["x", "z", "rate"]
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
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def add_tvb_multiscale_paths(tvb_root: Path, tvb_multiscale_root: Path) -> None:
    for path in reversed(
        [
            tvb_root / "tvb_library",
            tvb_root / "tvb_contrib",
            tvb_root / "tvb_storage",
            tvb_root / "tvb_framework",
            tvb_multiscale_root,
        ]
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def require_integer_multiple(value: float, unit: float, label: str) -> int:
    ratio = float(value) / float(unit)
    rounded = round(ratio)
    if rounded < 1 or not np.isclose(ratio, rounded, rtol=0.0, atol=1e-9):
        raise SystemExit(f"{label} must be an integer multiple of {unit}")
    return int(rounded)


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
    delays = np.asarray(data["delays"], dtype=float) if "delays" in data else np.where(weights > 0.0, default_delay_ms, 0.0)
    np.fill_diagonal(weights, 0.0)
    np.fill_diagonal(delays, 0.0)
    return labels, weights, delays


def initial_x0(label: str, ca3_label: str, ca3_x0: float, propagation_x0: float, healthy_x0: float) -> float:
    if label == ca3_label:
        return ca3_x0
    if any(token in label for token in ("CA1", "subiculum", "entorhinal", "CA3")):
        return propagation_x0
    return healthy_x0


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


def pyrcell_rule() -> dict:
    common_pas = {"e": -70.0, "g": 0.0000357}
    return {
        "conds": {"cellType": "PYR"},
        "secs": {
            "soma": {
                "geom": {"L": 20.0, "diam": 20.0, "nseg": 1, "Ra": 150.0, "cm": 1.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {},
                    "kacurrent": {},
                    "kdrcurrent": {},
                    "hcurrent": {},
                },
                "pointps": {"inj": {"mod": "IClamp", "loc": 0.5, "del": 2.0 * DT_MS, "dur": 1.0e9, "amp": 50e-3}},
            },
            "Bdend": {
                "geom": {"L": 200.0, "diam": 2.0, "nseg": 1, "Ra": 150.0, "cm": 1.0},
                "topol": {"parentSec": "soma", "parentX": 0.0, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 1.0},
                    "kacurrent": {},
                    "kdrcurrent": {},
                    "hcurrent": {},
                },
            },
            "Adend1": {
                "geom": {"L": 150.0, "diam": 2.0, "nseg": 1, "Ra": 150.0, "cm": 1.0},
                "topol": {"parentSec": "soma", "parentX": 0.5, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.072},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -82.0, "g": 0.0002},
                },
            },
            "Adend2": {
                "geom": {"L": 150.0, "diam": 2.0, "nseg": 1, "Ra": 150.0, "cm": 1.0},
                "topol": {"parentSec": "Adend1", "parentX": 1.0, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.0, "gd": 0.120},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -90.0, "g": 0.0004},
                },
            },
            "Adend3": {
                "geom": {"L": 150.0, "diam": 2.0, "nseg": 1, "Ra": 150.0, "cm": 2.0},
                "topol": {"parentSec": "Adend2", "parentX": 1.0, "childX": 0.0},
                "mechs": {
                    "pas": {"e": -70.0, "g": 0.0000714},
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.0, "gd": 0.200},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -90.0, "g": 0.0007},
                },
            },
        },
    }


def interneuron_rule(cell_type: str, olm: bool) -> dict:
    total_area_um2 = 10000.0
    diam = math.sqrt(total_area_um2)
    length = diam / math.pi
    mechs = {
        "pas": {"e": -65.0, "g": 0.1e-3},
        "Nafbwb": {},
        "Kdrbwb": {},
    }
    if olm:
        mechs.update({"Iholmw": {}, "Caolmw": {}, "ICaolmw": {}, "KCaolmw": {}})
    pointps = {}
    if olm:
        pointps["inj"] = {"mod": "IClamp", "loc": 0.5, "del": 2.0 * DT_MS, "dur": 1.0e9, "amp": -25e-3}
    return {
        "conds": {"cellType": cell_type},
        "secs": {
            "soma": {
                "geom": {"L": length, "diam": diam, "nseg": 1, "Ra": 35.4, "cm": 1.0},
                "mechs": mechs,
                "pointps": pointps,
            }
        },
    }


SYNAPSE_LOCS = {
    "somaGABAf": ("soma", 0.5, "gaba_fast"),
    "somaAMPAf": ("soma", 0.5, "ampa"),
    "BdendAMPA": ("Bdend", 1.0, "ampa"),
    "BdendNMDA": ("Bdend", 1.0, "nmda"),
    "Adend2GABAs": ("Adend2", 0.5, "gaba_slow"),
    "Adend3GABAf": ("Adend3", 0.5, "gaba_fast"),
    "Adend3AMPAf": ("Adend3", 0.5, "ampa"),
    "Adend3NMDA": ("Adend3", 0.5, "nmda"),
    "somaGABAss": ("soma", 0.5, "gaba_septal"),
    "somaNMDA": ("soma", 0.5, "nmda"),
}


def synapse_params(kind: str) -> dict:
    if kind == "ampa":
        return {"mod": "MyExp2SynBB", "tau1": 0.05, "tau2": 5.3, "e": 0.0}
    if kind == "gaba_fast":
        return {"mod": "MyExp2SynBB", "tau1": 0.07, "tau2": 9.1, "e": -80.0}
    if kind == "gaba_slow":
        return {"mod": "MyExp2SynBB", "tau1": 0.2, "tau2": 20.0, "e": -80.0}
    if kind == "gaba_septal":
        return {"mod": "MyExp2SynBB", "tau1": 20.0, "tau2": 40.0, "e": -80.0}
    if kind == "nmda":
        return {
            "mod": "MyExp2SynNMDABB",
            "tau1": 0.05,
            "tau2": 5.3,
            "tau1NMDA": 15.0,
            "tau2NMDA": 150.0,
            "r": 1.0,
            "e": 0.0,
        }
    raise ValueError(f"unknown synapse kind {kind}")


def make_conn(pre_n: int, post_n: int, conv: int) -> list[list[int]]:
    out: list[list[int]] = []
    for post in range(post_n):
        for pre in random.sample(list(range(pre_n)), conv):
            out.append([int(pre), int(post)])
    return out


def add_projection(net_params, *, pre: str, post: str, syn_name: str, delay: float, weight: float, conv: int) -> None:
    sec, loc, _kind = SYNAPSE_LOCS[syn_name]
    pre_n = {"PYR": PYR_COUNT, "BAS": BAS_COUNT, "OLM": OLM_COUNT}[pre]
    post_n = {"PYR": PYR_COUNT, "BAS": BAS_COUNT, "OLM": OLM_COUNT}[post]
    label = f"{pre}->{post}:{syn_name}"
    net_params.connParams[label] = {
        "preConds": {"pop": pre},
        "postConds": {"pop": post},
        "connList": make_conn(pre_n, post_n, conv),
        "weight": float(weight),
        "delay": float(delay),
        "synMech": syn_name,
        "sec": sec,
        "loc": loc,
    }


def make_netpyne_params(*, connections: bool, wseed: int, duration_ms: float):
    from netpyne import specs

    net_params = specs.NetParams()
    net_params.defaultThreshold = 0.0
    net_params.scale = 1.0
    net_params.sizeX = 1.0
    net_params.sizeY = 1.0
    net_params.sizeZ = 1.0
    net_params.cellParams["PYR"] = pyrcell_rule()
    net_params.cellParams["BAS"] = interneuron_rule("BAS", olm=False)
    net_params.cellParams["OLM"] = interneuron_rule("OLM", olm=True)
    net_params.popParams["PYR"] = {"cellType": "PYR", "numCells": PYR_COUNT}
    net_params.popParams["BAS"] = {"cellType": "BAS", "numCells": BAS_COUNT}
    net_params.popParams["OLM"] = {"cellType": "OLM", "numCells": OLM_COUNT}
    for syn_name, (_sec, _loc, kind) in SYNAPSE_LOCS.items():
        net_params.synMechParams[syn_name] = synapse_params(kind)
    if connections:
        random.seed(int(wseed))
        for pre, post, syn_name, delay, weight, conv in [
            ("PYR", "BAS", "somaNMDA", 2.0, 1.15 * 1.2e-3, 100),
            ("PYR", "OLM", "somaNMDA", 2.0, 0.7e-3, 10),
            ("PYR", "PYR", "BdendNMDA", 2.0, 0.004e-3, 25),
            ("PYR", "BAS", "somaAMPAf", 2.0, 0.3 * 1.2e-3, 100),
            ("PYR", "OLM", "somaAMPAf", 2.0, 0.3 * 1.2e-3, 10),
            ("PYR", "PYR", "BdendAMPA", 2.0, 0.5 * 0.04e-3, 25),
            ("BAS", "BAS", "somaGABAf", 2.0, 3.0 * 1.5e-3, 60),
            ("BAS", "PYR", "somaGABAf", 2.0, 4.0 * 0.18e-3, 50),
            ("BAS", "OLM", "somaGABAf", 2.0, 0.05 * 4.0 * 0.18e-3, 17),
            ("OLM", "PYR", "Adend2GABAs", 2.0, 0.08 * 4.0 * 3.0 * 6.0e-3, 10),
        ]:
            add_projection(net_params, pre=pre, post=post, syn_name=syn_name, delay=delay, weight=weight, conv=conv)

    sim_config = specs.SimConfig()
    sim_config.duration = float(duration_ms)
    sim_config.dt = DT_MS
    sim_config.hParams["v_init"] = V_INIT_MV
    sim_config.createNEURONObj = True
    sim_config.createPyStruct = True
    sim_config.addSynMechs = True
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
    sim_config.analysis = {}
    return net_params, sim_config


def lcg_uniform_pair(seed_state: float) -> tuple[float, float, float]:
    rng = int(seed_state) & ((1 << 64) - 1)
    rng = (rng * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    seed_state = float(rng & 9007199254740991)
    u = seed_state / 9007199254740992.0
    rng = (rng * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    seed_state = float(rng & 9007199254740991)
    offset_fraction = seed_state / 9007199254740992.0
    return seed_state, u, offset_fraction


def coupling_for_step(
    history_x: np.ndarray,
    weights: np.ndarray,
    delay_steps: np.ndarray,
    step: int,
    history_capacity: int,
) -> np.ndarray:
    roi_count = weights.shape[0]
    out = np.zeros(roi_count, dtype=float)
    current_slot = step % history_capacity
    sources = np.arange(roi_count)
    for target in range(roi_count):
        slots = (current_slot - delay_steps[target]) % history_capacity
        out[target] = float(np.dot(weights[target], history_x[slots, sources]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TVB-multiscale/NetPyNE migration of the reduced VEP + CA3 cosim reference.")
    parser.add_argument("--tvb-root", type=Path, default=DEFAULT_TVB_ROOT)
    parser.add_argument("--tvb-multiscale-root", type=Path, default=DEFAULT_TVB_MULTISCALE_ROOT)
    parser.add_argument("--connectivity-npz", type=Path, default=None)
    parser.add_argument("--ca3-label", default="Left-CA3")
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--macro-dt-ms", type=float, default=DT_MS)
    parser.add_argument("--exchange-window-ms", type=float, default=10.0)
    parser.add_argument("--connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--seed", type=int, default=1234)
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
    parser.add_argument("--workdir", type=Path, default=HERE / "outputs" / "tvb_multiscale_vep_ca3_workdir")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tvb_multiscale_vep_ca3_reference.npz")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    require_integer_multiple(args.duration_ms, args.macro_dt_ms, "--duration-ms")
    require_integer_multiple(args.macro_dt_ms, DT_MS, "--macro-dt-ms")
    add_tvb_multiscale_paths(args.tvb_root.expanduser(), args.tvb_multiscale_root.expanduser())

    labels, weights, delays = load_connectivity(args.connectivity_npz, args.exchange_window_ms)
    if args.ca3_label not in labels:
        raise SystemExit(f"{args.ca3_label!r} not found in connectivity labels")
    ca3_index = labels.index(args.ca3_label)
    x0, initial_x, initial_z = build_initial_state(labels, args.ca3_label, args)

    from neuron import h, load_mechanisms
    from netpyne import sim
    from tvb_multiscale.tvb_netpyne.config import Config
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    net_params, sim_config = make_netpyne_params(
        connections=bool(args.connections),
        wseed=int(args.wseed),
        duration_ms=float(args.duration_ms) + float(args.drive_delay_ms) + float(args.macro_dt_ms),
    )

    pre_start = time.perf_counter()
    with suppress_native_output(args.quiet):
        module = NetpyneModule()
        load_mechanisms(str(MODELDB_DIR))
        module.importModel(net_params, sim_config, DT_MS, Config(output_base=str(workdir)))
        module.createNetwork()

        input_netcons = []
        for gid in range(PYR_COUNT):
            cell = sim.net.cells[sim.net.gid2lid[gid]]
            sec = cell.secs["Adend3"]["hObj"]
            syn = h.MyExp2SynBB(0.5, sec=sec)
            syn.tau1 = 0.05
            syn.tau2 = 5.3
            syn.e = 0.0
            nc = h.NetCon(None, syn)
            nc.weight[0] = float(args.drive_weight)
            nc.delay = float(args.drive_delay_ms)
            input_netcons.append(nc)

        module.prepareSimulation(float(args.duration_ms) + float(args.drive_delay_ms) + float(args.macro_dt_ms))
        spike_times_vector = h.Vector()
        spike_gids_vector = h.Vector()
        sim.pc.spike_record(-1, spike_times_vector, spike_gids_vector)
        h.finitialize(V_INIT_MV)
    pre_run_s = time.perf_counter() - pre_start

    roi_count = len(labels)
    step_count = int(round(float(args.duration_ms) / float(args.macro_dt_ms)))
    exchange_step_count = require_integer_multiple(args.exchange_window_ms, args.macro_dt_ms, "--exchange-window-ms")
    times = np.arange(step_count + 1, dtype=float) * float(args.macro_dt_ms)
    delay_steps = np.rint(delays / float(args.macro_dt_ms)).astype(np.int64)
    history_capacity = max(1, int(delay_steps.max(initial=0)) + 1)
    history_x = np.empty((history_capacity, roi_count), dtype=float)
    history_x[:] = initial_x

    x = initial_x.astype(float).copy()
    z = initial_z.astype(float).copy()
    rate = np.zeros(roi_count, dtype=float)
    macro_x = np.zeros((step_count + 1, roi_count), dtype=float)
    macro_z = np.zeros((step_count + 1, roi_count), dtype=float)
    macro_rate = np.zeros((step_count + 1, roi_count), dtype=float)
    ca3_macro_input = np.zeros(step_count + 1, dtype=float)
    ca3_drive_hz = np.zeros(step_count + 1, dtype=float)
    macro_x[0] = x
    macro_z[0] = z

    activity = 0.0
    z_state = 0.0
    input_seed = float(int(args.seed) + 10_000)
    last_spike_index = 0
    all_spike_times: list[float] = []
    all_spike_gids: list[int] = []
    exchange_spikes = 0
    held_ca3_x = float(x[ca3_index])
    held_ca3_z = float(z[ca3_index])
    held_ca3_rate = 0.0

    run_start = time.perf_counter()
    with suppress_native_output(args.quiet):
        for step in range(step_count):
            t0 = step * float(args.macro_dt_ms)
            t1 = (step + 1) * float(args.macro_dt_ms)
            coupled_x = coupling_for_step(history_x, weights, delay_steps, step, history_capacity)
            ca3_input = coupled_x[ca3_index]
            ca3_macro_input[step] = ca3_input
            drive_hz = drive_from_input(
                np.asarray([ca3_input]),
                base_hz=float(args.drive_base_hz),
                gain_hz=float(args.drive_gain_hz),
                max_hz=float(args.drive_max_hz),
                threshold=float(args.drive_threshold),
                slope=float(args.drive_slope),
            )[0]
            ca3_drive_hz[step] = drive_hz
            probability = min(max(drive_hz * float(args.macro_dt_ms) / 1000.0, 0.0), 1.0)
            for cell in range(PYR_COUNT):
                input_seed, u, offset_fraction = lcg_uniform_pair(input_seed)
                if u < probability:
                    input_netcons[cell].event(t0 + offset_fraction * float(args.macro_dt_ms))

            module.run(float(args.macro_dt_ms))

            spike_count = int(spike_times_vector.size())
            step_spikes = 0
            spike_index = last_spike_index
            while spike_index < spike_count:
                spike_time = float(spike_times_vector.x[spike_index])
                if spike_time >= t1 + 1e-9:
                    break
                if t0 - 1e-9 <= spike_time < t1 + 1e-9:
                    spike_gid = int(spike_gids_vector.x[spike_index])
                    all_spike_times.append(spike_time)
                    all_spike_gids.append(spike_gid)
                    step_spikes += 1
                spike_index += 1
            last_spike_index = spike_index

            macro_rois = [idx for idx in range(roi_count) if idx != ca3_index]
            old_x = x.copy()
            new_x = x.copy()
            new_z = z.copy()
            for idx in macro_rois:
                new_x[idx] = old_x[idx] + float(args.macro_dt_ms) * (
                    1.0
                    - old_x[idx] ** 3
                    - 2.0 * old_x[idx] ** 2
                    - z[idx]
                    + float(args.i_ext)
                    + float(args.global_coupling) * coupled_x[idx]
                ) / float(args.tau_x_ms)
                new_z[idx] = z[idx] + float(args.macro_dt_ms) * (
                    4.0 * (new_x[idx] - x0[idx]) - z[idx]
                ) / float(args.tau_z_ms)

            exchange_spikes += step_spikes
            if (step + 1) % exchange_step_count == 0:
                activity += exchange_spikes / float(CA3_CELL_COUNT)
                activity *= math.exp(-float(args.exchange_window_ms) / 50.0)
                held_ca3_rate = 1000.0 * activity / 50.0
                z_state += float(args.exchange_window_ms) * ((activity) - z_state) / float(args.tau_z_ms)
                held_ca3_x = -1.8 + 2.0 * activity
                held_ca3_z = z_state
                exchange_spikes = 0
            new_x[ca3_index] = held_ca3_x
            new_z[ca3_index] = held_ca3_z
            rate[ca3_index] = held_ca3_rate

            x = new_x
            z = new_z
            macro_x[step + 1] = x
            macro_z[step + 1] = z
            macro_rate[step + 1] = rate
            history_x[(step + 1) % history_capacity] = x

        try:
            module.finalize()
        except TypeError:
            # NetPyNE 1.1.1 can fail in postRun timing bookkeeping under the
            # interval runner. Spikes and macro traces have already been
            # collected directly from NEURON vectors at this point.
            pass
    run_s = time.perf_counter() - run_start

    spike_times = np.asarray(all_spike_times, dtype=float)
    spike_gids = np.asarray(all_spike_gids, dtype=np.int32)
    if spike_times.size:
        order = np.lexsort((spike_gids, spike_times))
        spike_times = spike_times[order]
        spike_gids = spike_gids[order]
    ca3_macro_input = delayed_ca3_input(macro_x, weights, delays, ca3_index, float(args.macro_dt_ms))
    ca3_drive_hz = drive_from_input(
        ca3_macro_input,
        base_hz=float(args.drive_base_hz),
        gain_hz=float(args.drive_gain_hz),
        max_hz=float(args.drive_max_hz),
        threshold=float(args.drive_threshold),
        slope=float(args.drive_slope),
    )
    pop_rate = binned_rate(times, spike_times, CA3_CELL_COUNT)

    metadata = {
        "source": "TVB-multiscale NetPyNEModule migration of reduced VEP + ModelDB 186768 CA3",
        "micro_backend": "tvb_multiscale.tvb_netpyne.NetpyneModule + NEURON",
        "macro_backend": "TVB-compatible reduced VEP stepper matched to MIND Sim owner equations",
        "duration_ms": float(args.duration_ms),
        "dt_micro_ms": float(DT_MS),
        "dt_macro_ms": float(args.macro_dt_ms),
        "connections": bool(args.connections),
        "wseed": int(args.wseed),
        "seed": int(args.seed),
    }
    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        exposure_names=np.asarray(EXPOSURE_NAMES, dtype=object),
        time_ms=times,
        macro_x=macro_x,
        macro_z=macro_z,
        ca3_micro_rate_proxy_hz=macro_rate[:, ca3_index],
        ca3_drive_hz=ca3_drive_hz,
        ca3_macro_input=ca3_macro_input,
        ca3_macro_output_x=macro_x[:, ca3_index],
        ca3_macro_output_z=macro_z[:, ca3_index],
        rate_time_ms=times[1:],
        ca3_micro_population_rate_hz=pop_rate,
        spike_times_ms=spike_times,
        spike_gids=spike_gids,
        timing_s=np.asarray([pre_run_s, run_s, pre_run_s + run_s], dtype=float),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"output={output}")
    print("backend=tvb_multiscale_netpyne")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")
    print(f"spikes={len(spike_times)}")


if __name__ == "__main__":
    main()
