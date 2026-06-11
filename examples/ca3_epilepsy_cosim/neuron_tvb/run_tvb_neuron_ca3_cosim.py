#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200
CA3_CELL_COUNT = PYR_COUNT + BAS_COUNT + OLM_COUNT
SPIKE_THRESHOLD_MV = 0.0

MACRO_DT_MS = 0.1
MICRO_DT_MS = 0.025
MIN_NETCON_DELAY_MS = 2.0 * MICRO_DT_MS
EXCHANGE_WINDOW_MS = 0.5
SAMPLE_EPS_MS = 1.0e-12
NEURON_TIME_ALIGN_EPS_MS = 1.0e-7

PYR_CURRENT_NA = 0.1
OLM_CURRENT_NA = -25e-3


class DoubleSparseHistory:
    def __init__(self, weights, delays, cvars, n_mode):
        self.weights = np.asarray(weights, dtype=float)
        self.delays = np.asarray(delays, dtype=int)
        self.cvars = np.asarray(cvars, dtype=int)
        self.n_time = int(self.delays.max()) + 1
        self.n_cvar = int(len(self.cvars))
        self.n_node = int(self.delays.shape[0])
        self.n_mode = int(n_mode)
        self.time_stride = self.n_cvar * self.n_node * self.n_mode
        self.nnz_mask = self.weights != 0.0
        self.n_nnzw = int(self.nnz_mask.sum())
        self.nnz_weights = self.weights[self.nnz_mask]
        self.nnz_row_el_idx, self.nnz_col_el_idx = np.argwhere(self.nnz_mask).T
        nnz_row_idx = np.unique(self.nnz_row_el_idx)
        self.n_nnzr = int(len(nnz_row_idx))
        self.nnz_row_idx = nnz_row_idx
        self.nnz_idelays = self.delays[self.nnz_mask].astype(int)
        icvars = np.r_[: self.n_cvar].reshape((-1, 1, 1)) * self.n_node * self.n_mode
        nodes = np.tile(np.r_[: self.n_node], (self.n_node, 1))[self.nnz_mask, np.newaxis] * self.n_mode
        modes = np.r_[: self.n_mode]
        self.const_indices = icvars + nodes + modes

    def initialize(self, initial_history):
        if initial_history.shape[1] > self.n_cvar:
            initial_history = initial_history[:, self.cvars]
        if initial_history.shape[0] != self.n_time:
            raise ValueError(
                f"DoubleSparseHistory expects {self.n_time} chronological history samples, "
                f"got {initial_history.shape[0]}"
            )
        self.buffer = np.asarray(initial_history, dtype=float).copy()
        self.step_offset = int(initial_history.shape[0]) - 1

    def update(self, step, state):
        absolute_step = self.step_offset + int(step)
        self.buffer[absolute_step % self.n_time] = state[self.cvars]

    def query_sparse(self, step):
        absolute_step = self.step_offset + int(step)
        time_indices = ((absolute_step - 1 - self.nnz_idelays + self.n_time) % self.n_time).reshape((-1, 1))
        delayed_state = self.buffer.take(time_indices * self.time_stride + self.const_indices)
        current_state = self.buffer[(absolute_step - 1) % self.n_time]
        return current_state, delayed_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure TVB + NEURON CA3 co-simulation reference.")
    parser.add_argument(
        "--connectivity-csv",
        type=Path,
        required=True,
        help="Labelled matrix CSV with weights and delays_ms sections.",
    )
    parser.add_argument("--duration-ms", "--t-ms", dest="duration_ms", type=float, default=20.0)
    parser.add_argument("--macro-i-ext", "--i-ext", dest="macro_i_ext", type=float, default=3.1)
    parser.add_argument("--drive-weight", "--drive-w", dest="drive_weight", type=float, default=0.02e-3)
    parser.add_argument("--micro-threads", type=int, default=4)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_neuron_ca3_cosim.npz"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_neuron_ca3_workdir"),
    )
    parser.add_argument("--tvb-root", type=Path, default=Path.home() / "tvb-root")
    args = parser.parse_args()

    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    if args.micro_threads < 1:
        raise SystemExit("--micro-threads must be positive")
    duration_steps_float = float(args.duration_ms) / MACRO_DT_MS
    duration_steps = int(round(duration_steps_float))
    if not math.isclose(duration_steps_float, duration_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise SystemExit("--duration-ms must be an integer multiple of 0.1 ms")
    exchange_steps_float = EXCHANGE_WINDOW_MS / MACRO_DT_MS
    exchange_steps = int(round(exchange_steps_float))
    if not math.isclose(exchange_steps_float, exchange_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise SystemExit("exchange window must be an integer multiple of dt_macro")

    for path in reversed(
        [
            args.tvb_root / "tvb_library",
            args.tvb_root / "tvb_contrib",
            args.tvb_root / "tvb_storage",
            args.tvb_root / "tvb_framework",
        ]
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from neuron import h, load_mechanisms

    h.load_file("stdrun.hoc")

    pre_start = time.perf_counter()
    mod_dir = Path(__file__).resolve().parent / "mod"
    args.workdir.mkdir(parents=True, exist_ok=True)
    mechanism_dir = mod_dir
    if not (mod_dir / "x86_64" / "libnrnmech.so").exists():
        subprocess.run(["nrnivmodl", "."], cwd=mod_dir, check=True)
    load_mechanisms(str(mod_dir))

    rows = []
    with args.connectivity_csv.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            row = [cell.strip() for cell in row]
            if any(row):
                rows.append(row)
    if not rows:
        raise SystemExit(f"connectivity CSV is empty: {args.connectivity_csv}")

    weights_marker = -1
    delays_marker = -1
    for index, row in enumerate(rows):
        if len(row) >= 2 and row[0] == "matrix" and row[1] == "weights":
            weights_marker = index
        if len(row) >= 2 and row[0] == "matrix" and row[1] == "delays_ms":
            delays_marker = index
    if weights_marker < 0 or delays_marker < 0 or delays_marker <= weights_marker:
        raise SystemExit("connectivity CSV must contain matrix,weights followed by matrix,delays_ms")

    labels = rows[weights_marker + 1][1:]
    delay_labels = rows[delays_marker + 1][1:]
    if not labels or delay_labels != labels:
        raise SystemExit("connectivity CSV weights and delays_ms labels do not match")
    roi_count = len(labels)
    weights = np.empty((roi_count, roi_count), dtype=float)
    delays = np.empty((roi_count, roi_count), dtype=float)
    weight_row_labels = []
    for row_index, row in enumerate(rows[weights_marker + 2 : delays_marker]):
        if len(row) != roi_count + 1:
            raise SystemExit("connectivity CSV weights row width does not match labels")
        weight_row_labels.append(row[0])
        weights[row_index] = [float(value) for value in row[1:]]
    delay_row_labels = []
    for row_index, row in enumerate(rows[delays_marker + 2 :]):
        if len(row) != roi_count + 1:
            raise SystemExit("connectivity CSV delays_ms row width does not match labels")
        delay_row_labels.append(row[0])
        delays[row_index] = [float(value) for value in row[1:]]
    if weight_row_labels != labels or delay_row_labels != labels:
        raise SystemExit("connectivity CSV row labels must match column labels")
    for target in range(roi_count):
        for source in range(roi_count):
            weight = float(weights[target, source])
            delay = float(delays[target, source])
            if target == source and (weight != 0.0 or delay != 0.0):
                raise SystemExit("connectivity CSV self edges must be zero")
            if weight <= 0.0 and delay != 0.0:
                raise SystemExit("connectivity CSV delays_ms must be zero when weight <= 0")
            if weight > 0.0 and delay <= 0.0:
                raise SystemExit("active connectivity edges must have positive delays")

    ca3_label = "Left-CA3"
    if ca3_label not in labels:
        raise SystemExit("connectivity labels do not contain ROI 'Left-CA3'")
    ca3_index = labels.index(ca3_label)
    positive_delays = delays[delays > 0.0]
    min_positive_delay = float(np.min(positive_delays)) if positive_delays.size else 0.0
    if min_positive_delay <= 0.0:
        raise SystemExit("connectivity delays must contain at least one positive delay")
    if EXCHANGE_WINDOW_MS > min_positive_delay + 1.0e-9:
        raise SystemExit("exchange window must not exceed the minimum positive connectivity delay")

    from tvb.basic.profile import TvbProfile
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.coupling import Linear
    from tvb.simulator.integrators import EulerDeterministic
    from tvb.simulator.models.epileptor import Epileptor2D

    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    tvb_connectivity = Connectivity(
        weights=weights,
        tract_lengths=delays,
        region_labels=np.asarray(labels, dtype="U"),
        speed=np.asarray([1.0], dtype=float),
        centres=np.zeros((roi_count, 3), dtype=float),
        orientations=np.zeros((roi_count, 3), dtype=float),
        cortical=np.ones(roi_count, dtype=bool),
        hemispheres=np.zeros(roi_count, dtype=bool),
        areas=np.ones(roi_count, dtype=float),
    )
    tvb_connectivity.configure()
    tvb_connectivity.set_idelays(MACRO_DT_MS)
    if np.max(np.abs(np.asarray(tvb_connectivity.delays, dtype=float) - delays)) > 0.0:
        raise RuntimeError("TVB connectivity delay conversion changed the CSV delays")
    history_capacity = int(tvb_connectivity.horizon)
    macro_roi_indices = np.asarray([index for index in range(roi_count) if index != ca3_index], dtype=int)

    pc = h.ParallelContext()
    pc.nthread(int(args.micro_threads))

    pyr_population = []
    bas_population = []
    olm_population = []
    cells = []
    point_processes = []

    for gid in range(PYR_COUNT):
        soma = h.Section(name=f"PYR[{gid}].soma")
        bdend = h.Section(name=f"PYR[{gid}].Bdend")
        adend1 = h.Section(name=f"PYR[{gid}].Adend1")
        adend2 = h.Section(name=f"PYR[{gid}].Adend2")
        adend3 = h.Section(name=f"PYR[{gid}].Adend3")
        soma.nseg = 1
        bdend.nseg = 1
        adend1.nseg = 1
        adend2.nseg = 1
        adend3.nseg = 1
        h.pt3dclear(sec=soma)
        h.pt3dadd(0.0, 0.0, 0.0, 20.0, sec=soma)
        h.pt3dadd(0.0, 0.0, 20.0, 20.0, sec=soma)
        h.pt3dclear(sec=bdend)
        h.pt3dadd(0.0, 0.0, 0.0, 2.0, sec=bdend)
        h.pt3dadd(0.0, 0.0, -200.0, 2.0, sec=bdend)
        h.pt3dclear(sec=adend1)
        h.pt3dadd(0.0, 0.0, 20.0, 2.0, sec=adend1)
        h.pt3dadd(0.0, 0.0, 170.0, 2.0, sec=adend1)
        h.pt3dclear(sec=adend2)
        h.pt3dadd(0.0, 0.0, 170.0, 2.0, sec=adend2)
        h.pt3dadd(0.0, 0.0, 320.0, 2.0, sec=adend2)
        h.pt3dclear(sec=adend3)
        h.pt3dadd(0.0, 0.0, 320.0, 2.0, sec=adend3)
        h.pt3dadd(0.0, 0.0, 470.0, 2.0, sec=adend3)
        bdend.connect(soma(0.0), 0.0)
        adend1.connect(soma(0.5), 0.0)
        adend2.connect(adend1(1.0), 0.0)
        adend3.connect(adend2(1.0), 0.0)

        for section in (soma, bdend, adend1, adend2, adend3):
            section.Ra = 150.0
            section.cm = 1.0
            section.insert("kdrcurrent")

        soma.insert("pas")
        soma.insert("nacurrent")
        soma.insert("kacurrent")
        soma.insert("hcurrent")
        for segment in soma:
            segment.pas.e = -70.0
            segment.pas.g = 0.0000357

        bdend.insert("pas")
        bdend.insert("nacurrent")
        bdend.insert("kacurrent")
        bdend.insert("hcurrent")
        for segment in bdend:
            segment.pas.e = -70.0
            segment.pas.g = 0.0000357
            segment.nacurrent.ki = 1.0

        adend1.insert("pas")
        adend1.insert("nacurrent")
        adend1.insert("kacurrent")
        adend1.insert("hcurrent")
        for segment in adend1:
            segment.pas.e = -70.0
            segment.pas.g = 0.0000357
            segment.nacurrent.ki = 0.5
            segment.kacurrent.g = 0.072
            segment.hcurrent.v50 = -82.0
            segment.hcurrent.g = 0.0002

        adend2.insert("pas")
        adend2.insert("nacurrent")
        adend2.insert("kacurrent")
        adend2.insert("hcurrent")
        for segment in adend2:
            segment.pas.e = -70.0
            segment.pas.g = 0.0000357
            segment.nacurrent.ki = 0.5
            segment.kacurrent.g = 0.0
            segment.kacurrent.gd = 0.120
            segment.hcurrent.v50 = -90.0
            segment.hcurrent.g = 0.0004

        adend3.cm = 2.0
        adend3.insert("pas")
        adend3.insert("nacurrent")
        adend3.insert("kacurrent")
        adend3.insert("hcurrent")
        for segment in adend3:
            segment.pas.e = -70.0
            segment.pas.g = 0.0000714
            segment.nacurrent.ki = 0.5
            segment.kacurrent.g = 0.0
            segment.kacurrent.gd = 0.200
            segment.hcurrent.v50 = -90.0
            segment.hcurrent.g = 0.0007

        inj = h.IClamp(soma(0.5))
        inj.delay = 0.2
        inj.dur = 1.0e9
        inj.amp = PYR_CURRENT_NA
        point_processes.append(inj)

        cell = {
            "gid": int(gid),
            "pop": "PYR",
            "soma": soma,
            "Bdend": bdend,
            "Adend1": adend1,
            "Adend2": adend2,
            "Adend3": adend3,
        }
        pyr_population.append(cell)
        cells.append(cell)

    interneuron_area_um2 = 10000.0
    interneuron_diam = math.sqrt(interneuron_area_um2)
    interneuron_length = interneuron_diam / math.pi
    for local_gid in range(BAS_COUNT):
        gid = PYR_COUNT + local_gid
        soma = h.Section(name=f"BAS[{local_gid}].soma")
        soma.nseg = 1
        h.pt3dclear(sec=soma)
        h.pt3dadd(0.0, 0.0, 0.0, interneuron_diam, sec=soma)
        h.pt3dadd(0.0, 0.0, interneuron_length, interneuron_diam, sec=soma)
        soma.Ra = 35.4
        soma.cm = 1.0
        soma.insert("pas")
        soma.insert("Nafbwb")
        soma.insert("Kdrbwb")
        for segment in soma:
            segment.pas.e = -65.0
            segment.pas.g = 0.1e-3
        cell = {"gid": int(gid), "pop": "BAS", "soma": soma}
        bas_population.append(cell)
        cells.append(cell)

    for local_gid in range(OLM_COUNT):
        gid = PYR_COUNT + BAS_COUNT + local_gid
        soma = h.Section(name=f"OLM[{local_gid}].soma")
        soma.nseg = 1
        h.pt3dclear(sec=soma)
        h.pt3dadd(0.0, 0.0, 0.0, interneuron_diam, sec=soma)
        h.pt3dadd(0.0, 0.0, interneuron_length, interneuron_diam, sec=soma)
        soma.Ra = 35.4
        soma.cm = 1.0
        soma.insert("pas")
        soma.insert("Nafbwb")
        soma.insert("Kdrbwb")
        soma.insert("Iholmw")
        soma.insert("Caolmw")
        soma.insert("ICaolmw")
        soma.insert("KCaolmw")
        for segment in soma:
            segment.pas.e = -65.0
            segment.pas.g = 0.1e-3
        inj = h.IClamp(soma(0.5))
        inj.delay = 0.2
        inj.dur = 1.0e9
        inj.amp = OLM_CURRENT_NA
        point_processes.append(inj)
        cell = {"gid": int(gid), "pop": "OLM", "soma": soma}
        olm_population.append(cell)
        cells.append(cell)

    source_netcons = []
    for cell in cells:
        gid = int(cell["gid"])
        soma = cell["soma"]
        source = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        source.threshold = SPIKE_THRESHOLD_MV
        source.delay = MIN_NETCON_DELAY_MS
        pc.set_gid2node(gid, int(pc.id()))
        pc.cell(gid, source)
        source_netcons.append(source)

    recurrent_netcons = []
    conn_rng = random.Random(4321)

    for cell in bas_population:
        target = h.MyExp2SynNMDABB(cell["soma"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.tau1NMDA = 15.0
        target.tau2NMDA = 150.0
        target.r = 1.0
        target.e = 0.0
        point_processes.append(target)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 100):
            netcon = pc.gid_connect(int(pyr_local), target)
            netcon.weight[0] = 1.15 * 1.2e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in olm_population:
        target = h.MyExp2SynNMDABB(cell["soma"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.tau1NMDA = 15.0
        target.tau2NMDA = 150.0
        target.r = 1.0
        target.e = 0.0
        point_processes.append(target)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 10):
            netcon = pc.gid_connect(int(pyr_local), target)
            netcon.weight[0] = 0.7e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in pyr_population:
        pyr_local_post = int(cell["gid"])
        target = h.MyExp2SynNMDABB(cell["Bdend"](1.0))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.tau1NMDA = 15.0
        target.tau2NMDA = 150.0
        target.r = 1.0
        target.e = 0.0
        point_processes.append(target)
        for pyr_local_pre in conn_rng.sample(range(PYR_COUNT), 25):
            if pyr_local_pre == pyr_local_post:
                continue
            netcon = pc.gid_connect(int(pyr_local_pre), target)
            netcon.weight[0] = 0.004e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in bas_population:
        target = h.MyExp2SynBB(cell["soma"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.e = 0.0
        point_processes.append(target)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 100):
            netcon = pc.gid_connect(int(pyr_local), target)
            netcon.weight[0] = 0.3 * 1.2e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in olm_population:
        target = h.MyExp2SynBB(cell["soma"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.e = 0.0
        point_processes.append(target)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 10):
            netcon = pc.gid_connect(int(pyr_local), target)
            netcon.weight[0] = 0.3 * 1.2e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in pyr_population:
        pyr_local_post = int(cell["gid"])
        target = h.MyExp2SynBB(cell["Bdend"](1.0))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.e = 0.0
        point_processes.append(target)
        for pyr_local_pre in conn_rng.sample(range(PYR_COUNT), 25):
            if pyr_local_pre == pyr_local_post:
                continue
            netcon = pc.gid_connect(int(pyr_local_pre), target)
            netcon.weight[0] = 0.5 * 0.04e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in bas_population:
        bas_local_post = int(cell["gid"]) - PYR_COUNT
        target = h.MyExp2SynBB(cell["soma"](0.5))
        target.tau1 = 0.07
        target.tau2 = 9.1
        target.e = -80.0
        point_processes.append(target)
        for bas_local_pre in conn_rng.sample(range(BAS_COUNT), 60):
            if bas_local_pre == bas_local_post:
                continue
            netcon = pc.gid_connect(PYR_COUNT + int(bas_local_pre), target)
            netcon.weight[0] = 3.0 * 1.5e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in pyr_population:
        target = h.MyExp2SynBB(cell["soma"](0.5))
        target.tau1 = 0.07
        target.tau2 = 9.1
        target.e = -80.0
        point_processes.append(target)
        for bas_local in conn_rng.sample(range(BAS_COUNT), 50):
            netcon = pc.gid_connect(PYR_COUNT + int(bas_local), target)
            netcon.weight[0] = 4.0 * 0.18e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in olm_population:
        target = h.MyExp2SynBB(cell["soma"](0.5))
        target.tau1 = 0.07
        target.tau2 = 9.1
        target.e = -80.0
        point_processes.append(target)
        for bas_local in conn_rng.sample(range(BAS_COUNT), 17):
            netcon = pc.gid_connect(PYR_COUNT + int(bas_local), target)
            netcon.weight[0] = 0.05 * 4.0 * 0.18e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    for cell in pyr_population:
        target = h.MyExp2SynBB(cell["Adend2"](0.5))
        target.tau1 = 0.2
        target.tau2 = 20.0
        target.e = -80.0
        point_processes.append(target)
        for olm_local in conn_rng.sample(range(OLM_COUNT), 10):
            netcon = pc.gid_connect(PYR_COUNT + BAS_COUNT + int(olm_local), target)
            netcon.weight[0] = 0.08 * 4.0 * 3.0 * 6.0e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)

    effective_drive_weight = float(args.drive_weight) * 1.0e-2
    macro_input_netcons = []
    for cell in pyr_population:
        target = h.MyExp2SynBB(cell["Adend3"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.e = 0.0
        point_processes.append(target)
        netcon = h.NetCon(None, target)
        netcon.weight[0] = effective_drive_weight
        netcon.delay = MIN_NETCON_DELAY_MS
        macro_input_netcons.append(netcon)

    spike_times_vector = h.Vector()
    spike_gids_vector = h.Vector()
    pc.spike_record(-1, spike_times_vector, spike_gids_vector)

    voltage_time_vector = h.Vector().record(h._ref_t)
    pyr_voltage_vector = h.Vector().record(pyr_population[0]["soma"](0.5)._ref_v)
    bas_voltage_vector = h.Vector().record(bas_population[0]["soma"](0.5)._ref_v)
    olm_voltage_vector = h.Vector().record(olm_population[0]["soma"](0.5)._ref_v)
    adend3_voltage_vector = h.Vector().record(pyr_population[0]["Adend3"](0.5)._ref_v)

    macro_rng = np.random.default_rng(1234)
    propagation_labels = {
        "Left-CA1",
        "Right-CA1",
        "Left-CA3",
        "Right-CA3",
        "Left-subiculum",
        "Right-subiculum",
        "Left-entorhinal",
        "Right-entorhinal",
    }
    x = np.empty(roi_count, dtype=float)
    z = np.zeros(roi_count, dtype=float)
    x0_param = np.empty(roi_count, dtype=float)
    for roi_index, label in enumerate(labels):
        if roi_index == ca3_index:
            x0 = -1.6
        elif label in propagation_labels:
            x0 = -1.9
        else:
            x0 = -2.4
        x0_param[roi_index] = x0
        x[roi_index] = x0 + 0.02 * float(macro_rng.standard_normal())
    x[ca3_index] = 0.0

    macro_model = Epileptor2D(
        a=np.asarray([1.0], dtype=float),
        b=np.asarray([3.0], dtype=float),
        c=np.asarray([1.0], dtype=float),
        d=np.asarray([5.0], dtype=float),
        r=np.asarray([0.00035], dtype=float),
        x0=x0_param,
        Iext=np.asarray([float(args.macro_i_ext)], dtype=float),
        slope=np.asarray([0.0], dtype=float),
        Kvf=np.asarray([0.35], dtype=float),
        Ks=np.asarray([0.0], dtype=float),
        tt=np.asarray([1.0], dtype=float),
        modification=np.asarray([False], dtype=bool),
        variables_of_interest=("x1", "z"),
    )
    macro_model.configure()
    macro_integrator = EulerDeterministic(dt=MACRO_DT_MS)
    macro_integrator.configure()
    macro_coupling = Linear(a=np.asarray([1.0], dtype=float), b=np.asarray([0.0], dtype=float))
    macro_coupling.configure()

    initial_history = np.empty(
        (history_capacity, len(macro_model.state_variables), roi_count, macro_model.number_of_modes),
        dtype=float,
    )
    history_alpha = np.linspace(-1.0, 0.0, history_capacity, dtype=float)[:, np.newaxis]
    roi_phase = np.linspace(0.0, 2.0 * np.pi, roi_count, endpoint=False, dtype=float)[np.newaxis, :]
    chronological_x = x + 0.01 * history_alpha * np.sin(roi_phase)
    chronological_z = z + 0.002 * history_alpha * np.cos(roi_phase)
    chronological_x[-1] = x
    chronological_z[-1] = z
    initial_history[:, 0, :, 0] = chronological_x
    initial_history[:, 1, :, 0] = chronological_z
    macro_history = DoubleSparseHistory(
        np.asarray(tvb_connectivity.weights, dtype=float),
        np.asarray(tvb_connectivity.idelays, dtype=int),
        macro_model.cvar,
        macro_model.number_of_modes,
    )
    macro_history.initialize(initial_history)

    time_ms = np.arange(duration_steps + 1, dtype=float) * MACRO_DT_MS
    macro_x = np.empty((duration_steps + 1, roi_count), dtype=float)
    macro_z = np.empty((duration_steps + 1, roi_count), dtype=float)
    macro_x[0] = x
    macro_z[0] = z
    macro_z[0, ca3_index] = np.nan

    seed = 1
    seed_low = int(seed) & 0xFFFFFFFF
    seed_high = (int(seed) >> 32) & 0xFFFFFFFF
    macro2micro_streams = []
    for neuron_index in range(PYR_COUNT):
        stream = h.Random()
        stream.Random123(int(seed_low), int(neuron_index), int(seed_high ^ int(ca3_index)))
        macro2micro_streams.append(stream)

    h.dt = MICRO_DT_MS
    h.CVode().active(0)
    pc.set_maxstep(10.0)
    h.finitialize(-65.0)
    pc.psolve(0.0)
    if abs(float(h.t)) > NEURON_TIME_ALIGN_EPS_MS:
        raise RuntimeError(f"NEURON time drift after initialization: h.t={float(h.t):.17g}")
    h.t = 0.0

    pyr_activity = 0.0
    bas_activity = 0.0
    olm_activity = 0.0
    scheduled_macro2micro_events = 0

    pre_run_s = time.perf_counter() - pre_start
    run_start = time.perf_counter()

    for exchange_start in range(0, duration_steps, exchange_steps):
        exchange_stop = min(duration_steps, exchange_start + exchange_steps)
        exchange_start_time = float(exchange_start) * MACRO_DT_MS

        for step in range(exchange_start, exchange_stop):
            interval_start = float(step) * MACRO_DT_MS
            interval_stop = min(float(args.duration_ms), interval_start + MACRO_DT_MS)
            if interval_start >= float(args.duration_ms) - SAMPLE_EPS_MS:
                continue
            macro_coupling_input = macro_coupling(step + 1, macro_history)
            ca3_input = float(macro_coupling_input[0, ca3_index, 0])
            exponent = np.clip(-4.0 * (ca3_input - (-0.35)), -60.0, 60.0)
            rate_hz = 1.0 + 45.0 / (1.0 + math.exp(float(exponent)))
            rate_hz = float(np.clip(rate_hz, 0.0, 120.0))
            window_ms = interval_stop - interval_start
            mean = rate_hz * window_ms / 1000.0
            if mean <= 0.0:
                continue
            for neuron_index, stream in enumerate(macro2micro_streams):
                stream.seq(max(0, int(step)))
                limit = math.exp(-mean)
                product = 1.0
                spike_count = -1
                while product > limit:
                    spike_count += 1
                    product *= float(stream.uniform(0.0, 1.0))
                for _ in range(int(spike_count)):
                    event_time = interval_start + window_ms * float(stream.uniform(0.0, 1.0))
                    delivery_time = event_time + 0.2
                    if delivery_time <= float(h.t) + NEURON_TIME_ALIGN_EPS_MS:
                        raise RuntimeError(
                            "macro2micro generated an already elapsed delivery: "
                            f"event={event_time:.17g}, delivery={delivery_time:.17g}, h.t={float(h.t):.17g}"
                        )
                    macro_input_netcons[neuron_index].event(delivery_time)
                    scheduled_macro2micro_events += 1

        exchange_stop_time = float(exchange_stop) * MACRO_DT_MS
        pc.psolve(exchange_stop_time)
        if abs(float(h.t) - exchange_stop_time) > NEURON_TIME_ALIGN_EPS_MS:
            raise RuntimeError(
                f"NEURON time drift at exchange {exchange_start}:{exchange_stop}: "
                f"h.t={float(h.t):.17g}, nominal={exchange_stop_time:.17g}"
            )
        h.t = exchange_stop_time

        spike_times = np.asarray(spike_times_vector.to_python(), dtype=float)
        spike_gids = np.asarray(spike_gids_vector.to_python(), dtype=int)
        if spike_times.size:
            order = np.lexsort((spike_gids, spike_times))
            spike_times = spike_times[order]
            spike_gids = spike_gids[order]

        spike_cursor = int(np.searchsorted(spike_times, exchange_start_time, side="left"))
        for step in range(exchange_start, exchange_stop):
            start_time = float(step) * MACRO_DT_MS
            stop_time = float(step + 1) * MACRO_DT_MS
            current_time = start_time
            while spike_cursor < spike_times.size:
                spike_time = float(spike_times[spike_cursor])
                if step + 1 == exchange_stop:
                    if spike_time >= exchange_stop_time:
                        break
                elif spike_time > stop_time + SAMPLE_EPS_MS:
                    break
                event_time = min(max(spike_time, current_time), stop_time)
                decay_ms = event_time - current_time
                if decay_ms > 0.0:
                    pyr_activity *= math.exp(-decay_ms / 50.0)
                    bas_activity *= math.exp(-decay_ms / 20.0)
                    olm_activity *= math.exp(-decay_ms / 80.0)
                current_time = event_time
                gid = int(spike_gids[spike_cursor])
                if 0 <= gid < PYR_COUNT:
                    pyr_activity += 1.0 / float(PYR_COUNT)
                elif PYR_COUNT <= gid < PYR_COUNT + BAS_COUNT:
                    bas_activity += 1.0 / float(BAS_COUNT)
                elif PYR_COUNT + BAS_COUNT <= gid < CA3_CELL_COUNT:
                    olm_activity += 1.0 / float(OLM_COUNT)
                spike_cursor += 1
            decay_ms = stop_time - current_time
            if decay_ms > 0.0:
                pyr_activity *= math.exp(-decay_ms / 50.0)
                bas_activity *= math.exp(-decay_ms / 20.0)
                olm_activity *= math.exp(-decay_ms / 80.0)
            x[ca3_index] = -1.8 + 2.0 * pyr_activity - 0.7 * bas_activity - 0.4 * olm_activity

            state = np.empty(
                (len(macro_model.state_variables), roi_count, macro_model.number_of_modes),
                dtype=float,
            )
            state[0, :, 0] = x
            state[1, :, 0] = z
            macro_coupling_input = macro_coupling(step + 1, macro_history)
            next_state = macro_integrator.scheme(state, macro_model.dfun, macro_coupling_input, 0.0, 0.0)
            x[macro_roi_indices] = next_state[0, macro_roi_indices, 0]
            z[macro_roi_indices] = next_state[1, macro_roi_indices, 0]
            state[0, :, 0] = x
            state[1, :, 0] = z
            macro_history.update(step + 1, state)
            macro_x[step + 1] = x
            macro_z[step + 1] = z
            macro_z[step + 1, ca3_index] = np.nan

    run_s = time.perf_counter() - run_start

    voltage_time = np.asarray(voltage_time_vector.to_python(), dtype=float)
    pyr_voltage = np.asarray(pyr_voltage_vector.to_python(), dtype=float)
    bas_voltage = np.asarray(bas_voltage_vector.to_python(), dtype=float)
    olm_voltage = np.asarray(olm_voltage_vector.to_python(), dtype=float)
    voltage_traces = np.empty((voltage_time.size, 3), dtype=float)
    voltage_traces[:, 0] = pyr_voltage
    voltage_traces[:, 1] = bas_voltage
    voltage_traces[:, 2] = olm_voltage
    adend3_voltage = np.asarray(adend3_voltage_vector.to_python(), dtype=float)
    voltage_labels = np.asarray(["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"], dtype=object)

    metadata = {
        "source": "TVB Connectivity/SparseHistory/Linear coupling/Euler Epileptor2D plus direct NEURON ModelDB 186768 CA3 micro reference",
        "macro_backend": "TVB Connectivity, Linear coupling, EulerDeterministic, and Epileptor2D with a TVB-compatible double-precision sparse history",
        "micro_backend": "NEURON",
        "interface": "direct serial TVB+NEURON loop; macro2micro via NetCon.event; recurrent micro via ParallelContext gid_connect",
        "connectivity_csv": str(args.connectivity_csv),
        "connectivity_format": "matrix_csv_v1",
        "ca3_label": ca3_label,
        "ca3_index": int(ca3_index),
        "duration_ms": float(args.duration_ms),
        "dt_macro_ms": MACRO_DT_MS,
        "dt_micro_ms": MICRO_DT_MS,
        "micro_num_threads": int(args.micro_threads),
        "exchange_window_ms": EXCHANGE_WINDOW_MS,
        "min_positive_delay_ms": float(min_positive_delay),
        "initial_history_steps": int(history_capacity),
        "initial_history": "explicit non-constant chronological history; history[-1] is the t=0 state",
        "macro_i_ext": float(args.macro_i_ext),
        "epileptor2d_kvf": 0.35,
        "epileptor2d_ks": 0.0,
        "epileptor2d_r": 0.00035,
        "epileptor2d_tt": 1.0,
        "epileptor2d_modification": False,
        "drive_weight": float(args.drive_weight),
        "effective_drive_weight": float(effective_drive_weight),
        "drive_delay_ms": 0.2,
        "pyr_current_na": PYR_CURRENT_NA,
        "olm_current_na": OLM_CURRENT_NA,
        "connections": True,
        "micro_source_reuse": "one registered source NetCon per cell; recurrent connections use pc.gid_connect",
        "macro2micro_random_stream": "NEURON h.Random Random123(seed_low, neuron_index, seed_high^roi) + Knuth Poisson + uniform-in-window event placement",
        "scheduled_macro2micro_events": int(scheduled_macro2micro_events),
        "recorded_spike_count": int(spike_times_vector.size()),
        "micro2macro_window": "[exchange_start, exchange_stop) per exchange window; within an exchange, spikes at a macro sample boundary are consumed by the sample ending at that boundary",
        "mechanism_dir": str(mechanism_dir),
        "voltage_recording": "representative PYR/BAS/OLM soma voltages in voltage_traces; fixed output key voltage remains PYR[0].soma; PYR[0].Adend3(0.5) voltage is in adend3_voltage",
        "voltage_trace_labels": voltage_labels.tolist(),
        "spike_validation": "derive representative PYR/BAS/OLM soma spikes from recorded voltage threshold crossings; no spike array is exported",
        "record_names": ["x", "z"],
        "pre_run_s": float(pre_run_s),
        "run_s": float(run_s),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    macro_records = np.stack((macro_x, macro_z), axis=2)
    timing_s = np.asarray([pre_run_s, run_s, pre_run_s + run_s], dtype=float)
    np.savez_compressed(
        args.output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        record_names=np.asarray(["x", "z"], dtype=object),
        exposure_names=np.asarray(["x", "z"], dtype=object),
        time_ms=time_ms,
        tvb_time_ms=time_ms,
        macro_records=macro_records,
        macro_exposures=macro_records,
        macro_x=macro_x,
        macro_z=macro_z,
        voltage_time=voltage_time,
        voltage=pyr_voltage.copy(),
        voltage_trace_time=voltage_time,
        voltage_labels=voltage_labels,
        voltage_traces=voltage_traces,
        adend3_voltage=adend3_voltage,
        left_ca3_macro_output_x=macro_x[:, ca3_index],
        timing_s=timing_s,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )

    print(f"output={args.output}")
    print("backend=tvb_neuron")
    print("micro_backend=neuron")
    print(f"num_threads={int(args.micro_threads)}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


if __name__ == "__main__":
    main()
