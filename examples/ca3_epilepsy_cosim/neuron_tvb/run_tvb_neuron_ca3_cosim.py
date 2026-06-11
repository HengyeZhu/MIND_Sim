#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200
SPIKE_THRESHOLD_MV = 0.0

CORENEURON_FIXED_STEP_TOL = 1.0e-9


def fixed_step_count(value: float, dt: float, what: str) -> int:
    exact_steps = value / dt
    steps = int(round(exact_steps))
    if not math.isclose(exact_steps, steps, rel_tol=0.0, abs_tol=CORENEURON_FIXED_STEP_TOL):
        raise SystemExit(f"{what} must be an integer multiple of {dt:g} ms")
    return steps


def micro_tick(time_ms: float) -> int:
    return int(round(time_ms / 0.025))


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
    parser.add_argument("--rebuild-mods", action="store_true")
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
    duration_steps = fixed_step_count(args.duration_ms, 0.1, "--duration-ms")
    exchange_steps = fixed_step_count(0.5, 0.1, "exchange window")

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
    from tvb.basic.profile import TvbProfile
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.coupling import Linear
    from tvb.simulator.history import SparseHistory
    from tvb.simulator.integrators import EulerDeterministic
    from tvb.simulator.models.epileptor import Epileptor2D

    h.load_file("stdrun.hoc")
    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    pre_start = time.perf_counter()
    mod_dir = Path(__file__).resolve().parent / "mod"
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_mods:
        for build_name in ("x86_64", "i686", "aarch64", "arm64"):
            shutil.rmtree(mod_dir / build_name, ignore_errors=True)
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
    weights = np.empty((roi_count, roi_count))
    delays = np.empty((roi_count, roi_count))
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
            weight = weights[target, source]
            delay = delays[target, source]
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
    min_positive_delay_steps = int(round(min_positive_delay / 0.1))
    if fixed_step_count(0.5, 0.1, "exchange window") > min_positive_delay_steps:
        raise SystemExit("exchange window must not exceed the minimum positive connectivity delay")

    tvb_connectivity = Connectivity(
        weights=weights,
        tract_lengths=delays,
        region_labels=np.array(labels),
        speed=np.array([1.0]),
        centres=np.zeros((roi_count, 3)),
        orientations=np.zeros((roi_count, 3)),
        cortical=np.ones(roi_count, dtype=bool),
        hemispheres=np.zeros(roi_count, dtype=bool),
        areas=np.ones(roi_count),
    )
    tvb_connectivity.configure()
    tvb_connectivity.set_idelays(0.1)
    if np.max(np.abs(np.asarray(tvb_connectivity.delays) - delays)) > 0.0:
        raise RuntimeError("TVB connectivity delay conversion changed the CSV delays")
    history_capacity = int(tvb_connectivity.horizon)
    macro_roi_indices = np.array([index for index in range(roi_count) if index != ca3_index])

    pc = h.ParallelContext()
    pc.nthread(args.micro_threads)

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
        inj.amp = 0.1
        point_processes.append(inj)

        cell = {
            "gid": gid,
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
        cell = {"gid": gid, "pop": "BAS", "soma": soma}
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
        inj.amp = -25e-3
        point_processes.append(inj)
        cell = {"gid": gid, "pop": "OLM", "soma": soma}
        olm_population.append(cell)
        cells.append(cell)

    source_netcons = []
    for cell in cells:
        gid = int(cell["gid"])
        soma = cell["soma"]
        source = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        source.threshold = SPIKE_THRESHOLD_MV
        source.delay = 0.05
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
            netcon = pc.gid_connect(pyr_local, target)
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
            netcon = pc.gid_connect(pyr_local, target)
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
            netcon = pc.gid_connect(pyr_local_pre, target)
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
            netcon = pc.gid_connect(pyr_local, target)
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
            netcon = pc.gid_connect(pyr_local, target)
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
            netcon = pc.gid_connect(pyr_local_pre, target)
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
            netcon = pc.gid_connect(PYR_COUNT + bas_local_pre, target)
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
            netcon = pc.gid_connect(PYR_COUNT + bas_local, target)
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
            netcon = pc.gid_connect(PYR_COUNT + bas_local, target)
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
            netcon = pc.gid_connect(PYR_COUNT + BAS_COUNT + olm_local, target)
            netcon.weight[0] = 0.08 * 4.0 * 3.0 * 6.0e-3
            netcon.delay = 2.0
            recurrent_netcons.append(netcon)
    effective_drive_weight = args.drive_weight * 1.0e-2
    macro_input_netcons = []
    for cell in pyr_population:
        target = h.MyExp2SynBB(cell["Adend3"](0.5))
        target.tau1 = 0.05
        target.tau2 = 5.3
        target.e = 0.0
        point_processes.append(target)
        netcon = h.NetCon(None, target)
        netcon.weight[0] = effective_drive_weight
        netcon.delay = 0.05
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
    x = np.empty(roi_count)
    z = np.zeros(roi_count)
    x0_param = np.empty(roi_count)
    for roi_index, label in enumerate(labels):
        if roi_index == ca3_index:
            x0 = -1.6
        elif label in propagation_labels:
            x0 = -1.9
        else:
            x0 = -2.4
        x0_param[roi_index] = x0
        x[roi_index] = x0 + 0.02 * macro_rng.standard_normal()
    x[ca3_index] = 0.0

    macro_model = Epileptor2D(
        a=np.array([1.0]),
        b=np.array([3.0]),
        c=np.array([1.0]),
        d=np.array([5.0]),
        r=np.array([0.00035]),
        x0=x0_param,
        Iext=np.array([args.macro_i_ext]),
        slope=np.array([0.0]),
        Kvf=np.array([0.35]),
        Ks=np.array([0.0]),
        tt=np.array([1.0]),
        modification=np.array([False]),
        variables_of_interest=("x1", "z"),
    )
    macro_model.configure()
    macro_integrator = EulerDeterministic(dt=0.1)
    macro_integrator.configure()
    macro_coupling = Linear(a=np.array([1.0]), b=np.array([0.0]))
    macro_coupling.configure()

    initial_history = np.empty(
        (history_capacity, len(macro_model.state_variables), roi_count, macro_model.number_of_modes),
    )
    history_alpha = np.linspace(-1.0, 0.0, history_capacity)[:, np.newaxis]
    roi_phase = np.linspace(0.0, 2.0 * np.pi, roi_count, endpoint=False)[np.newaxis, :]
    chronological_x = x + 0.01 * history_alpha * np.sin(roi_phase)
    chronological_z = z + 0.002 * history_alpha * np.cos(roi_phase)
    chronological_x[-1] = x
    chronological_z[-1] = z
    initial_history[:, 0, :, 0] = chronological_x
    initial_history[:, 1, :, 0] = chronological_z
    macro_history = SparseHistory(
        tvb_connectivity.weights,
        tvb_connectivity.idelays,
        macro_model.cvar,
        macro_model.number_of_modes,
    )
    macro_history.initialize(initial_history)
    macro_step_offset = history_capacity - 1

    time_ms = np.arange(duration_steps + 1) * 0.1
    macro_x = np.empty((duration_steps + 1, roi_count))
    macro_z = np.empty((duration_steps + 1, roi_count))
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

    h.dt = 0.025
    h.CVode().active(0)
    pc.set_maxstep(10.0)
    h.finitialize(-65.0)
    pc.psolve(0.0)
    if micro_tick(float(h.t)) != micro_tick(0.0):
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
        exchange_start_time = exchange_start * 0.1

        for step in range(exchange_start, exchange_stop):
            interval_start = step * 0.1
            interval_stop = min(args.duration_ms, interval_start + 0.1)
            if interval_start >= args.duration_ms:
                continue
            macro_coupling_input = macro_coupling(macro_step_offset + step + 1, macro_history)
            ca3_input = float(macro_coupling_input[0, ca3_index, 0])
            exponent = np.clip(-4.0 * (ca3_input - (-0.35)), -60.0, 60.0)
            rate_hz = 1.0 + 45.0 / (1.0 + math.exp(exponent))
            rate_hz = np.clip(rate_hz, 0.0, 120.0)
            window_ms = interval_stop - interval_start
            mean = rate_hz * window_ms / 1000.0
            if mean <= 0.0:
                continue
            for neuron_index, stream in enumerate(macro2micro_streams):
                stream.seq(max(0, step))
                limit = math.exp(-mean)
                product = 1.0
                spike_count = -1
                while product > limit:
                    spike_count += 1
                    product *= stream.uniform(0.0, 1.0)
                for _ in range(spike_count):
                    event_time = interval_start + window_ms * stream.uniform(0.0, 1.0)
                    delivery_time = event_time + 0.2
                    if delivery_time <= float(h.t):
                        raise RuntimeError(
                            "macro2micro generated an already elapsed delivery: "
                            f"event={event_time:.17g}, delivery={delivery_time:.17g}, h.t={float(h.t):.17g}"
                        )
                    macro_input_netcons[neuron_index].event(delivery_time)
                    scheduled_macro2micro_events += 1

        exchange_stop_time = exchange_stop * 0.1
        pc.psolve(exchange_stop_time)
        if micro_tick(float(h.t)) != micro_tick(exchange_stop_time):
            raise RuntimeError(
                f"NEURON time drift at exchange {exchange_start}:{exchange_stop}: "
                f"h.t={float(h.t):.17g}, nominal={exchange_stop_time:.17g}"
            )
        h.t = exchange_stop_time

        spike_times = np.asarray(spike_times_vector.to_python())
        spike_gids = np.asarray(spike_gids_vector.to_python(), dtype=int)
        if spike_times.size:
            spike_times = np.rint(spike_times / 0.025) * 0.025
            order = np.lexsort((spike_gids, spike_times))
            spike_times = spike_times[order]
            spike_gids = spike_gids[order]

        spike_cursor = int(np.searchsorted(spike_times, exchange_start_time, side="left"))
        for step in range(exchange_start, exchange_stop):
            start_time = step * 0.1
            stop_time = (step + 1) * 0.1
            current_time = start_time
            while spike_cursor < spike_times.size:
                spike_time = float(spike_times[spike_cursor])
                if spike_time >= stop_time:
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
                    pyr_activity += 1.0 / PYR_COUNT
                elif PYR_COUNT <= gid < PYR_COUNT + BAS_COUNT:
                    bas_activity += 1.0 / BAS_COUNT
                elif PYR_COUNT + BAS_COUNT <= gid < PYR_COUNT + BAS_COUNT + OLM_COUNT:
                    olm_activity += 1.0 / OLM_COUNT
                spike_cursor += 1
            decay_ms = stop_time - current_time
            if decay_ms > 0.0:
                pyr_activity *= math.exp(-decay_ms / 50.0)
                bas_activity *= math.exp(-decay_ms / 20.0)
                olm_activity *= math.exp(-decay_ms / 80.0)
            x[ca3_index] = -1.8 + 2.0 * pyr_activity - 0.7 * bas_activity - 0.4 * olm_activity

            state = np.empty(
                (len(macro_model.state_variables), roi_count, macro_model.number_of_modes),
            )
            state[0, :, 0] = x
            state[1, :, 0] = z
            macro_coupling_input = macro_coupling(macro_step_offset + step + 1, macro_history)
            next_state = macro_integrator.scheme(state, macro_model.dfun, macro_coupling_input, 0.0, 0.0)
            x[macro_roi_indices] = next_state[0, macro_roi_indices, 0]
            z[macro_roi_indices] = next_state[1, macro_roi_indices, 0]
            state[0, :, 0] = x
            state[1, :, 0] = z
            macro_history.update(macro_step_offset + step + 1, state)
            macro_x[step + 1] = x
            macro_z[step + 1] = z
            macro_z[step + 1, ca3_index] = np.nan

    run_s = time.perf_counter() - run_start

    voltage_time = np.asarray(voltage_time_vector.to_python())
    pyr_voltage = np.asarray(pyr_voltage_vector.to_python())
    bas_voltage = np.asarray(bas_voltage_vector.to_python())
    olm_voltage = np.asarray(olm_voltage_vector.to_python())
    voltage_traces = np.empty((voltage_time.size, 3))
    voltage_traces[:, 0] = pyr_voltage
    voltage_traces[:, 1] = bas_voltage
    voltage_traces[:, 2] = olm_voltage
    adend3_voltage = np.asarray(adend3_voltage_vector.to_python())
    voltage_labels = ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]

    metadata = {
        "source": "TVB Connectivity/SparseHistory/Linear coupling/Euler Epileptor2D plus direct NEURON ModelDB 186768 CA3 micro reference",
        "macro_backend": "TVB Connectivity, SparseHistory, Linear coupling, EulerDeterministic, and Epileptor2D",
        "micro_backend": "NEURON",
        "interface": "direct serial TVB+NEURON loop; macro2micro via NetCon.event; recurrent micro via ParallelContext gid_connect",
        "connectivity_csv": str(args.connectivity_csv),
        "connectivity_format": "matrix_csv_v1",
        "ca3_label": ca3_label,
        "ca3_index": ca3_index,
        "duration_ms": args.duration_ms,
        "dt_macro_ms": 0.1,
        "dt_micro_ms": 0.025,
        "micro_num_threads": args.micro_threads,
        "exchange_window_ms": 0.5,
        "min_positive_delay_ms": float(min_positive_delay),
        "initial_history_steps": int(history_capacity),
        "initial_history": "explicit non-constant chronological history following TVB SparseHistory.from_simulator semantics; history[-1] is t=0 and current_step is offset by history_steps - 1",
        "macro_i_ext": args.macro_i_ext,
        "epileptor2d_kvf": 0.35,
        "epileptor2d_ks": 0.0,
        "epileptor2d_r": 0.00035,
        "epileptor2d_tt": 1.0,
        "epileptor2d_modification": False,
        "drive_weight": args.drive_weight,
        "effective_drive_weight": effective_drive_weight,
        "drive_delay_ms": 0.2,
        "pyr_current_na": 0.1,
        "olm_current_na": -25e-3,
        "connections": True,
        "micro_source_reuse": "one registered source NetCon per cell; recurrent connections use pc.gid_connect",
        "macro2micro_random_stream": "NEURON h.Random Random123(seed_low, neuron_index, seed_high^roi) + Knuth Poisson + uniform-in-window event placement",
        "scheduled_macro2micro_events": int(scheduled_macro2micro_events),
        "recorded_spike_count": int(spike_times_vector.size()),
        "micro2macro_window": "spike times are canonicalized to 0.025 ms ticks and consumed by half-open macro bins [step, step+1)",
        "mechanism_dir": str(mod_dir),
        "voltage_recording": "representative PYR/BAS/OLM soma voltages in voltage_traces; fixed output key voltage remains PYR[0].soma; PYR[0].Adend3(0.5) voltage is in adend3_voltage",
        "voltage_trace_labels": voltage_labels,
        "spike_validation": "derive representative PYR/BAS/OLM soma spikes from recorded voltage threshold crossings; no spike array is exported",
        "record_names": ["x", "z"],
        "pre_run_s": pre_run_s,
        "run_s": run_s,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    macro_records = np.stack((macro_x, macro_z), axis=2)
    timing_s = [pre_run_s, run_s, pre_run_s + run_s]
    np.savez_compressed(
        args.output,
        labels=labels,
        weights=weights,
        delays=delays,
        record_names=["x", "z"],
        exposure_names=["x", "z"],
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
    print(f"num_threads={args.micro_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


if __name__ == "__main__":
    main()
