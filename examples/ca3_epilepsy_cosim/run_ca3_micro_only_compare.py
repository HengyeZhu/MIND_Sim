#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
PYR_CURRENT_NA = 0.1
OLM_CURRENT_NA = -25e-3
SPIKE_THRESHOLD_MV = 0.0


def voltage_spikes(voltage: np.ndarray, dt_ms: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.asarray(voltage, dtype=float)
    indices = np.flatnonzero((voltage[:-1] < SPIKE_THRESHOLD_MV) & (voltage[1:] >= SPIKE_THRESHOLD_MV)) + 1
    return indices.astype(int), indices.astype(float) * float(dt_ms)


def add_mind_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (root / "src" / "python_api", root / "build"):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def build_mind_ca3_network(ms, *, threads: int):
    micro = ms.Sim()
    micro.set_device("cpu")
    micro.set_num_threads(int(threads))
    micro.set_dt(0.025)
    micro.load_mech(str(Path(__file__).resolve().parent / "mind_sim" / "mod"))

    pyr_soma = ms.section("soma", "soma")
    pyr_soma.nseg = 1
    pyr_soma.pt3d = [(0.0, 0.0, 0.0, 20.0), (0.0, 0.0, 20.0, 20.0)]
    pyr_bdend = ms.section("Bdend", "Bdend")
    pyr_bdend.nseg = 1
    pyr_bdend.pt3d = [(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, -200.0, 2.0)]
    pyr_adend1 = ms.section("Adend1", "Adend1")
    pyr_adend1.nseg = 1
    pyr_adend1.pt3d = [(0.0, 0.0, 20.0, 2.0), (0.0, 0.0, 170.0, 2.0)]
    pyr_adend2 = ms.section("Adend2", "Adend2")
    pyr_adend2.nseg = 1
    pyr_adend2.pt3d = [(0.0, 0.0, 170.0, 2.0), (0.0, 0.0, 320.0, 2.0)]
    pyr_adend3 = ms.section("Adend3", "Adend3")
    pyr_adend3.nseg = 1
    pyr_adend3.pt3d = [(0.0, 0.0, 320.0, 2.0), (0.0, 0.0, 470.0, 2.0)]
    pyr_bdend.connect(pyr_soma, 0.0)
    pyr_adend1.connect(pyr_soma, 0.5)
    pyr_adend2.connect(pyr_adend1, 1.0)
    pyr_adend3.connect(pyr_adend2, 1.0)

    interneuron_area_um2 = 10000.0
    interneuron_diam = math.sqrt(interneuron_area_um2)
    interneuron_length = interneuron_diam / math.pi
    bas_soma = ms.section("soma", "soma")
    bas_soma.nseg = 1
    bas_soma.pt3d = [
        (0.0, 0.0, 0.0, interneuron_diam),
        (0.0, 0.0, interneuron_length, interneuron_diam),
    ]
    olm_soma = ms.section("soma", "soma")
    olm_soma.nseg = 1
    olm_soma.pt3d = [
        (0.0, 0.0, 0.0, interneuron_diam),
        (0.0, 0.0, interneuron_length, interneuron_diam),
    ]
    micro.build_morphology(
        [
            {"name": "PYR", "num_cells": PYR_COUNT, "sections": [pyr_soma, pyr_bdend, pyr_adend1, pyr_adend2, pyr_adend3]},
            {"name": "BAS", "num_cells": BAS_COUNT, "sections": [bas_soma]},
            {"name": "OLM", "num_cells": OLM_COUNT, "sections": [olm_soma]},
        ]
    )

    pyr_population = micro.population("PYR")
    bas_population = micro.population("BAS")
    olm_population = micro.population("OLM")

    for cell in pyr_population:
        cell.v_init = -65.0
        for label in ("soma", "Bdend", "Adend1", "Adend2", "Adend3"):
            group = cell.group(label)
            group.Ra = 150.0
            group.cm = 1.0
            group.insert("kdrcurrent")
        cell.group("soma").insert("pas", e=-70.0, g=0.0000357)
        cell.group("soma").insert("nacurrent")
        cell.group("soma").insert("kacurrent")
        cell.group("soma").insert("hcurrent")
        cell.group("Bdend").insert("pas", e=-70.0, g=0.0000357)
        cell.group("Bdend").insert("nacurrent", ki=1.0)
        cell.group("Bdend").insert("kacurrent")
        cell.group("Bdend").insert("hcurrent")
        cell.group("Adend1").insert("pas", e=-70.0, g=0.0000357)
        cell.group("Adend1").insert("nacurrent", ki=0.5)
        cell.group("Adend1").insert("kacurrent", g=0.072)
        cell.group("Adend1").insert("hcurrent", v50=-82.0, g=0.0002)
        cell.group("Adend2").insert("pas", e=-70.0, g=0.0000357)
        cell.group("Adend2").insert("nacurrent", ki=0.5)
        cell.group("Adend2").insert("kacurrent", g=0.0, gd=0.120)
        cell.group("Adend2").insert("hcurrent", v50=-90.0, g=0.0004)
        cell.group("Adend3").cm = 2.0
        cell.group("Adend3").insert("pas", e=-70.0, g=0.0000714)
        cell.group("Adend3").insert("nacurrent", ki=0.5)
        cell.group("Adend3").insert("kacurrent", g=0.0, gd=0.200)
        cell.group("Adend3").insert("hcurrent", v50=-90.0, g=0.0007)
        cell.group("soma")[0](0.5).insert("IClamp", **{"del": 0.2, "dur": 1.0e9, "amp": PYR_CURRENT_NA})
        micro.network().register_spike_source(int(cell.gid), cell.group("soma")[0](0.5)._ref_v, SPIKE_THRESHOLD_MV)

    for cell in bas_population:
        cell.v_init = -65.0
        soma = cell.group("soma")
        soma.Ra = 35.4
        soma.cm = 1.0
        soma.insert("pas", e=-65.0, g=0.1e-3)
        soma.insert("Nafbwb")
        soma.insert("Kdrbwb")
        micro.network().register_spike_source(int(cell.gid), soma[0](0.5)._ref_v, SPIKE_THRESHOLD_MV)

    for cell in olm_population:
        cell.v_init = -65.0
        soma = cell.group("soma")
        soma.Ra = 35.4
        soma.cm = 1.0
        soma.insert("pas", e=-65.0, g=0.1e-3)
        soma.insert("Nafbwb")
        soma.insert("Kdrbwb")
        soma.insert("Iholmw")
        soma.insert("Caolmw")
        soma.insert("ICaolmw")
        soma.insert("KCaolmw")
        soma[0](0.5).insert("IClamp", **{"del": 0.2, "dur": 1.0e9, "amp": OLM_CURRENT_NA})
        micro.network().register_spike_source(int(cell.gid), soma[0](0.5)._ref_v, SPIKE_THRESHOLD_MV)

    conn_rng = random.Random(4321)
    pyr_gid_base = int(pyr_population.gid_begin)
    bas_gid_base = int(bas_population.gid_begin)
    olm_gid_base = int(olm_population.gid_begin)

    pyr_to_bas_nmda = [conn_rng.sample(range(PYR_COUNT), 100) for _ in range(BAS_COUNT)]
    pyr_to_olm_nmda = [conn_rng.sample(range(PYR_COUNT), 10) for _ in range(OLM_COUNT)]
    pyr_to_pyr_nmda = [
        [pre for pre in conn_rng.sample(range(PYR_COUNT), 25) if pre != post]
        for post in range(PYR_COUNT)
    ]
    pyr_to_bas_ampa = [conn_rng.sample(range(PYR_COUNT), 100) for _ in range(BAS_COUNT)]
    pyr_to_olm_ampa = [conn_rng.sample(range(PYR_COUNT), 10) for _ in range(OLM_COUNT)]
    pyr_to_pyr_ampa = [
        [pre for pre in conn_rng.sample(range(PYR_COUNT), 25) if pre != post]
        for post in range(PYR_COUNT)
    ]
    bas_to_bas_gaba = [
        [pre for pre in conn_rng.sample(range(BAS_COUNT), 60) if pre != post]
        for post in range(BAS_COUNT)
    ]
    bas_to_pyr_gaba = [conn_rng.sample(range(BAS_COUNT), 50) for _ in range(PYR_COUNT)]
    bas_to_olm_gaba = [conn_rng.sample(range(BAS_COUNT), 17) for _ in range(OLM_COUNT)]
    olm_to_pyr_gaba = [conn_rng.sample(range(OLM_COUNT), 10) for _ in range(PYR_COUNT)]

    def insert_nmda_target(cell, section: str, loc: float):
        return cell.group(section)[0](loc).insert(
            "MyExp2SynNMDABB",
            tau1=0.05,
            tau2=5.3,
            tau1NMDA=15.0,
            tau2NMDA=150.0,
            r=1.0,
            e=0.0,
        )

    def insert_exp2_target(cell, section: str, loc: float, tau1: float, tau2: float, e: float):
        return cell.group(section)[0](loc).insert("MyExp2SynBB", tau1=tau1, tau2=tau2, e=e)

    def connect_sids(sids, target, weight: float) -> None:
        for sid in sids:
            micro.network().sid_connect(int(sid), target, float(weight), 2.0)

    for cell in pyr_population:
        pyr_local_post = int(cell.gid) - pyr_gid_base
        target = insert_nmda_target(cell, "Bdend", 1.0)
        connect_sids((pyr_gid_base + int(pyr_local_pre) for pyr_local_pre in pyr_to_pyr_nmda[pyr_local_post]), target, 0.004e-3)
        target = insert_exp2_target(cell, "Bdend", 1.0, 0.05, 5.3, 0.0)
        connect_sids(
            (pyr_gid_base + int(pyr_local_pre) for pyr_local_pre in pyr_to_pyr_ampa[pyr_local_post]),
            target,
            0.5 * 0.04e-3,
        )
        target = insert_exp2_target(cell, "soma", 0.5, 0.07, 9.1, -80.0)
        connect_sids((bas_gid_base + int(bas_local) for bas_local in bas_to_pyr_gaba[pyr_local_post]), target, 4.0 * 0.18e-3)
        target = insert_exp2_target(cell, "Adend2", 0.5, 0.2, 20.0, -80.0)
        connect_sids(
            (olm_gid_base + int(olm_local) for olm_local in olm_to_pyr_gaba[pyr_local_post]),
            target,
            0.08 * 4.0 * 3.0 * 6.0e-3,
        )

    for cell in bas_population:
        bas_local_post = int(cell.gid) - bas_gid_base
        target = insert_nmda_target(cell, "soma", 0.5)
        connect_sids((pyr_gid_base + int(pyr_local) for pyr_local in pyr_to_bas_nmda[bas_local_post]), target, 1.15 * 1.2e-3)
        target = insert_exp2_target(cell, "soma", 0.5, 0.05, 5.3, 0.0)
        connect_sids((pyr_gid_base + int(pyr_local) for pyr_local in pyr_to_bas_ampa[bas_local_post]), target, 0.3 * 1.2e-3)
        target = insert_exp2_target(cell, "soma", 0.5, 0.07, 9.1, -80.0)
        connect_sids((bas_gid_base + int(bas_local_pre) for bas_local_pre in bas_to_bas_gaba[bas_local_post]), target, 3.0 * 1.5e-3)

    for cell in olm_population:
        olm_local_post = int(cell.gid) - olm_gid_base
        target = insert_nmda_target(cell, "soma", 0.5)
        connect_sids((pyr_gid_base + int(pyr_local) for pyr_local in pyr_to_olm_nmda[olm_local_post]), target, 0.7e-3)
        target = insert_exp2_target(cell, "soma", 0.5, 0.05, 5.3, 0.0)
        connect_sids((pyr_gid_base + int(pyr_local) for pyr_local in pyr_to_olm_ampa[olm_local_post]), target, 0.3 * 1.2e-3)
        target = insert_exp2_target(cell, "soma", 0.5, 0.07, 9.1, -80.0)
        connect_sids((bas_gid_base + int(bas_local) for bas_local in bas_to_olm_gaba[olm_local_post]), target, 0.05 * 4.0 * 0.18e-3)

    return micro, pyr_population[0], bas_population[0], olm_population[0]


def run_mind(args: argparse.Namespace) -> None:
    add_mind_paths()
    import mind_sim as ms

    pre_start = time.perf_counter()
    micro, pyr0, bas0, olm0 = build_mind_ca3_network(ms, threads=int(args.threads))
    micro.build_microcircuit()
    pyr_soma_trace = ms.Vector().record(pyr0.group("soma")[0](0.5)._ref_v)
    bas_soma_trace = ms.Vector().record(bas0.group("soma")[0](0.5)._ref_v)
    olm_soma_trace = ms.Vector().record(olm0.group("soma")[0](0.5)._ref_v)
    time_trace = ms.Vector().record(micro._ref_t)
    micro.finitialize(-65.0)
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    micro.run(float(args.duration_ms))
    run_s = time.perf_counter() - run_start

    time_ms = np.asarray(time_trace.to_python(), dtype=float)
    pyr_soma = np.asarray(pyr_soma_trace.to_python(), dtype=float)
    bas_soma = np.asarray(bas_soma_trace.to_python(), dtype=float)
    olm_soma = np.asarray(olm_soma_trace.to_python(), dtype=float)
    save_micro_output(
        args.output,
        backend="mind_sim_micro_only",
        time_ms=time_ms,
        traces=np.column_stack((pyr_soma, bas_soma, olm_soma)),
        labels=["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"],
        metadata={
            "macro2micro": False,
            "full_micro_network": True,
            "connections": True,
            "micro_threads": int(args.threads),
            "duration_ms": float(args.duration_ms),
            "pre_run_s": pre_run_s,
            "run_s": run_s,
            "total_s": pre_run_s + run_s,
        },
    )
    print(f"output={args.output}")
    print("backend=mind_sim_micro_only")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


def add_tvb_script_path() -> None:
    path = str(Path(__file__).resolve().parent / "netpyne_tvb_multiscale")
    if path not in sys.path:
        sys.path.insert(0, path)


def prepare_neuron_mod_library(workdir: Path) -> Path:
    from neuron import load_mechanisms

    source_dir = Path(__file__).resolve().parent / "netpyne_tvb_multiscale" / "mod"
    build_dir = workdir / "neuron_mod"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted([*source_dir.glob("*.mod"), *source_dir.glob("*.inc")]):
        shutil.copy2(source, build_dir / source.name)
    subprocess.run(["nrnivmodl", "."], cwd=build_dir, check=True)
    load_mechanisms(str(build_dir))
    return build_dir


def run_netpyne(args: argparse.Namespace) -> None:
    add_tvb_script_path()
    from netpyne import sim
    import run_tvb_netpyne_ca3_cosim as tvbn

    backend = str(args.backend)
    coreneuron_enabled = backend == "coreneuron"
    compile_start = time.perf_counter()
    if coreneuron_enabled:
        mod_build_dir = tvbn.prepare_coreneuron_mod_library(args.workdir)
    else:
        mod_build_dir = prepare_neuron_mod_library(args.workdir)
    mod_compile_s = time.perf_counter() - compile_start
    netpyne_threads = tvbn.install_netpyne_threads(int(args.threads))

    net_params, sim_config = tvbn.make_netpyne_params(
        connections=True,
        wseed=4321,
        duration_ms=float(args.duration_ms),
        pyr_current_na=PYR_CURRENT_NA,
        olm_current_na=OLM_CURRENT_NA,
        record_voltage=True,
        coreneuron_enabled=coreneuron_enabled,
    )
    sim_config.timing = True

    pre_start = time.perf_counter()
    sim.create(netParams=net_params, simConfig=sim_config, output=False, clearAll=False)
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    sim.simulate()
    run_s = time.perf_counter() - run_start

    def vector_to_array(value) -> np.ndarray:
        if hasattr(value, "to_python"):
            return np.asarray(value.to_python(), dtype=float)
        return np.asarray(value, dtype=float)

    sim_data = getattr(sim, "allSimData", None) or sim.simData
    time_ms = vector_to_array(sim_data["t"])
    voltage_data = sim_data["voltage"]
    pyr_soma_voltage = vector_to_array(voltage_data["cell_0"])
    bas_soma_voltage = vector_to_array(voltage_data[f"cell_{PYR_COUNT}"])
    olm_soma_voltage = vector_to_array(voltage_data[f"cell_{PYR_COUNT + BAS_COUNT}"])
    save_micro_output(
        args.output,
        backend=f"netpyne_{backend}_micro_only",
        time_ms=time_ms,
        traces=np.column_stack((pyr_soma_voltage, bas_soma_voltage, olm_soma_voltage)),
        labels=["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"],
        metadata={
            "macro2micro": False,
            "full_micro_network": True,
            "connections": True,
            "coreneuron": coreneuron_enabled,
            "mod_build_dir": str(mod_build_dir),
            "mod_compile_s": mod_compile_s,
            "netpyne_threads": int(netpyne_threads),
            "duration_ms": float(args.duration_ms),
            "pre_run_s": pre_run_s,
            "run_s": run_s,
            "total_s": mod_compile_s + pre_run_s + run_s,
        },
    )
    print(f"output={args.output}")
    print(f"backend=netpyne_{backend}_micro_only")
    print(f"mod_compile_s={mod_compile_s:.6f}")
    print(f"netpyne_threads={netpyne_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


def save_micro_output(output: Path, *, backend: str, time_ms: np.ndarray, traces: np.ndarray, labels: list[str], metadata: dict) -> None:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    spike_indices = []
    spike_times = []
    for column in range(traces.shape[1]):
        indices, times = voltage_spikes(traces[:, column])
        spike_indices.append(indices)
        spike_times.append(times)
    metadata = dict(metadata)
    metadata.update(
        {
            "backend": backend,
            "dt_micro_ms": 0.025,
            "voltage_labels": labels,
            "spike_source": "voltage_threshold_rising_crossings",
            "spike_threshold_mv": SPIKE_THRESHOLD_MV,
        }
    )
    np.savez_compressed(
        output,
        time_ms=np.asarray(time_ms, dtype=float),
        voltage_labels=np.asarray(labels, dtype=object),
        voltage_traces=np.asarray(traces, dtype=float),
        spike_indices=np.asarray(spike_indices, dtype=object),
        spike_times_ms=np.asarray(spike_times, dtype=object),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def metric(a: np.ndarray, b: np.ndarray, time_a: np.ndarray, time_b: np.ndarray) -> dict:
    rounded_a = np.round(np.asarray(time_a, dtype=float), 9)
    rounded_b = np.round(np.asarray(time_b, dtype=float), 9)
    common, ia, ib = np.intersect1d(rounded_a, rounded_b, return_indices=True)
    aa = np.asarray(a, dtype=float)[ia]
    bb = np.asarray(b, dtype=float)[ib]
    diff = aa - bb
    finite = np.isfinite(diff)
    absdiff = np.where(finite, np.abs(diff), -1.0)
    max_index = np.unravel_index(int(np.argmax(absdiff)), absdiff.shape)
    return {
        "common_sample_count": int(common.size),
        "max_abs": float(absdiff[max_index]),
        "rms": float(np.sqrt(np.mean(diff[finite] * diff[finite]))),
        "time_ms": float(common[max_index[0]]),
        "index": [int(value) for value in max_index],
        "a_value": float(aa[max_index]),
        "b_value": float(bb[max_index]),
        "signed_diff": float(diff[max_index]),
    }


def compare(args: argparse.Namespace) -> None:
    mind = np.load(args.mind, allow_pickle=True)
    netpyne = np.load(args.netpyne, allow_pickle=True)
    mind_labels = [str(label) for label in mind["voltage_labels"].tolist()]
    netpyne_labels = [str(label) for label in netpyne["voltage_labels"].tolist()]
    report = {
        "mind": str(args.mind.resolve()),
        "netpyne": str(args.netpyne.resolve()),
        "labels_match": mind_labels == netpyne_labels,
        "labels": mind_labels,
        "voltage": {},
        "spikes": {},
        "metadata": {
            "mind": json.loads(str(mind["metadata_json"])),
            "netpyne": json.loads(str(netpyne["metadata_json"])),
        },
    }
    for index, label in enumerate(mind_labels):
        report["voltage"][label] = metric(
            mind["voltage_traces"][:, index],
            netpyne["voltage_traces"][:, index],
            mind["time_ms"],
            netpyne["time_ms"],
        )
        mind_indices, mind_times = voltage_spikes(mind["voltage_traces"][:, index])
        netpyne_indices, netpyne_times = voltage_spikes(netpyne["voltage_traces"][:, index])
        report["spikes"][label] = {
            "mind_count": int(mind_indices.size),
            "netpyne_count": int(netpyne_indices.size),
            "exact_index_equal": bool(np.array_equal(mind_indices, netpyne_indices)),
            "exact_time_equal": bool(np.array_equal(mind_times, netpyne_times)),
            "mind_indices": mind_indices.tolist(),
            "netpyne_indices": netpyne_indices.tolist(),
            "mind_times_ms": mind_times.tolist(),
            "netpyne_times_ms": netpyne_times.tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-network CA3 micro-only MIND vs NetPyNE diagnostics.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    mind = subparsers.add_parser("mind")
    mind.add_argument("--duration-ms", type=float, default=300.0)
    mind.add_argument("--threads", type=int, default=1)
    mind.add_argument("--output", type=Path, required=True)
    mind.set_defaults(func=run_mind)

    netpyne = subparsers.add_parser("netpyne")
    netpyne.add_argument("--duration-ms", type=float, default=300.0)
    netpyne.add_argument("--threads", type=int, default=1)
    netpyne.add_argument("--backend", choices=("neuron", "coreneuron"), default="neuron")
    netpyne.add_argument(
        "--workdir",
        type=Path,
        default=Path("examples/ca3_epilepsy_cosim/outputs/micro_only_netpyne_workdir"),
    )
    netpyne.add_argument("--output", type=Path, required=True)
    netpyne.set_defaults(func=run_netpyne)

    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("--mind", type=Path, required=True)
    cmp_parser.add_argument("--netpyne", type=Path, required=True)
    cmp_parser.add_argument("--output", type=Path, required=True)
    cmp_parser.set_defaults(func=compare)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
