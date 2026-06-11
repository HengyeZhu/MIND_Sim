#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200
SPIKE_THRESHOLD_MV = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="MIND Sim VEP neural-mass plus ModelDB 186768 CA3 micro cosimulation.")
    parser.add_argument(
        "--connectivity-csv",
        type=Path,
        required=True,
        help="Labelled matrix CSV with weights and delays_ms sections.",
    )
    parser.add_argument("--duration-ms", "--t-ms", dest="duration_ms", type=float, default=20.0)
    parser.add_argument("--micro-threads", type=int, default=4)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/vep_ca3_mindsim_cosim.npz"),
    )
    args = parser.parse_args()

    output = args.output

    repo_root = Path(__file__).resolve().parents[3]
    build_path = repo_root / "build"
    source_python_path = repo_root / "src" / "python_api"
    if any((build_path / "mind_sim").glob("_native*.so")):
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if finder.__class__.__module__ != "_mind_sim_editable"
        ]
    for path in (source_python_path, build_path):
        path_text = str(path)
        if path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)

    import mind_sim as ms

    ms.macro.load_mech(Path(__file__).resolve().parent / "mod")
    ms.macro.dt(0.1)
    ms.macro.exchange_window(0.5)
    rois = ms.macro.load_rois(args.connectivity_csv)
    left_ca3_roi = rois.roi("Left-CA3")

    pre_start = time.perf_counter()
    # Micro model
    micro = ms.Sim()
    micro.set_device("cpu")
    micro.set_num_threads(args.micro_threads)
    micro.set_dt(0.025)
    micro.load_mech(str(Path(__file__).resolve().parent / "mod"))

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
    pyr_sections = [pyr_soma, pyr_bdend, pyr_adend1, pyr_adend2, pyr_adend3]

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
            {"name": "PYR", "num_cells": PYR_COUNT, "sections": pyr_sections},
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
        cell.group("soma")[0](0.5).insert("IClamp", **{"del": 0.2, "dur": 1.0e9, "amp": 0.1})
        soma = cell.group("soma")[0](0.5)
        sid = int(cell.gid)
        micro.network().register_spike_source(sid, soma._ref_v, SPIKE_THRESHOLD_MV)
    for cell in bas_population:
        cell.v_init = -65.0
        soma = cell.group("soma")
        soma.Ra = 35.4
        soma.cm = 1.0
        soma.insert("pas", e=-65.0, g=0.1e-3)
        soma.insert("Nafbwb")
        soma.insert("Kdrbwb")
        sid = int(cell.gid)
        micro.network().register_spike_source(sid, soma[0](0.5)._ref_v, SPIKE_THRESHOLD_MV)
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
        soma[0](0.5).insert("IClamp", **{"del": 0.2, "dur": 1.0e9, "amp": -25e-3})
        sid = int(cell.gid)
        micro.network().register_spike_source(sid, soma[0](0.5)._ref_v, SPIKE_THRESHOLD_MV)
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
    initial_x = np.zeros(len(rois.labels))
    initial_z = np.zeros(len(rois.labels))
    for roi_index, roi in enumerate(rois.rois()):
        if roi.label == left_ca3_roi.label:
            x0 = -1.6
        elif roi.label in propagation_labels:
            x0 = -1.9
        else:
            x0 = -2.4
        x_initial = x0 + 0.02 * macro_rng.standard_normal()
        initial_state = {"x": x_initial, "z": 0.0}
        if roi.label != left_ca3_roi.label:
            initial_x[roi_index] = x_initial
            roi.use_macro(
                "tvb_epileptor2d",
                initial_state=initial_state,
                params={
                    "x0": x0,
                    "a": 1.0,
                    "b": 3.0,
                    "c": 1.0,
                    "d": 5.0,
                    "r": 0.00035,
                    "slope": 0.0,
                    "kvf": 0.35,
                    "ks": 0.0,
                    "tt": 1.0,
                    "i_ext": 3.1,
                    "modification": 0.0,
                },
            )
        else:
            initial_x[roi_index] = 0.0

    left_ca3_roi.use_micro()
    ca3_input_to_spikes_params = {
        "base_hz": 1.0,
        "gain_hz": 45.0,
        "max_rate_hz": 120.0,
        "threshold": -0.35,
        "slope": 4.0,
    }
    # Micro recurrent connections
    conn_rng = random.Random(4321)

    for cell in bas_population:
        target = cell.group("soma")[0](0.5).insert(
            "MyExp2SynNMDABB",
            tau1=0.05,
            tau2=5.3,
            tau1NMDA=15.0,
            tau2NMDA=150.0,
            r=1.0,
            e=0.0,
        )
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 100):
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local), target, 1.15 * 1.2e-3, 2.0)

    for cell in olm_population:
        target = cell.group("soma")[0](0.5).insert(
            "MyExp2SynNMDABB",
            tau1=0.05,
            tau2=5.3,
            tau1NMDA=15.0,
            tau2NMDA=150.0,
            r=1.0,
            e=0.0,
        )
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 10):
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local), target, 0.7e-3, 2.0)

    for cell in pyr_population:
        pyr_local_post = int(cell.gid) - int(pyr_population.gid_begin)
        target = cell.group("Bdend")[0](1.0).insert(
            "MyExp2SynNMDABB",
            tau1=0.05,
            tau2=5.3,
            tau1NMDA=15.0,
            tau2NMDA=150.0,
            r=1.0,
            e=0.0,
        )
        for pyr_local_pre in conn_rng.sample(range(PYR_COUNT), 25):
            if pyr_local_pre == pyr_local_post:
                continue
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local_pre), target, 0.004e-3, 2.0)

    for cell in bas_population:
        target = cell.group("soma")[0](0.5).insert("MyExp2SynBB", tau1=0.05, tau2=5.3, e=0.0)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 100):
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local), target, 0.3 * 1.2e-3, 2.0)

    for cell in olm_population:
        target = cell.group("soma")[0](0.5).insert("MyExp2SynBB", tau1=0.05, tau2=5.3, e=0.0)
        for pyr_local in conn_rng.sample(range(PYR_COUNT), 10):
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local), target, 0.3 * 1.2e-3, 2.0)

    for cell in pyr_population:
        pyr_local_post = int(cell.gid) - int(pyr_population.gid_begin)
        target = cell.group("Bdend")[0](1.0).insert("MyExp2SynBB", tau1=0.05, tau2=5.3, e=0.0)
        for pyr_local_pre in conn_rng.sample(range(PYR_COUNT), 25):
            if pyr_local_pre == pyr_local_post:
                continue
            micro.network().sid_connect(int(pyr_population.gid_begin) + int(pyr_local_pre), target, 0.5 * 0.04e-3, 2.0)

    for cell in bas_population:
        bas_local_post = int(cell.gid) - int(bas_population.gid_begin)
        target = cell.group("soma")[0](0.5).insert("MyExp2SynBB", tau1=0.07, tau2=9.1, e=-80.0)
        for bas_local_pre in conn_rng.sample(range(BAS_COUNT), 60):
            if bas_local_pre == bas_local_post:
                continue
            micro.network().sid_connect(int(bas_population.gid_begin) + int(bas_local_pre), target, 3.0 * 1.5e-3, 2.0)

    for cell in pyr_population:
        target = cell.group("soma")[0](0.5).insert("MyExp2SynBB", tau1=0.07, tau2=9.1, e=-80.0)
        for bas_local in conn_rng.sample(range(BAS_COUNT), 50):
            micro.network().sid_connect(int(bas_population.gid_begin) + int(bas_local), target, 4.0 * 0.18e-3, 2.0)

    for cell in olm_population:
        target = cell.group("soma")[0](0.5).insert("MyExp2SynBB", tau1=0.07, tau2=9.1, e=-80.0)
        for bas_local in conn_rng.sample(range(BAS_COUNT), 17):
            micro.network().sid_connect(int(bas_population.gid_begin) + int(bas_local), target, 0.05 * 4.0 * 0.18e-3, 2.0)

    for cell in pyr_population:
        target = cell.group("Adend2")[0](0.5).insert("MyExp2SynBB", tau1=0.2, tau2=20.0, e=-80.0)
        for olm_local in conn_rng.sample(range(OLM_COUNT), 10):
            micro.network().sid_connect(int(olm_population.gid_begin) + int(olm_local), target, 0.08 * 4.0 * 3.0 * 6.0e-3, 2.0)
    for cell in pyr_population:
        target = cell.group("Adend3")[0](0.5).insert("MyExp2SynBB", tau1=0.05, tau2=5.3, e=0.0)
        left_ca3_roi.macro2micro(
            "ca3_input_to_spikes",
            target=target,
            weight=0.02e-3 * 1.0e-2,
            delay=0.2,
            params=ca3_input_to_spikes_params,
        )
    for target_index, target in enumerate(rois.rois()):
        for source_index, source_label in enumerate(rois.labels):
            if rois.weights[target_index][source_index] == 0.0:
                continue
            target.insert(
                source_label,
                "ca3_input_macro2macro" if target.label == left_ca3_roi.label else "vep_x_macro2macro",
            )
    ca3_pyr_spikes_to_vep_params = {
        "tau_ms": 50.0,
        "x_baseline": -1.8,
        "gain": 2.0,
        "population_size": 800.0,
    }
    ca3_bas_spikes_to_vep_params = {
        "tau_ms": 20.0,
        "gain": -0.7,
        "population_size": 200.0,
    }
    ca3_olm_spikes_to_vep_params = {
        "tau_ms": 80.0,
        "gain": -0.4,
        "population_size": 200.0,
    }
    for cell in pyr_population:
        left_ca3_roi.micro2macro(
            "ca3_pyr_spikes_to_vep",
            sid=int(cell.gid),
            params=ca3_pyr_spikes_to_vep_params,
        )
    for cell in bas_population:
        left_ca3_roi.micro2macro(
            "ca3_bas_spikes_to_vep",
            sid=int(cell.gid),
            params=ca3_bas_spikes_to_vep_params,
        )
    for cell in olm_population:
        left_ca3_roi.micro2macro(
            "ca3_olm_spikes_to_vep",
            sid=int(cell.gid),
            params=ca3_olm_spikes_to_vep_params,
        )

    history_steps = round(np.max(rois.delays) / 0.1) + 1
    history_alpha = np.linspace(-1.0, 0.0, history_steps)[:, np.newaxis]
    roi_phase = np.linspace(0.0, 2.0 * np.pi, len(rois.labels), endpoint=False)[np.newaxis, :]
    macro_initial_history = np.empty((history_steps, 2, len(rois.labels)))
    macro_initial_history[:, 0] = initial_x + 0.01 * history_alpha * np.sin(roi_phase)
    macro_initial_history[:, 1] = initial_z + 0.002 * history_alpha * np.cos(roi_phase)
    macro_initial_history[-1, 0] = initial_x
    macro_initial_history[-1, 1] = initial_z
    rois.initial_history(macro_initial_history, outputs=["x", "z"])

    micro.build_microcircuit()

    for roi in rois.rois():
        roi.record("x")
        roi.record("z")
    pyr_voltage_trace = ms.Vector().record(pyr_population[0].group("soma")[0](0.5)._ref_v)
    bas_voltage_trace = ms.Vector().record(bas_population[0].group("soma")[0](0.5)._ref_v)
    olm_voltage_trace = ms.Vector().record(olm_population[0].group("soma")[0](0.5)._ref_v)
    adend3_voltage_trace = ms.Vector().record(pyr_population[0].group("Adend3")[0](0.5)._ref_v)
    voltage_time_trace = ms.Vector().record(micro._ref_t)
    micro.finitialize(-65.0)
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    simulator = ms.Simulator(rois)
    result = simulator.run(args.duration_ms)
    run_s = time.perf_counter() - run_start

    times_ms = np.asarray(result.times)
    recorded_macro = result.records
    cube = np.asarray(recorded_macro.values).reshape(
        recorded_macro.sample_count,
        recorded_macro.recorded_roi_count,
        recorded_macro.output_count,
    )
    x = cube[:, :, 0]
    z = cube[:, :, 1].copy()
    left_ca3_column = rois.labels.index(left_ca3_roi.label)
    z[:, left_ca3_column] = np.nan
    voltage_time = np.asarray(voltage_time_trace.to_python())
    pyr_voltage = np.asarray(pyr_voltage_trace.to_python())
    bas_voltage = np.asarray(bas_voltage_trace.to_python())
    olm_voltage = np.asarray(olm_voltage_trace.to_python())
    voltage_traces = np.empty((voltage_time.size, 3))
    voltage_traces[:, 0] = pyr_voltage
    voltage_traces[:, 1] = bas_voltage
    voltage_traces[:, 2] = olm_voltage
    voltage = pyr_voltage.copy()
    adend3_voltage = np.asarray(adend3_voltage_trace.to_python())
    voltage_labels = ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]

    metadata = {
        "source": "MIND Sim TVB Epileptor2D-equivalent neural mass + MIND Sim API rewrite of ModelDB 186768 CA3",
        "left_ca3_label": left_ca3_roi.label,
        "connectivity_csv": str(args.connectivity_csv),
        "connectivity_format": "matrix_csv_v1",
        "ca3_micro_model": "ModelDB 186768 CA3, MIND Sim API/CoreNEURON rewrite",
        "macro_model": "TVB built-in Epileptor2D equations implemented as a MOD mechanism",
        "micro_backend": "CoreNEURON",
        "duration_ms": args.duration_ms,
        "dt_micro_ms": 0.025,
        "dt_macro_ms": 0.1,
        "micro_num_threads": args.micro_threads,
        "min_positive_delay_ms": rois.min_positive_delay(),
        "exchange_window_ms": 0.5,
        "macro_i_ext": 3.1,
        "pyr_current_na": 0.1,
        "olm_current_na": -25e-3,
        "epileptor2d_kvf": 0.35,
        "epileptor2d_ks": 0.0,
        "epileptor2d_r": 0.00035,
        "epileptor2d_tt": 1.0,
        "epileptor2d_modification": False,
        "drive_weight": 0.02e-3,
        "effective_drive_weight": 0.02e-3 * 1.0e-2,
        "drive_delay_ms": 0.2,
        "connections": True,
        "initial_history": "explicit non-constant TVB-style chronological history with axes time,output,roi and outputs ['x', 'z']; history[-1] is the t=0 state",
        "notes": "Left-CA3 ROI is replaced by population-specific PYR/BAS/OLM event-driven micro x output; macro input to Left-CA3 is transformed into external AMPA events on PYR Adend3 synapses. Left-CA3 z is not a transform output.",
        "voltage_recording": "representative PYR/BAS/OLM soma voltages in voltage_traces; fixed output key voltage remains PYR[0].soma; PYR[0].Adend3(0.5) voltage is in adend3_voltage",
        "voltage_trace_labels": voltage_labels,
        "spike_validation": "derive representative PYR/BAS/OLM soma spikes from recorded voltage threshold crossings; no spike array is exported",
        "record_names": ["x", "z"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        labels=rois.labels,
        weights=rois.weights,
        delays=rois.delays,
        record_names=["x", "z"],
        time_ms=times_ms,
        macro_records=cube,
        macro_x=x,
        macro_z=z,
        voltage_time=voltage_time,
        voltage=voltage,
        voltage_trace_time=voltage_time,
        voltage_labels=voltage_labels,
        voltage_traces=voltage_traces,
        adend3_voltage=adend3_voltage,
        left_ca3_macro_output_x=x[:, left_ca3_column],
        timing_s=[pre_run_s, run_s, pre_run_s + run_s],
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"output={output}")
    print("backend=mind_sim")
    print("device=cpu")
    print(f"num_threads={args.micro_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


if __name__ == "__main__":
    main()
