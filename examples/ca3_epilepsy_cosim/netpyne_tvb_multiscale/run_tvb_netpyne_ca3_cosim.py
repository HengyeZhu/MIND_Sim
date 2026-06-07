#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan

PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200


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


def make_initial_conditions(labels: list[str], micro_index: int, history_steps: int = 1) -> np.ndarray:
    rng = np.random.default_rng(1234)
    micro_label = labels[int(micro_index)]
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
    x0 = []
    for label in labels:
        if label == micro_label:
            x0.append(0.0)
        elif label in propagation_labels:
            x0.append(-1.9)
        else:
            x0.append(-2.4)
    x0 = np.asarray(x0, dtype=float)
    initial = np.zeros((1, 2, len(labels), 1), dtype=float)
    initial[0, 0, :, 0] = x0 + 0.02 * rng.standard_normal(len(labels))
    initial[0, 0, int(micro_index), 0] = 0.0
    initial[0, 1, :, 0] = 0.0
    return np.repeat(initial, max(1, int(history_steps)), axis=0)


def tvb_history_steps(weights: np.ndarray, delays_ms: np.ndarray, dt_ms: float) -> int:
    delays_ms = np.asarray(delays_ms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0.0):
        return 1
    max_delay_ms = float(np.max(delays_ms[weights > 0.0]))
    return int(np.ceil(max_delay_ms / float(dt_ms))) + 1


def make_tvb_connectivity(labels: list[str], weights: np.ndarray, delays_ms: np.ndarray, speed_mm_per_ms: float):
    from tvb.datatypes.connectivity import Connectivity

    label_array = np.asarray(labels, dtype="<U128")
    weights = np.asarray(weights, dtype=float)
    delays_ms = np.asarray(delays_ms, dtype=float)
    centres = np.zeros((len(labels), 3), dtype=float)
    centres[:, 0] = np.arange(len(labels), dtype=float)

    conn = Connectivity()
    conn.weights = weights.copy()
    conn.tract_lengths = np.where(weights > 0.0, delays_ms * float(speed_mm_per_ms), 0.0)
    conn.speed = np.asarray([float(speed_mm_per_ms)], dtype=float)
    conn.region_labels = label_array
    conn.centres = centres
    conn.orientations = np.zeros((len(labels), 3), dtype=float)
    conn.areas = np.ones(len(labels), dtype=float)
    conn.cortical = np.asarray([label.startswith("ctx-") for label in labels], dtype=bool)
    conn.hemispheres = np.asarray([label.startswith("Right") or label.startswith("ctx-rh") for label in labels], dtype=bool)
    conn.configure()
    return conn


def ca3_bridge_x_trace(
    time_ms: np.ndarray,
    initial_x: float,
    spike_times_ms: np.ndarray,
    spike_gids: np.ndarray,
    *,
    tau_pyr_ms: float = 50.0,
    tau_bas_ms: float = 20.0,
    tau_olm_ms: float = 80.0,
    x_baseline: float = -1.8,
    pyr_gain: float = 2.0,
    bas_gain: float = -0.7,
    olm_gain: float = -0.4,
) -> np.ndarray:
    time_ms = np.asarray(time_ms, dtype=float)
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    spike_gids = np.asarray(spike_gids, dtype=int)
    out = np.full(time_ms.shape, float(x_baseline), dtype=float)
    if time_ms.size == 0:
        return out
    out[time_ms <= 0.0] = float(initial_x)

    order = np.argsort(spike_times_ms, kind="stable")
    spike_times_ms = spike_times_ms[order]
    spike_gids = spike_gids[order]
    pyr_activity = 0.0
    bas_activity = 0.0
    olm_activity = 0.0
    current_time = 0.0
    spike_index = 0

    def advance(next_time: float) -> None:
        nonlocal pyr_activity, bas_activity, olm_activity, current_time
        dt = max(0.0, float(next_time) - current_time)
        pyr_activity *= float(np.exp(-dt / float(tau_pyr_ms)))
        bas_activity *= float(np.exp(-dt / float(tau_bas_ms)))
        olm_activity *= float(np.exp(-dt / float(tau_olm_ms)))
        current_time = float(next_time)

    for sample, sample_time in enumerate(time_ms):
        if sample_time <= 0.0:
            continue
        while spike_index < spike_times_ms.size and spike_times_ms[spike_index] <= sample_time:
            event_time = float(spike_times_ms[spike_index])
            if event_time > current_time:
                advance(event_time)
            gid = int(spike_gids[spike_index])
            if 0 <= gid < PYR_COUNT:
                pyr_activity += 1.0 / float(PYR_COUNT)
            elif PYR_COUNT <= gid < PYR_COUNT + BAS_COUNT:
                bas_activity += 1.0 / float(BAS_COUNT)
            elif PYR_COUNT + BAS_COUNT <= gid < PYR_COUNT + BAS_COUNT + OLM_COUNT:
                olm_activity += 1.0 / float(OLM_COUNT)
            spike_index += 1
        advance(float(sample_time))
        out[sample] = (
            float(x_baseline)
            + float(pyr_gain) * pyr_activity
            + float(bas_gain) * bas_activity
            + float(olm_gain) * olm_activity
        )
    return out


def neuron_vector_to_array(vector) -> np.ndarray:
    try:
        return np.asarray(vector.as_numpy(), dtype=float).copy()
    except Exception:
        return np.asarray(list(vector), dtype=float)


def install_direct_voltage_recorder(record_step_ms: float, metadata: dict) -> dict[str, object]:
    from neuron import h
    from netpyne import sim as netpyne_sim

    net = getattr(netpyne_sim, "net", None)
    cells = list(getattr(net, "cells", []) or [])
    if not cells:
        raise RuntimeError("NetPyNE cells are not available for direct voltage recording")

    cell = next((item for item in cells if getattr(item, "tags", {}).get("pop") == "PYR"), cells[0])
    soma = getattr(cell, "secs", {}).get("soma", {}).get("hObj")
    if soma is None:
        raise RuntimeError("first recorded NetPyNE cell has no soma hObj")

    voltage = h.Vector()
    voltage.record(soma(0.5)._ref_v, float(record_step_ms), sec=soma)
    metadata["direct_voltage_recording_gid"] = int(getattr(cell, "gid", 0))
    metadata["direct_voltage_record_step_ms"] = float(record_step_ms)
    return {"record_step_ms": float(record_step_ms), "voltage": voltage}


def extract_recorded_voltage(
    orchestrator,
    metadata: dict,
    direct_voltage_recorder: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if direct_voltage_recorder is not None:
        voltage = neuron_vector_to_array(direct_voltage_recorder["voltage"])
        if voltage.size:
            time_ms = np.arange(voltage.size, dtype=float) * float(direct_voltage_recorder["record_step_ms"])
            metadata["voltage_source"] = "direct_neuron_vector"
            return time_ms, voltage

    spiking = orchestrator.spikeNet_app.spiking_cosimulator
    netpyne_instance = getattr(spiking, "netpyne_instance", None)
    all_sim_data = getattr(netpyne_instance, "allSimData", None)
    if all_sim_data is None:
        all_sim_data = getattr(spiking, "allSimData", None)
    if all_sim_data is None:
        try:
            from netpyne import sim as netpyne_sim

            all_sim_data = getattr(netpyne_sim, "allSimData", None)
            if all_sim_data is None:
                all_sim_data = getattr(netpyne_sim, "simData", None)
        except Exception as exc:  # pragma: no cover - diagnostic path
            metadata["voltage_global_read_error"] = str(exc)
    if all_sim_data is None:
        metadata["voltage_read_error"] = "NetPyNE allSimData is not available"
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    trace = all_sim_data.get("voltage") if hasattr(all_sim_data, "get") else None
    if trace is None:
        metadata["voltage_read_error"] = "NetPyNE voltage trace is not available"
        if hasattr(all_sim_data, "keys"):
            metadata["voltage_available_keys"] = [str(key) for key in all_sim_data.keys()]
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if isinstance(trace, dict):
        if not trace:
            metadata["voltage_read_error"] = "NetPyNE voltage trace is empty"
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        voltage = np.asarray(next(iter(trace.values())), dtype=float)
    else:
        voltage = np.asarray(trace, dtype=float)

    if hasattr(all_sim_data, "get") and all_sim_data.get("t") is not None:
        time_ms = np.asarray(all_sim_data.get("t"), dtype=float)
    else:
        time_ms = np.arange(voltage.size, dtype=float) * 0.1
    if time_ms.size != voltage.size:
        sample_count = min(time_ms.size, voltage.size)
        metadata["voltage_trimmed_samples"] = {"time": int(time_ms.size), "voltage": int(voltage.size)}
        time_ms = time_ms[:sample_count]
        voltage = voltage[:sample_count]
    return time_ms, voltage


def add_pt3d_section(name: str, label: str, points: list[tuple[float, float, float, float]]) -> dict:
    first = points[0]
    last = points[-1]
    dx = float(last[0] - first[0])
    dy = float(last[1] - first[1])
    dz = float(last[2] - first[2])
    length = max(float((dx * dx + dy * dy + dz * dz) ** 0.5), 1e-9)
    diam = float(np.mean([point[3] for point in points]))
    return {"geom": {"L": length, "diam": diam, "nseg": 1}, "mechs": {}, "label": label}


def pyrcell_rule(pyr_current_na: float) -> dict:
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
                "pointps": {"inj": {"mod": "IClamp", "loc": 0.5, "del": 0.2, "dur": 1.0e9, "amp": float(pyr_current_na)}},
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


def interneuron_rule(cell_type: str, olm: bool, olm_current_na: float) -> dict:
    total_area_um2 = 10000.0
    diam = float(total_area_um2**0.5)
    length = float(diam / np.pi)
    mechs = {
        "pas": {"e": -65.0, "g": 0.1e-3},
        "Nafbwb": {},
        "Kdrbwb": {},
    }
    if olm:
        mechs.update({"Iholmw": {}, "Caolmw": {}, "ICaolmw": {}, "KCaolmw": {}})
    pointps = {}
    if olm:
        pointps["inj"] = {"mod": "IClamp", "loc": 0.5, "del": 0.2, "dur": 1.0e9, "amp": float(olm_current_na)}
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


def install_ca3_stimulus_mapping() -> None:
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    current = NetpyneModule.connectStimuli
    if getattr(current, "_mind_ca3_stimulus_mapping", False):
        return
    original_connect_stimuli = current

    def connect_stimuli(self, sourcePop, targetPop, weight, delay, receptorType, prob=None):
        if targetPop == "PYR" and receptorType in SYNAPSE_LOCS:
            source_cells = int(self.netParams.popParams[sourcePop]["numCells"])
            target_cells = int(self.netParams.popParams[targetPop]["numCells"])
            sec, loc, _kind = SYNAPSE_LOCS[receptorType]
            params = {
                "preConds": {"pop": sourcePop},
                "postConds": {"pop": targetPop},
                "weight": weight,
                "delay": delay,
                "synMech": receptorType,
                "sec": sec,
                "loc": loc,
            }
            if source_cells == target_cells:
                params["connList"] = [[int(i), int(i)] for i in range(source_cells)]
            elif prob:
                params["probability"] = prob
            elif source_cells <= target_cells:
                params["divergence"] = 1.0
            else:
                params["convergence"] = 1.0
            self.netParams.connParams[sourcePop + "->" + targetPop] = params
            return
        return original_connect_stimuli(self, sourcePop, targetPop, weight, delay, receptorType, prob)

    connect_stimuli._mind_ca3_stimulus_mapping = True
    NetpyneModule.connectStimuli = connect_stimuli


def make_netpyne_params(
    *,
    connections: bool,
    wseed: int,
    duration_ms: float,
    pyr_current_na: float,
    olm_current_na: float,
):
    from netpyne import specs

    net_params = specs.NetParams()
    net_params.defaultThreshold = 0.0
    net_params.scale = 1.0
    net_params.sizeX = 1.0
    net_params.sizeY = 1.0
    net_params.sizeZ = 1.0
    net_params.cellParams["PYR"] = pyrcell_rule(pyr_current_na)
    net_params.cellParams["BAS"] = interneuron_rule("BAS", olm=False, olm_current_na=olm_current_na)
    net_params.cellParams["OLM"] = interneuron_rule("OLM", olm=True, olm_current_na=olm_current_na)
    net_params.popParams["PYR"] = {"cellType": "PYR", "numCells": PYR_COUNT}
    net_params.popParams["BAS"] = {"cellType": "BAS", "numCells": BAS_COUNT}
    net_params.popParams["OLM"] = {"cellType": "OLM", "numCells": OLM_COUNT}
    for syn_name, (_sec, _loc, kind) in SYNAPSE_LOCS.items():
        net_params.synMechParams[syn_name] = synapse_params(kind)
    if connections:
        random.seed(int(wseed))

        net_params.connParams["PYR->BAS:somaNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(pyr_local), int(bas_local)]
                for bas_local in range(BAS_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 100)
            ],
            "weight": 1.15 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaNMDA",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->OLM:somaNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(pyr_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 10)
            ],
            "weight": 0.7e-3,
            "delay": 2.0,
            "synMech": "somaNMDA",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->PYR:BdendNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(pyr_local_pre), int(pyr_local_post)]
                for pyr_local_post in range(PYR_COUNT)
                for pyr_local_pre in random.sample(range(PYR_COUNT), 25)
            ],
            "weight": 0.004e-3,
            "delay": 2.0,
            "synMech": "BdendNMDA",
            "sec": "Bdend",
            "loc": 1.0,
        }

        net_params.connParams["PYR->BAS:somaAMPAf"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(pyr_local), int(bas_local)]
                for bas_local in range(BAS_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 100)
            ],
            "weight": 0.3 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaAMPAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->OLM:somaAMPAf"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(pyr_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 10)
            ],
            "weight": 0.3 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaAMPAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->PYR:BdendAMPA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(pyr_local_pre), int(pyr_local_post)]
                for pyr_local_post in range(PYR_COUNT)
                for pyr_local_pre in random.sample(range(PYR_COUNT), 25)
            ],
            "weight": 0.5 * 0.04e-3,
            "delay": 2.0,
            "synMech": "BdendAMPA",
            "sec": "Bdend",
            "loc": 1.0,
        }

        net_params.connParams["BAS->BAS:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(bas_local_pre), int(bas_local_post)]
                for bas_local_post in range(BAS_COUNT)
                for bas_local_pre in random.sample(range(BAS_COUNT), 60)
            ],
            "weight": 3.0 * 1.5e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["BAS->PYR:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(bas_local), int(pyr_local)]
                for pyr_local in range(PYR_COUNT)
                for bas_local in random.sample(range(BAS_COUNT), 50)
            ],
            "weight": 4.0 * 0.18e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["BAS->OLM:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(bas_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for bas_local in random.sample(range(BAS_COUNT), 17)
            ],
            "weight": 0.05 * 4.0 * 0.18e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["OLM->PYR:Adend2GABAs"] = {
            "preConds": {"pop": "OLM"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(olm_local), int(pyr_local)]
                for pyr_local in range(PYR_COUNT)
                for olm_local in random.sample(range(OLM_COUNT), 10)
            ],
            "weight": 0.08 * 4.0 * 3.0 * 6.0e-3,
            "delay": 2.0,
            "synMech": "Adend2GABAs",
            "sec": "Adend2",
            "loc": 0.5,
        }

    sim_config = specs.SimConfig()
    sim_config.duration = float(duration_ms)
    sim_config.dt = 0.025
    sim_config.hParams["v_init"] = -65.0
    sim_config.createNEURONObj = True
    sim_config.createPyStruct = True
    sim_config.addSynMechs = True
    sim_config.recordCells = []
    sim_config.recordTraces = {}
    sim_config.recordCellsSpikes = -1
    sim_config.recordTime = True
    sim_config.recordStep = 0.025
    sim_config.use_local_dt = False
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


class CA3NetpyneBuilderMixin:
    connections = True
    wseed = 4321
    duration_ms = 20.0
    pyr_current_na = 50e-3
    olm_current_na = -25e-3
    modeldb_dir = Path(__file__).resolve().parents[1] / "mind_sim" / "mod"
    population_order = 1

    def configure(self):
        from neuron import load_mechanisms

        load_mechanisms(str(self.modeldb_dir))
        net_params, sim_config = make_netpyne_params(
            connections=bool(self.connections),
            wseed=int(self.wseed),
            duration_ms=float(self.duration_ms),
            pyr_current_na=float(self.pyr_current_na),
            olm_current_na=float(self.olm_current_na),
        )
        super().configure(net_params, sim_config, autoCreateSpikingNodes=True)

    def proxy_node_synaptic_model_funcs(self):
        return {"PYR": lambda _source_node, _target_node: "Adend3AMPAf"}

    def set_defaults(self):
        self.populations = [
            {"label": "PYR", "model": "PYR", "nodes": self.spiking_nodes_inds, "params": {"global_label": "PYR"}, "scale": PYR_COUNT},
            {"label": "BAS", "model": "BAS", "nodes": self.spiking_nodes_inds, "params": {"global_label": "BAS"}, "scale": BAS_COUNT},
            {"label": "OLM", "model": "OLM", "nodes": self.spiking_nodes_inds, "params": {"global_label": "OLM"}, "scale": OLM_COUNT},
        ]
        self.populations_connections = []
        self.nodes_connections = []
        self.output_devices = [self.set_spike_recorder()]
        self.input_devices = []

    def set_spike_recorder(self):
        connections = OrderedDict()
        connections["CA3"] = ("PYR", "BAS", "OLM")
        return {
            "model": "spike_recorder_all",
            "params": {},
            "connections": connections,
            "nodes": self.spiking_nodes_inds,
        }

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super().build()


class CA3TVBNetpyneInterfaceBuilderMixin:
    ca3_spikes_to_epileptor_x = None

    def default_output_config(self):
        drive_weight = float(getattr(self, "_ca3_drive_weight", 0.02e-3))
        drive_delay_ms = float(getattr(self, "_ca3_drive_delay_ms", 0.2))
        interface = self._get_output_interfaces()
        interface["model"] = "RATE"
        interface["voi"] = "x1"
        interface["populations"] = "PYR"
        interface["spiking_proxy_inds"] = self.proxy_inds
        interface["proxy_inds"] = self.proxy_inds
        interface["coupling_mode"] = "TVB"
        interface["transformer_params"] = {"scale_factor": np.asarray([PYR_COUNT], dtype=float)}
        interface["proxy_params"] = {"number_of_neurons": PYR_COUNT}
        interface["weights"] = drive_weight
        interface["delays"] = drive_delay_ms
        interface["receptor_type"] = lambda _source_node, _target_node: "Adend3AMPAf"

    def default_input_config(self):
        drive_delay_ms = float(getattr(self, "_ca3_drive_delay_ms", 0.2))
        interface = self._get_input_interfaces()
        interface["model"] = "SPIKES"
        interface["voi"] = "x1"
        interface["populations"] = ("PYR", "BAS", "OLM")
        interface["spiking_proxy_inds"] = self.proxy_inds
        interface["proxy_inds"] = self.proxy_inds
        interface["transformer_model"] = self.ca3_spikes_to_epileptor_x
        interface["transformer_params"] = {
            "tau_pyr_ms": np.asarray([50.0], dtype=float),
            "tau_bas_ms": np.asarray([20.0], dtype=float),
            "tau_olm_ms": np.asarray([80.0], dtype=float),
            "x_baseline": np.asarray([-1.8], dtype=float),
            "pyr_gain": np.asarray([2.0], dtype=float),
            "bas_gain": np.asarray([-0.7], dtype=float),
            "olm_gain": np.asarray([-0.4], dtype=float),
            "exchange_window_ms": np.asarray([float(getattr(self, "_ca3_exchange_window_ms", 0.5))], dtype=float),
        }
        interface["proxy_params"] = {}
        interface["delays"] = drive_delay_ms


def build_tvb_model(labels: list[str], micro_index: int, macro_i_ext: float, kvf: float, ks: float):
    from tvb.simulator.models.epileptor import Epileptor2D

    micro_label = labels[int(micro_index)]
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
    x0_values = []
    for label in labels:
        if label == micro_label:
            x0_values.append(-1.6)
        elif label in propagation_labels:
            x0_values.append(-1.9)
        else:
            x0_values.append(-2.4)
    x0 = np.asarray(x0_values, dtype=float)
    return Epileptor2D(
        a=np.asarray([1.0], dtype=float),
        b=np.asarray([3.0], dtype=float),
        c=np.asarray([1.0], dtype=float),
        d=np.asarray([5.0], dtype=float),
        r=np.asarray([0.00035], dtype=float),
        x0=x0,
        Iext=np.asarray([float(macro_i_ext)], dtype=float),
        slope=np.asarray([0.0], dtype=float),
        Kvf=np.asarray([float(kvf)], dtype=float),
        Ks=np.asarray([float(ks)], dtype=float),
        tt=np.asarray([1.0], dtype=float),
        modification=np.asarray([False], dtype=bool),
        variables_of_interest=("x1", "z"),
    )


def configure_tvb_multiscale(
    *,
    labels: list[str],
    weights: np.ndarray,
    delays: np.ndarray,
    ca3_index: int,
    duration_ms: float,
    macro_i_ext: float,
    drive_weight: float,
    pyr_current_na: float,
    olm_current_na: float,
    conduction_speed_mm_per_ms: float,
    output_base: Path,
    exchange_window_ms: float,
):
    from tvb.basic.profile import TvbProfile

    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    from tvb.simulator import coupling, integrators, monitors
    from tvb.basic.neotraits.api import NArray
    from tvb_multiscale.core.interfaces.base.transformers.models.elephant import ElephantSpikesHistogramRate
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorNetpyneBuilder
    from tvb_multiscale.tvb_netpyne.config import Config, initialize_logger
    from tvb_multiscale.tvb_netpyne.interfaces.models.default import DefaultTVBNetpyneInterfaceBuilder
    from tvb_multiscale.tvb_netpyne.interfaces.io import NetpyneSpikeRecorderSet
    from tvb_multiscale.tvb_netpyne.netpyne_models.devices import NetpyneOutputDeviceDict, NetpyneSpikeRecorder
    from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
    from tvb_multiscale.tvb_netpyne.orchestrators import TVBNetpyneSerialOrchestrator

    install_ca3_stimulus_mapping()

    class NetpyneAllSpikeRecorder(NetpyneSpikeRecorder):
        def __init__(self, *args, **kwargs):
            self._population_labels = []
            super().__init__(*args, **kwargs)

        @property
        def population_label(self):
            return "+".join(self._population_labels)

        @population_label.setter
        def population_label(self, value):
            if value not in self._population_labels:
                self._population_labels = [*self._population_labels, value]

        @property
        def neurons(self):
            return np.asarray(
                [gid for label in self._population_labels for gid in self.netpyne_instance.cellGidsForPop(label)],
                dtype=int,
            )

    class CA3SpikesToEpileptorX(ElephantSpikesHistogramRate):
        tau_pyr_ms = NArray(label="PYR activity decay tau", required=True, default=np.asarray([50.0], dtype=float))
        tau_bas_ms = NArray(label="BAS activity decay tau", required=True, default=np.asarray([20.0], dtype=float))
        tau_olm_ms = NArray(label="OLM activity decay tau", required=True, default=np.asarray([80.0], dtype=float))
        x_baseline = NArray(label="Epileptor x baseline", required=True, default=np.asarray([-1.8], dtype=float))
        pyr_gain = NArray(label="PYR Epileptor x gain", required=True, default=np.asarray([2.0], dtype=float))
        bas_gain = NArray(label="BAS Epileptor x gain", required=True, default=np.asarray([-0.7], dtype=float))
        olm_gain = NArray(label="OLM Epileptor x gain", required=True, default=np.asarray([-0.4], dtype=float))
        exchange_window_ms = NArray(label="Exchange window", required=True, default=np.asarray([0.5], dtype=float))
        activity = NArray(label="Activity state", required=False, default=np.asarray([], dtype=float))

        def _compute(self, input_buffer, *args, **kwargs):
            proxy_count = len(input_buffer)
            if self.activity.size != proxy_count * 3:
                self.activity = np.zeros(proxy_count * 3, dtype=float)
            activity_state = self.activity.reshape((proxy_count, 3))
            tau_pyr_ms = self._assert_size("tau_pyr_ms")
            tau_bas_ms = self._assert_size("tau_bas_ms")
            tau_olm_ms = self._assert_size("tau_olm_ms")
            x_baseline = self._assert_size("x_baseline")
            pyr_gain = self._assert_size("pyr_gain")
            bas_gain = self._assert_size("bas_gain")
            olm_gain = self._assert_size("olm_gain")
            exchange_window_ms = self._assert_size("exchange_window_ms")
            steps = np.arange(int(self.input_time[0]), int(self.input_time[-1]) + 1, dtype=int)
            output = np.zeros((proxy_count, steps.size), dtype=float)
            for proxy_index, proxy_buffer in enumerate(input_buffer):
                if isinstance(proxy_buffer, dict):
                    spikes = np.asarray(proxy_buffer.get("times", []), dtype=float)
                    senders = np.asarray(proxy_buffer.get("senders", []), dtype=int)
                else:
                    spikes = np.asarray(proxy_buffer, dtype=float)
                    senders = np.full(spikes.shape, -1, dtype=int)
                pyr_activity, bas_activity, olm_activity = activity_state[proxy_index]
                window_ms = float(exchange_window_ms[proxy_index])
                for step_index, step in enumerate(steps):
                    t0 = float(step - 1) * window_ms
                    t1 = float(step) * window_ms
                    window_spikes = (spikes > t0) & (spikes <= t1)
                    current_time = t0
                    window_times = spikes[window_spikes]
                    if senders.size == spikes.size and np.any(senders >= 0):
                        window_senders = senders[window_spikes]
                    else:
                        window_senders = np.full(window_times.shape, 0, dtype=int)
                    if window_times.size:
                        order = np.argsort(window_times, kind="mergesort")
                        for spike_time, sender in zip(window_times[order], window_senders[order]):
                            event_time = min(float(spike_time), t1)
                            decay_ms = max(0.0, event_time - current_time)
                            pyr_activity *= np.exp(-decay_ms / float(tau_pyr_ms[proxy_index]))
                            bas_activity *= np.exp(-decay_ms / float(tau_bas_ms[proxy_index]))
                            olm_activity *= np.exp(-decay_ms / float(tau_olm_ms[proxy_index]))
                            current_time = event_time
                            if 0 <= int(sender) < PYR_COUNT:
                                pyr_activity += 1.0 / float(PYR_COUNT)
                            elif PYR_COUNT <= int(sender) < PYR_COUNT + BAS_COUNT:
                                bas_activity += 1.0 / float(BAS_COUNT)
                            elif PYR_COUNT + BAS_COUNT <= int(sender) < PYR_COUNT + BAS_COUNT + OLM_COUNT:
                                olm_activity += 1.0 / float(OLM_COUNT)
                    decay_ms = max(0.0, t1 - current_time)
                    pyr_activity *= np.exp(-decay_ms / float(tau_pyr_ms[proxy_index]))
                    bas_activity *= np.exp(-decay_ms / float(tau_bas_ms[proxy_index]))
                    olm_activity *= np.exp(-decay_ms / float(tau_olm_ms[proxy_index]))
                    output[proxy_index, step_index] = (
                        float(x_baseline[proxy_index])
                        + float(pyr_gain[proxy_index]) * pyr_activity
                        + float(bas_gain[proxy_index]) * bas_activity
                        + float(olm_gain[proxy_index]) * olm_activity
                    )
                activity_state[proxy_index] = [pyr_activity, bas_activity, olm_activity]
            return output

    NetpyneOutputDeviceDict["spike_recorder_all"] = NetpyneAllSpikeRecorder
    NetpyneSpikeRecorderSet.model = "spike_recorder_all"

    class CA3NetpyneBuilder(CA3NetpyneBuilderMixin, NetpyneNetworkBuilder):
        pass

    class CA3TVBNetpyneInterfaceBuilder(CA3TVBNetpyneInterfaceBuilderMixin, DefaultTVBNetpyneInterfaceBuilder):
        pass

    config = Config(output_base=str(output_base), initialize_logger=True, verbosity=0)
    config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder_all"] = {}
    config.SIMULATION_LENGTH = float(duration_ms)
    config.TVB_TO_SPIKING_DT_RATIO = 4
    config.MIN_SPIKING_DT = 0.025
    config.DEFAULT_SPIKING_MIN_DELAY = 0.025

    conn = make_tvb_connectivity(labels, weights, delays, conduction_speed_mm_per_ms)
    cosim_builder = CoSimulatorNetpyneBuilder()
    cosim_builder.connectivity = conn
    cosim_builder.model = build_tvb_model(labels, ca3_index, macro_i_ext=macro_i_ext, kvf=0.35, ks=0.0)
    cosim_builder.coupling = coupling.Linear(a=np.asarray([1.0], dtype=float))
    cosim_builder.integrator = integrators.EulerDeterministic(dt=0.1)
    cosim_builder.noise_strength = np.asarray([0.0], dtype=float)
    cosim_builder.monitors = (monitors.Raw(period=0.1),)
    cosim_builder.monitor_period = 0.1
    cosim_builder.simulation_length = float(duration_ms)
    cosim_builder.initial_conditions = make_initial_conditions(labels, ca3_index, tvb_history_steps(weights, delays, 0.1))
    cosim_builder.scale_connectivity_weights = None
    cosim_builder.scale_connectivity_weights_by_percentile = None
    cosim_builder.remove_self_connections = True
    cosim_builder.delays_flag = True
    cosim_builder.min_tract_length = 0.1 * conduction_speed_mm_per_ms

    ca3_builder = CA3NetpyneBuilder()
    ca3_builder.config = config
    ca3_builder.spiking_nodes_inds = np.asarray([ca3_index], dtype=int)
    ca3_builder.duration_ms = float(duration_ms) + 1.0
    ca3_builder.pyr_current_na = float(pyr_current_na)
    ca3_builder.olm_current_na = float(olm_current_na)
    ca3_builder.population_order = 1

    interface_builder = CA3TVBNetpyneInterfaceBuilder()
    interface_builder.config = config
    interface_builder.proxy_inds = np.asarray([ca3_index], dtype=int)
    interface_builder.exclusive_nodes = True
    interface_builder.model = "RATE"
    interface_builder.default_coupling_mode = "TVB"
    interface_builder._ca3_drive_weight = float(drive_weight)
    interface_builder._ca3_drive_delay_ms = 0.2
    interface_builder._ca3_exchange_window_ms = float(exchange_window_ms)
    interface_builder.ca3_spikes_to_epileptor_x = CA3SpikesToEpileptorX
    interface_builder.output_interfaces = []
    interface_builder.input_interfaces = []

    logger = initialize_logger(__name__, config=config)
    orchestrator = TVBNetpyneSerialOrchestrator(config=config, logger=logger)
    orchestrator.tvb_app.cosimulator_builder = cosim_builder
    orchestrator.spikeNet_app.spikeNet_builder = ca3_builder
    orchestrator.tvb_app.interfaces_builder = interface_builder
    return orchestrator


def save_outputs(
    output: Path,
    orchestrator,
    labels: list[str],
    weights: np.ndarray,
    delays: np.ndarray,
    metadata: dict,
    ca3_index: int,
    exchange_window_ms: float,
    direct_voltage_recorder: dict[str, object] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tvb_results = orchestrator.tvb_app.results
    tvb_time = np.asarray([], dtype=float)
    tvb_data = np.asarray([], dtype=float)
    if tvb_results:
        first = tvb_results[0]
        if first is not None:
            tvb_time = np.asarray(first[0], dtype=float)
            tvb_data = np.asarray(first[1], dtype=float)
    voltage_time, voltage = extract_recorded_voltage(orchestrator, metadata, direct_voltage_recorder)

    spike_times = np.asarray([], dtype=float)
    spike_gids = np.asarray([], dtype=int)
    try:
        spike_times, spike_gids = orchestrator.spikeNet_app.spiking_cosimulator.getSpikes()
        spike_times = np.asarray(spike_times, dtype=float)
        spike_gids = np.asarray(spike_gids, dtype=int)
    except Exception as exc:  # pragma: no cover - diagnostic path
        metadata["spike_read_error"] = str(exc)

    macro_x = np.asarray([], dtype=float)
    macro_z = np.asarray([], dtype=float)
    if tvb_data.ndim == 4 and tvb_data.shape[1] >= 2:
        macro_x = tvb_data[:, 0, :, 0].copy()
        macro_z = tvb_data[:, 1, :, 0].copy()
    elif tvb_data.ndim == 3 and tvb_data.shape[1] >= 2:
        macro_x = tvb_data[:, 0, :].copy()
        macro_z = tvb_data[:, 1, :].copy()
    if macro_x.size:
        macro_x[:, ca3_index] = ca3_bridge_x_trace(
            tvb_time,
            0.0,
            spike_times,
            spike_gids,
            tau_pyr_ms=50.0,
            tau_bas_ms=20.0,
            tau_olm_ms=80.0,
            x_baseline=-1.8,
            pyr_gain=2.0,
            bas_gain=-0.7,
            olm_gain=-0.4,
        )
        macro_z[:, ca3_index] = np.nan
        metadata["ca3_bridge_output"] = (
            "macro_x contains population-specific PYR/BAS/OLM event-derived "
            "CA3 x; tvb_raw keeps raw TVB samples"
        )
    macro_exposures = (
        np.stack((macro_x, macro_z), axis=2)
        if macro_x.size and macro_z.size
        else np.empty((0, len(labels), 2), dtype=float)
    )

    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        exposure_names=np.asarray(["x", "z"], dtype=object),
        time_ms=tvb_time,
        tvb_time_ms=tvb_time,
        tvb_raw=tvb_data,
        macro_exposures=macro_exposures,
        macro_x=macro_x,
        macro_z=macro_z,
        voltage_time=voltage_time,
        voltage=voltage,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def force_tvb_coupling_scale(orchestrator, scale: float) -> float:
    cosimulator = orchestrator.tvb_app.cosimulator
    value = np.asarray([float(scale)], dtype=float)
    cosimulator.coupling.a = value
    cosimulator.coupling.configure()
    for monitor in getattr(cosimulator, "cosim_monitors", ()):
        monitor_coupling = getattr(monitor, "coupling", None)
        if monitor_coupling is not None and hasattr(monitor_coupling, "a"):
            monitor_coupling.a = value.copy()
            monitor_coupling.configure()
    try:
        orchestrator.tvb_app.dumb_tvb_simulator_serialized()
    except Exception:
        pass
    return float(cosimulator.coupling.a[0])


def install_neuron_random_poisson_generator(*, seed: int, roi_index: int, dt_macro_ms: float) -> str:
    from neuron import h
    from tvb_multiscale.tvb_netpyne.netpyne import utils as netpyne_utils
    from tvb_multiscale.tvb_netpyne.netpyne_models import devices as netpyne_devices

    seed_low = int(seed) & 0xFFFFFFFF
    seed_high = (int(seed) >> 32) & 0xFFFFFFFF
    stream_cache: dict[int, list] = {}

    def streams_for(num_neurons: int):
        if num_neurons not in stream_cache:
            streams = []
            for neuron_index in range(int(num_neurons)):
                stream = h.Random()
                stream.Random123(int(seed_high ^ int(roi_index)), int(seed_low), int(neuron_index))
                streams.append(stream)
            stream_cache[int(num_neurons)] = streams
        return stream_cache[int(num_neurons)]

    def poisson_count(stream, mean: float) -> int:
        if mean <= 0.0:
            return 0
        limit = math.exp(-float(mean))
        product = 1.0
        count = -1
        stream.uniform(0.0, 1.0)
        while product > limit:
            count += 1
            product *= float(stream.repick())
        return int(count)

    def generate_spikes_for_population(num_neurons, rates, times):
        rates = np.asarray(rates, dtype=float).reshape(-1)
        times = np.asarray(times, dtype=float).reshape(-1)
        interval_count = min(int(rates.size), int(times.size))
        spikes_per_neuron: dict[int, list[float]] = {}
        streams = streams_for(int(num_neurons))

        for interval_index in range(1, interval_count):
            interval_start = float(times[interval_index - 1])
            interval_stop = float(times[interval_index])
            window_ms = interval_stop - interval_start
            rate_hz = max(0.0, float(rates[interval_index]))
            if window_ms <= 0.0 or rate_hz <= 0.0:
                continue
            exchange_start_step = int(round(interval_start / float(dt_macro_ms)))
            mean = rate_hz * window_ms / 1000.0
            for neuron_index, stream in enumerate(streams):
                stream.seq(max(0, exchange_start_step))
                count = poisson_count(stream, mean)
                if count <= 0:
                    continue
                stream.uniform(0.0, 1.0)
                neuron_spikes = spikes_per_neuron.setdefault(int(neuron_index), [])
                for _ in range(count):
                    neuron_spikes.append(interval_start + window_ms * float(stream.repick()))
        return spikes_per_neuron

    netpyne_utils.generateSpikesForPopulation = generate_spikes_for_population
    netpyne_devices.generateSpikesForPopulation = generate_spikes_for_population
    return "NEURON h.Random Random123 + Knuth Poisson + uniform-in-window event placement"


def install_netpyne_threads(num_threads: int) -> int:
    from neuron import h
    from netpyne import sim as netpyne_sim
    from netpyne.sim import setup as netpyne_setup

    original_create_parallel_context = getattr(
        netpyne_setup.createParallelContext,
        "_mind_original_create_parallel_context",
        netpyne_setup.createParallelContext,
    )

    def create_parallel_context_with_threads():
        original_create_parallel_context()
        netpyne_sim.pc.nthread(int(num_threads))

    create_parallel_context_with_threads._mind_original_create_parallel_context = (
        original_create_parallel_context
    )
    netpyne_setup.createParallelContext = create_parallel_context_with_threads
    netpyne_sim.createParallelContext = create_parallel_context_with_threads
    pc = getattr(netpyne_sim, "pc", None)
    if pc is None:
        pc = h.ParallelContext()
        netpyne_sim.pc = pc
    pc.nthread(int(num_threads))
    return int(pc.nthread())


def main() -> None:
    parser = argparse.ArgumentParser(description="True TVB-multiscale Epileptor2D macro plus NetPyNE CA3 micro reference.")
    parser.add_argument(
        "--connectivity-csv",
        type=Path,
        required=True,
        help="Labelled matrix CSV with weights and delays_ms sections.",
    )
    parser.add_argument("--duration-ms", "--t-ms", dest="duration_ms", type=float, default=20.0)
    parser.add_argument("--macro-i-ext", "--i-ext", dest="macro_i_ext", type=float, default=3.1)
    parser.add_argument("--drive-weight", "--drive-w", dest="drive_weight", type=float, default=0.02e-3)
    parser.add_argument("--pyr-current-na", "--pyr-i-na", dest="pyr_current_na", type=float, default=50e-3)
    parser.add_argument("--olm-current-na", "--olm-i-na", dest="olm_current_na", type=float, default=-25e-3)
    parser.add_argument("--conduction-speed-mm-per-ms", type=float, default=5.0)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_netpyne_ca3_cosim.npz"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_netpyne_ca3_workdir"),
    )
    args = parser.parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")

    for path in (
        str(Path(__file__).resolve().parents[3] / "src" / "python_api"),
        str(Path(__file__).resolve().parents[3] / "build"),
    ):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import mind_sim as ms

    add_tvb_multiscale_paths(Path.home() / "tvb-root", Path.home() / "tvb-multiscale")
    dt_macro_ms = 0.1
    exchange_window_ms = 0.5
    rois = ms.macro.load_rois(args.connectivity_csv)
    labels = list(rois.labels)
    weights = np.asarray(rois.weights, dtype=float)
    delays = np.asarray(rois.delays, dtype=float)
    ca3_label = "Left-CA3"
    if ca3_label not in labels:
        raise SystemExit("connectivity labels do not contain ROI 'Left-CA3'")
    ca3_index = labels.index(ca3_label)
    min_positive_delay = float(np.min(delays[delays > 0.0])) if np.any(delays > 0.0) else 0.0
    if min_positive_delay <= 0.0:
        raise SystemExit("connectivity delays must contain at least one positive delay")
    exchange_steps_float = exchange_window_ms / dt_macro_ms
    exchange_steps = int(round(exchange_steps_float))
    if exchange_steps < 1 or not math.isclose(exchange_steps_float, exchange_steps, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("exchange window must be an integer multiple of dt_macro")
    if exchange_window_ms > min_positive_delay + 1e-9:
        raise SystemExit("exchange window must not exceed the minimum positive connectivity delay")

    args.workdir.mkdir(parents=True, exist_ok=True)
    pre_start = time.perf_counter()
    netpyne_num_threads = install_netpyne_threads(4)
    macro2micro_random_stream = install_neuron_random_poisson_generator(
        seed=1,
        roi_index=ca3_index,
        dt_macro_ms=dt_macro_ms,
    )
    orchestrator = configure_tvb_multiscale(
        labels=labels,
        weights=weights,
        delays=delays,
        ca3_index=ca3_index,
        duration_ms=float(args.duration_ms),
        macro_i_ext=float(args.macro_i_ext),
        drive_weight=float(args.drive_weight),
        pyr_current_na=float(args.pyr_current_na),
        olm_current_na=float(args.olm_current_na),
        conduction_speed_mm_per_ms=float(args.conduction_speed_mm_per_ms),
        output_base=args.workdir,
        exchange_window_ms=exchange_window_ms,
    )
    orchestrator.start()
    orchestrator.configure()
    orchestrator.build()
    direct_voltage_metadata: dict[str, object] = {}
    direct_voltage_recorder = install_direct_voltage_recorder(dt_macro_ms / 4.0, direct_voltage_metadata)
    actual_coupling_a = force_tvb_coupling_scale(orchestrator, 1.0)
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    orchestrator.simulate(float(args.duration_ms))
    run_s = time.perf_counter() - run_start

    metadata = {
        "source": "TVB-multiscale TVB macro API plus NetPyNE/NEURON ModelDB 186768 CA3",
        "macro_backend": "tvb_multiscale CoSimulatorNetpyneBuilder + TVB built-in Epileptor2D",
        "micro_backend": "tvb_multiscale NetPyNE + NEURON",
        "interface": "TVBNetpyneSerialOrchestrator, TVB coupling to PYR poisson proxy, PYR+BAS+OLM spikes to Epileptor x1",
        "macro2micro_random_stream": macro2micro_random_stream,
        "stimulus_mapping": "CA3 external proxy uses explicit source_i_to_PYR_i connList at Adend3(0.5)",
        "connectivity_csv": str(args.connectivity_csv),
        "duration_ms": float(args.duration_ms),
        "dt_macro_ms": float(dt_macro_ms),
        "dt_micro_ms": 0.025,
        "netpyne_num_threads": int(netpyne_num_threads),
        "initial_history_steps": int(tvb_history_steps(weights, delays, dt_macro_ms)),
        "min_positive_delay_ms": float(min_positive_delay),
        "exchange_window_ms": float(exchange_window_ms),
        "exchange_window_steps": int(exchange_steps),
        "ca3_label": ca3_label,
        "ca3_index": int(ca3_index),
        "macro_i_ext": float(args.macro_i_ext),
        "tvb_coupling_a": float(actual_coupling_a),
        "epileptor2d_kvf": 0.35,
        "epileptor2d_ks": 0.0,
        "epileptor2d_r": 0.00035,
        "epileptor2d_tt": 1.0,
        "epileptor2d_modification": False,
        "drive_weight": float(args.drive_weight),
        "drive_delay_ms": 0.2,
        "pyr_current_na": float(args.pyr_current_na),
        "olm_current_na": float(args.olm_current_na),
        "conduction_speed_mm_per_ms": float(args.conduction_speed_mm_per_ms),
        "voltage_recording": "first PYR soma voltage, fixed output key voltage",
        "recorded_exposures": ["x", "z"],
        "pre_run_s": float(pre_run_s),
        "run_s": float(run_s),
    }
    metadata.update(direct_voltage_metadata)
    try:
        save_outputs(
            args.output,
            orchestrator,
            labels,
            weights,
            delays,
            metadata,
            ca3_index,
            exchange_window_ms,
            direct_voltage_recorder,
        )
    finally:
        try:
            orchestrator.clean_up()
        except TypeError:
            # NetPyNE 1.1.1 can fail in postRun timing bookkeeping after
            # interval runs; traces and spikes are already collected above.
            pass
        orchestrator.stop()

    print(f"output={args.output}")
    print("backend=tvb_multiscale_tvb_netpyne")
    print(f"netpyne_num_threads={netpyne_num_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


if __name__ == "__main__":
    main()
