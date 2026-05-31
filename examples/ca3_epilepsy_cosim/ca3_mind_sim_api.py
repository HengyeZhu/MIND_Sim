#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MIND_MOD_DIR = HERE / "mind_mod_ca3"
MIND_BRIDGE_MOD_DIR = HERE / "mind_mod"
RESULT_DIR = HERE / "outputs"

PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200
PSR_COUNT = 1
SPIKE_THRESHOLD_MV = 0.0
DT_MS = 0.1
V_INIT_MV = -65.0
DEFAULT_RECORD = "0:soma:0.5,0:Bdend:1,0:Adend3:0.5,800:soma:0.5,1000:soma:0.5"

POPULATION_BANDS = {
    "PYR": (0, PYR_COUNT),
    "BAS": (PYR_COUNT, PYR_COUNT + BAS_COUNT),
    "OLM": (PYR_COUNT + BAS_COUNT, PYR_COUNT + BAS_COUNT + OLM_COUNT),
}


@dataclass(frozen=True)
class PopulationSpec:
    name: str
    count: int
    gid_begin: int


@dataclass
class MindBuild:
    sim: object
    populations: dict[str, object]
    soma_locs: dict[int, object]
    section_locs: dict[tuple[int, str, float], object]
    synapses: dict[tuple[int, str], object]
    gid_to_pop: dict[int, str]
    stimuli: list[object]


def ensure_mind_sim_import() -> None:
    build_dir = REPO_ROOT / "build"
    src_api = REPO_ROOT / "src" / "python_api"
    for path in (str(build_dir), str(src_api)):
        if path not in sys.path:
            sys.path.insert(0, path)


def add_pt3d_section(name: str, label: str, points: list[tuple[float, float, float, float]]):
    import mind_sim as ms

    sec = ms.section(name, label)
    sec.nseg = 1
    sec.pt3d = points
    return sec


def make_pyr_sections():
    soma = add_pt3d_section("soma", "soma", [(0.0, 0.0, 0.0, 20.0), (0.0, 0.0, 20.0, 20.0)])
    bdend = add_pt3d_section("Bdend", "Bdend", [(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, -200.0, 2.0)])
    adend1 = add_pt3d_section("Adend1", "Adend1", [(0.0, 0.0, 20.0, 2.0), (0.0, 0.0, 170.0, 2.0)])
    adend2 = add_pt3d_section("Adend2", "Adend2", [(0.0, 0.0, 170.0, 2.0), (0.0, 0.0, 320.0, 2.0)])
    adend3 = add_pt3d_section("Adend3", "Adend3", [(0.0, 0.0, 320.0, 2.0), (0.0, 0.0, 470.0, 2.0)])
    bdend.connect(soma, 0.0)
    adend1.connect(soma, 0.5)
    adend2.connect(adend1, 1.0)
    adend3.connect(adend2, 1.0)
    return [soma, bdend, adend1, adend2, adend3]


def make_interneuron_sections():
    total_area_um2 = 10000.0
    diam = math.sqrt(total_area_um2)
    length = diam / math.pi
    return [add_pt3d_section("soma", "soma", [(0.0, 0.0, 0.0, diam), (0.0, 0.0, length, diam)])]


def insert_pyr_biophysics(cell) -> None:
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


def insert_bas_biophysics(cell) -> None:
    soma = cell.group("soma")
    soma.Ra = 35.4
    soma.cm = 1.0
    soma.insert("pas", e=-65.0, g=0.1e-3)
    soma.insert("Nafbwb")
    soma.insert("Kdrbwb")


def insert_olm_biophysics(cell) -> None:
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


def synapse(loc, kind: str):
    if kind == "ampa":
        return loc.insert("MyExp2SynBB", tau1=0.05, tau2=5.3, e=0.0)
    if kind == "gaba_fast":
        return loc.insert("MyExp2SynBB", tau1=0.07, tau2=9.1, e=-80.0)
    if kind == "gaba_slow":
        return loc.insert("MyExp2SynBB", tau1=0.2, tau2=20.0, e=-80.0)
    if kind == "gaba_septal":
        return loc.insert("MyExp2SynBB", tau1=20.0, tau2=40.0, e=-80.0)
    if kind == "nmda":
        return loc.insert("MyExp2SynNMDABB", tau1=0.05, tau2=5.3, tau1NMDA=15.0, tau2NMDA=150.0, r=1.0, e=0.0)
    raise ValueError(f"unknown synapse kind: {kind}")


def make_conn(pre_n: int, post_n: int, conv: int) -> np.ndarray:
    out = np.zeros((post_n, conv), dtype=np.int32)
    for post in range(post_n):
        out[post, :] = random.sample(list(range(pre_n)), conv)
    return out


def gid_for(populations: dict[str, PopulationSpec], pop: str, local: int) -> int:
    return populations[pop].gid_begin + int(local)


def connect_projection(
    network,
    populations: dict[str, PopulationSpec],
    synapses: dict[tuple[int, str], object],
    *,
    pre_pop: str,
    post_pop: str,
    syn_name: str,
    delay_ms: float,
    weight: float,
    conv: int,
) -> int:
    conn = make_conn(populations[pre_pop].count, populations[post_pop].count, conv)
    count = 0
    for post_local, pres in enumerate(conn):
        post_gid = gid_for(populations, post_pop, post_local)
        target = synapses[(post_gid, syn_name)]
        for pre_local in pres:
            pre_gid = gid_for(populations, pre_pop, int(pre_local))
            network.gid_connect(pre_gid, target, float(weight), float(delay_ms))
            count += 1
    return count


def population_spike_counts(spike_gids: np.ndarray) -> dict[str, int]:
    gids = np.asarray(spike_gids, dtype=np.int64)
    return {
        name: int(np.count_nonzero((gids >= begin) & (gids < end)))
        for name, (begin, end) in POPULATION_BANDS.items()
    }


def population_spike_rates(spike_gids: np.ndarray, duration_ms: float) -> dict[str, float]:
    duration_s = max(float(duration_ms) / 1000.0, 1e-12)
    counts = population_spike_counts(spike_gids)
    sizes = {"PYR": PYR_COUNT, "BAS": BAS_COUNT, "OLM": OLM_COUNT}
    return {name: float(counts[name]) / float(sizes[name]) / duration_s for name in counts}


def set_mind_nmda_r(build: MindBuild, *, olm: float, bas: float, pyr_bdend: float, pyr_adend3: float) -> None:
    for gid in range(OLM_COUNT):
        build.synapses[(PYR_COUNT + BAS_COUNT + gid, "somaNMDA")].r = float(olm)
    for gid in range(BAS_COUNT):
        build.synapses[(PYR_COUNT + gid, "somaNMDA")].r = float(bas)
    for gid in range(PYR_COUNT):
        build.synapses[(gid, "BdendNMDA")].r = float(pyr_bdend)
        build.synapses[(gid, "Adend3NMDA")].r = float(pyr_adend3)


def set_mind_nmda_wash(
    build: MindBuild,
    *,
    washin_ms: float,
    washout_ms: float,
    olm_washin: float,
    olm_washout: float,
    bas_washin: float,
    bas_washout: float,
    pyr_bdend_washin: float,
    pyr_bdend_washout: float,
    pyr_adend3_washin: float,
    pyr_adend3_washout: float,
) -> None:
    def apply(syn, wash_in: float, wash_out: float) -> None:
        syn.washEnabled = 1.0
        syn.washInT = float(washin_ms)
        syn.washOutT = float(washout_ms)
        syn.washInR = float(wash_in)
        syn.washOutR = float(wash_out)

    for gid in range(OLM_COUNT):
        apply(build.synapses[(PYR_COUNT + BAS_COUNT + gid, "somaNMDA")], olm_washin, olm_washout)
    for gid in range(BAS_COUNT):
        apply(build.synapses[(PYR_COUNT + gid, "somaNMDA")], bas_washin, bas_washout)
    for gid in range(PYR_COUNT):
        apply(build.synapses[(gid, "BdendNMDA")], pyr_bdend_washin, pyr_bdend_washout)
        apply(build.synapses[(gid, "Adend3NMDA")], pyr_adend3_washin, pyr_adend3_washout)


def enable_mind_periodic_micro_inputs(
    build: MindBuild,
    *,
    background: bool,
    medial_septal: bool,
    ms_gain: float,
    duration_ms: float,
) -> None:
    network = build.sim.network()

    def connect_group(count: int, gid_begin: int, syn_name: str, weight: float, interval_ms: float, start_ms: float) -> None:
        stim = build.sim.insert(
            "NetStim",
            interval=float(interval_ms),
            number=(1000.0 / float(interval_ms)) * float(duration_ms),
            start=float(start_ms),
            noise=0.0,
        )
        build.stimuli.append(stim)
        for local in range(count):
            network.event_connect(
                stim,
                build.synapses[(gid_begin + local, syn_name)],
                float(weight),
                2.0 * DT_MS,
            )

    if background:
        connect_group(PYR_COUNT, 0, "somaAMPAf", 0.05e-3, 1.0, 0.0)
        connect_group(PYR_COUNT, 0, "Adend3AMPAf", 0.05e-3, 1.0, 0.0)
        connect_group(PYR_COUNT, 0, "somaGABAf", 0.012e-3, 1.0, 0.0)
        connect_group(PYR_COUNT, 0, "Adend3GABAf", 0.012e-3, 1.0, 0.0)
        connect_group(PYR_COUNT, 0, "Adend3NMDA", 6.5e-3, 100.0, 0.0)
        connect_group(BAS_COUNT, PYR_COUNT, "somaAMPAf", 0.02e-3, 1.0, 0.0)
        connect_group(BAS_COUNT, PYR_COUNT, "somaGABAf", 0.2e-3, 1.0, 0.0)
        connect_group(OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaAMPAf", 0.0625e-3, 1.0, 0.0)
        connect_group(OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaGABAf", 0.2e-3, 1.0, 0.0)
    if medial_septal:
        connect_group(BAS_COUNT, PYR_COUNT, "somaGABAss", 1.6e-3 * float(ms_gain), 150.0, 50.0)
        connect_group(OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaGABAss", 1.6e-3 * float(ms_gain), 150.0, 50.0)


def add_background_input_ports(build: MindBuild, *, ms_gain: float) -> dict[str, object]:
    network = build.sim.network()
    ports: dict[str, object] = {}

    def connect_group(port_name: str, count: int, gid_begin: int, syn_name: str, weight: float) -> None:
        group = network.spike_inputs(count)
        ports[port_name] = group
        for local in range(count):
            network.spike_connect(
                group[local],
                build.synapses[(gid_begin + local, syn_name)],
                float(weight),
                2.0 * DT_MS,
            )

    connect_group("pyr_soma_ampa", PYR_COUNT, 0, "somaAMPAf", 0.05e-3)
    connect_group("pyr_adend3_ampa", PYR_COUNT, 0, "Adend3AMPAf", 0.05e-3)
    connect_group("pyr_soma_gaba", PYR_COUNT, 0, "somaGABAf", 0.012e-3)
    connect_group("pyr_adend3_gaba", PYR_COUNT, 0, "Adend3GABAf", 0.012e-3)
    connect_group("pyr_adend3_nmda", PYR_COUNT, 0, "Adend3NMDA", 6.5e-3)
    connect_group("bas_soma_ampa", BAS_COUNT, PYR_COUNT, "somaAMPAf", 0.02e-3)
    connect_group("bas_soma_gaba", BAS_COUNT, PYR_COUNT, "somaGABAf", 0.2e-3)
    connect_group("olm_soma_ampa", OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaAMPAf", 0.0625e-3)
    connect_group("olm_soma_gaba", OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaGABAf", 0.2e-3)
    connect_group("bas_septal_gaba", BAS_COUNT, PYR_COUNT, "somaGABAss", 1.6e-3 * float(ms_gain))
    connect_group("olm_septal_gaba", OLM_COUNT, PYR_COUNT + BAS_COUNT, "somaGABAss", 1.6e-3 * float(ms_gain))
    return ports


def add_standard_synapses(build: MindBuild) -> None:
    for pop_name, population in build.populations.items():
        for cell in population:
            gid = int(cell.gid)
            soma = cell.group("soma")[0](0.5)
            build.soma_locs[gid] = soma
            build.section_locs[(gid, "soma", 0.5)] = soma

            if pop_name in {"PYR", "PSR"}:
                bdend = cell.group("Bdend")[0](1.0)
                adend2 = cell.group("Adend2")[0](0.5)
                adend3 = cell.group("Adend3")[0](0.5)
                build.section_locs[(gid, "Bdend", 1.0)] = bdend
                build.section_locs[(gid, "Adend2", 0.5)] = adend2
                build.section_locs[(gid, "Adend3", 0.5)] = adend3
                build.synapses[(gid, "somaGABAf")] = synapse(soma, "gaba_fast")
                build.synapses[(gid, "somaAMPAf")] = synapse(soma, "ampa")
                build.synapses[(gid, "BdendAMPA")] = synapse(bdend, "ampa")
                build.synapses[(gid, "BdendNMDA")] = synapse(bdend, "nmda")
                build.synapses[(gid, "Adend2GABAs")] = synapse(adend2, "gaba_slow")
                build.synapses[(gid, "Adend3GABAf")] = synapse(adend3, "gaba_fast")
                build.synapses[(gid, "Adend3AMPAf")] = synapse(adend3, "ampa")
                build.synapses[(gid, "Adend3NMDA")] = synapse(adend3, "nmda")
            else:
                build.synapses[(gid, "somaAMPAf")] = synapse(soma, "ampa")
                build.synapses[(gid, "somaGABAf")] = synapse(soma, "gaba_fast")
                build.synapses[(gid, "somaGABAss")] = synapse(soma, "gaba_septal")
                build.synapses[(gid, "somaNMDA")] = synapse(soma, "nmda")


def population_specs(include_psr: bool) -> dict[str, PopulationSpec]:
    specs: dict[str, PopulationSpec] = {}
    gid = 0
    for name, count in (("PYR", PYR_COUNT), ("BAS", BAS_COUNT), ("OLM", OLM_COUNT)):
        specs[name] = PopulationSpec(name, count, gid)
        gid += count
    if include_psr:
        specs["PSR"] = PopulationSpec("PSR", PSR_COUNT, gid)
    return specs


def build_mind_ca3(*, connections: bool, include_psr: bool, wseed: int, device: str, num_threads: int) -> MindBuild:
    ensure_mind_sim_import()
    import mind_sim as ms

    sim = ms.Sim()
    sim.set_device(device)
    sim.set_num_threads(num_threads)
    sim.set_dt(DT_MS)
    sim.set_spike_output_enabled(True)
    sim.load_mech_metadata(str(MIND_MOD_DIR))

    specs = population_specs(include_psr)
    templates = [
        {"name": "PYR", "num_cells": specs["PYR"].count, "sections": make_pyr_sections()},
        {"name": "BAS", "num_cells": specs["BAS"].count, "sections": make_interneuron_sections()},
        {"name": "OLM", "num_cells": specs["OLM"].count, "sections": make_interneuron_sections()},
    ]
    if include_psr:
        templates.append({"name": "PSR", "num_cells": specs["PSR"].count, "sections": make_pyr_sections()})
    sim.build_morphology(templates)

    populations = {name: sim.population(name) for name in specs}
    build = MindBuild(
        sim=sim,
        populations=populations,
        soma_locs={},
        section_locs={},
        synapses={},
        gid_to_pop={},
        stimuli=[],
    )

    for cell in populations["PYR"]:
        cell.v_init = V_INIT_MV
        build.gid_to_pop[int(cell.gid)] = "PYR"
        insert_pyr_biophysics(cell)
        cell.group("soma")[0](0.5).insert("IClamp", **{"del": 2.0 * DT_MS}, dur=1.0e9, amp=50e-3)
    for cell in populations["BAS"]:
        cell.v_init = V_INIT_MV
        build.gid_to_pop[int(cell.gid)] = "BAS"
        insert_bas_biophysics(cell)
    for cell in populations["OLM"]:
        cell.v_init = V_INIT_MV
        build.gid_to_pop[int(cell.gid)] = "OLM"
        insert_olm_biophysics(cell)
        cell.group("soma")[0](0.5).insert("IClamp", **{"del": 2.0 * DT_MS}, dur=1.0e9, amp=-25e-3)
    if include_psr:
        for cell in populations["PSR"]:
            cell.v_init = V_INIT_MV
            build.gid_to_pop[int(cell.gid)] = "PSR"
            insert_pyr_biophysics(cell)
            cell.group("soma")[0](0.5).insert("IClamp", **{"del": 2.0 * DT_MS}, dur=1.0e9, amp=50e-3)

    add_standard_synapses(build)
    network = sim.network()
    for gid, soma in build.soma_locs.items():
        network.register_gid_source(gid, soma._ref_v, SPIKE_THRESHOLD_MV)

    if connections:
        random.seed(int(wseed))
        projections = [
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
        ]
        if include_psr:
            projections.extend(
                [
                    ("PYR", "PSR", "BdendNMDA", 2.0, 0.004e-3, 25),
                    ("PYR", "PSR", "BdendAMPA", 2.0, 0.5 * 0.04e-3, 25),
                ]
            )
        for pre, post, syn_name, delay, weight, conv in projections:
            connect_projection(
                network,
                specs,
                build.synapses,
                pre_pop=pre,
                post_pop=post,
                syn_name=syn_name,
                delay_ms=delay,
                weight=weight,
                conv=conv,
            )

    return build


def parse_record(raw: str) -> list[tuple[int, str, float]]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        gid = int(parts[0])
        sec = parts[1] if len(parts) > 1 else "soma"
        loc = float(parts[2]) if len(parts) > 2 else 0.5
        out.append((gid, sec, loc))
    return out


def collect_spikes(sim, gids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    times = []
    ids = []
    for gid in gids:
        spk = np.asarray(sim.get_spk_by_gid(int(gid)), dtype=float)
        if spk.size:
            times.append(spk)
            ids.append(np.full(spk.shape, int(gid), dtype=np.int32))
    if not times:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=np.int32)
    t = np.concatenate(times)
    g = np.concatenate(ids)
    order = np.lexsort((g, t))
    return t[order], g[order]


def run_mind_validation(args: argparse.Namespace) -> Path:
    ensure_mind_sim_import()
    import mind_sim as ms

    records = parse_record(args.record)
    include_psr = any(gid >= PYR_COUNT + BAS_COUNT + OLM_COUNT for gid, _, _ in records)
    include_psr = include_psr or bool(args.include_psr)
    t0 = time.perf_counter()
    build = build_mind_ca3(
        connections=bool(args.connections),
        include_psr=include_psr,
        wseed=int(args.wseed),
        device=args.device,
        num_threads=int(args.num_threads),
    )
    sim = build.sim
    set_mind_nmda_r(
        build,
        olm=float(args.olm_soma_nmda),
        bas=float(args.bas_soma_nmda),
        pyr_bdend=float(args.pyr_bdend_nmda),
        pyr_adend3=float(args.pyr_adend3_nmda),
    )
    if args.washin_washout:
        set_mind_nmda_wash(
            build,
            washin_ms=float(args.washin_ms),
            washout_ms=float(args.washout_ms),
            olm_washin=0.0,
            olm_washout=float(args.olm_soma_nmda),
            bas_washin=float(args.bas_soma_nmda),
            bas_washout=float(args.bas_soma_nmda),
            pyr_bdend_washin=float(args.pyr_bdend_nmda),
            pyr_bdend_washout=float(args.pyr_bdend_nmda),
            pyr_adend3_washin=float(args.pyr_adend3_nmda),
            pyr_adend3_washout=float(args.pyr_adend3_nmda),
        )
    protocol_background = bool(args.background_inputs or args.medial_septal)
    use_bridge_background = bool(protocol_background and args.background_noise)
    if protocol_background and not args.background_noise:
        enable_mind_periodic_micro_inputs(
            build,
            background=bool(args.background_inputs),
            medial_septal=bool(args.medial_septal),
            ms_gain=float(args.ms_gain),
            duration_ms=float(args.duration_ms),
        )
    ports = add_background_input_ports(build, ms_gain=float(args.ms_gain)) if use_bridge_background else {}
    vectors = []
    for gid, sec, loc in records:
        vectors.append(ms.Vector().record(build.section_locs[(gid, sec, loc)]._ref_v))
    time_vector = ms.Vector().record(sim._ref_t)
    sim.build_microcircuit()
    sim.finitialize(V_INIT_MV)
    initial_record_mv = np.asarray([build.section_locs[(gid, sec, loc)]._ref_v.value() for gid, sec, loc in records], dtype=float)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    washin_applied = False
    if use_bridge_background:
        network = ms.Network(labels=["CA3"], weights=[[0.0]], delays=[[DT_MS]], exposures=["x"])
        network.record(rois="all")
        network.use_micro(
            ms.MicroCircuit(sim).bind_roi(
                0,
                gid_ranges=[
                    (0, PYR_COUNT),
                    (PYR_COUNT, PYR_COUNT + BAS_COUNT),
                    (PYR_COUNT + BAS_COUNT, PYR_COUNT + BAS_COUNT + OLM_COUNT),
                ],
                ports=ports,
            )
        )
        network.roi(0).connect(
            network.roi(0),
            MIND_BRIDGE_MOD_DIR / "ca3_paper_background_to_spikes.mod",
            params={
                "pyr_count": float(PYR_COUNT),
                "bas_count": float(BAS_COUNT),
                "olm_count": float(OLM_COUNT),
                "noise_enabled": 1.0 if args.background_noise else 0.0,
                "background_enabled": 1.0 if args.background_inputs else 0.0,
                "septal_enabled": 1.0 if args.medial_septal else 0.0,
                "rate_fast_hz": 1000.0,
                "rate_nmda_hz": 10.0,
                "septal_start_ms": 50.0,
                "septal_interval_ms": 150.0,
            },
            random={
                "rng": {
                    "state": {"seed": float(int(args.seed))},
                    "uniform": r"""
    std::uint64_t rng = (std::uint64_t) seed;
    rng = rng * 2862933555777941757ULL + 3037000493ULL;
    seed = (double) (rng & 9007199254740991ULL);
    return seed / 9007199254740992.0;
""",
                }
            },
        )
        network.roi(0).connect(network.roi(0), MIND_BRIDGE_MOD_DIR / "ca3_zero_micro_output.mod")
        result = ms.Simulator(
            network,
            dt_micro=DT_MS,
            dt_macro=DT_MS,
            batch_window=DT_MS,
            record_micro_spikes=True,
        ).run(float(args.duration_ms))
        table = result.micro_spikes_by_roi[0]
        spike_times = np.asarray(table.time, dtype=float)
        spike_gids = np.asarray(table.gid, dtype=np.int32)
        if spike_times.size:
            order = np.lexsort((spike_gids, spike_times))
            spike_times = spike_times[order]
            spike_gids = spike_gids[order]
        times_ms = np.asarray(result.times, dtype=float)
        final_record_mv = np.asarray([build.section_locs[(gid, sec, loc)]._ref_v.value() for gid, sec, loc in records], dtype=float)
        voltages = np.vstack([initial_record_mv, final_record_mv]).T
        washin_applied = bool(args.washin_washout)
    else:
        if args.washin_washout:
            sim.run(float(args.duration_ms))
            washin_applied = True
        else:
            sim.run(float(args.duration_ms))
        all_gids = sorted(build.soma_locs)
        spike_times, spike_gids = collect_spikes(sim, all_gids)
        times_ms = np.asarray(time_vector.to_python(), dtype=float)
        voltages = np.vstack([np.asarray(vec.to_python(), dtype=float) for vec in vectors])
    run_s = time.perf_counter() - t1

    labels = np.asarray([f"{gid}:{sec}:{loc:g}" for gid, sec, loc in records], dtype=object)

    output = args.output or RESULT_DIR / f"ca3_mindsim_{int(args.duration_ms)}ms.npz"
    output = Path(output).expanduser()
    if not output.is_absolute():
        output = (HERE / output).resolve()
    else:
        output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        backend=np.asarray("mind_sim", dtype=object),
        duration_ms=np.asarray(float(args.duration_ms)),
        dt_ms=np.asarray(DT_MS),
        connections=np.asarray(bool(args.connections)),
        record_labels=labels,
        times_ms=times_ms,
        voltages_mv=voltages,
        record_sample_times_ms=np.asarray([0.0, float(args.duration_ms)] if use_bridge_background else times_ms, dtype=float),
        record_initial_mv=initial_record_mv,
        record_final_mv=voltages[:, -1],
        spike_times_ms=spike_times,
        spike_gids=spike_gids,
        population_spike_counts_json=json.dumps(population_spike_counts(spike_gids), sort_keys=True),
        population_rate_hz_json=json.dumps(population_spike_rates(spike_gids, float(args.duration_ms)), sort_keys=True),
        timing_s=np.asarray([build_s, run_s, build_s + run_s], dtype=float),
        metadata_json=json.dumps(
            {
                "model": "MIND_Sim API rewrite of ModelDB 186768 CA3 microcircuit",
                "protocol": "paper_periodic_micro" if (protocol_background and not args.background_noise) else ("paper_background_bridge" if use_bridge_background else "deterministic_micro"),
                "background_inputs": bool(args.background_inputs),
                "background_noise": bool(args.background_noise),
                "medial_septal": bool(args.medial_septal),
                "nmda_ampa_ratio": {
                    "olm_somaNMDA": float(args.olm_soma_nmda),
                    "bas_somaNMDA": float(args.bas_soma_nmda),
                    "pyr_BdendNMDA": float(args.pyr_bdend_nmda),
                    "pyr_Adend3NMDA": float(args.pyr_adend3_nmda),
                },
                "washin_washout_requested": bool(args.washin_washout),
                "washin_washout_applied": bool(washin_applied),
                "washin_washout_backend": "MyExp2SynNMDABB time-dependent r" if args.washin_washout else "none",
                "connections": bool(args.connections),
                "wseed": int(args.wseed),
                "seed": int(args.seed),
                "record": [str(x) for x in labels],
            },
            sort_keys=True,
        ),
    )
    print(f"output={output}")
    print(f"build_s={build_s:.6f}")
    print(f"run_s={run_s:.6f}")
    print(f"spikes={len(spike_times)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MIND_Sim API rewrite of the ModelDB 186768 CA3 network.")
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--paper-protocol", action="store_true", help="Enable the upstream-style 5 s background/MS input protocol.")
    parser.add_argument("--connections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background-inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--medial-septal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ms-gain", type=float, default=1.0)
    parser.add_argument("--washin-washout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--washin-ms", type=float, default=1000.0)
    parser.add_argument("--washout-ms", type=float, default=2000.0)
    parser.add_argument("--olm-soma-nmda", type=float, default=1.0)
    parser.add_argument("--bas-soma-nmda", type=float, default=1.0)
    parser.add_argument("--pyr-bdend-nmda", type=float, default=1.0)
    parser.add_argument("--pyr-adend3-nmda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-psr", action="store_true")
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=os.environ.get("MIND_SIM_DEVICE", "cpu"))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MIND_SIM_NUM_THREADS", "1") or "1"))
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.paper_protocol:
        args.connections = True
        args.background_inputs = True
        args.medial_septal = True
        args.washin_washout = True
        args.duration_ms = 5000.0 if args.duration_ms is None else args.duration_ms
    elif args.duration_ms is None:
        args.duration_ms = 50.0
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    run_mind_validation(args)


if __name__ == "__main__":
    main()
