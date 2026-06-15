#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import mind_sim as ms


SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets"
DEFAULT_MOD_DIR = ASSET_DIR / "mod"

PYR_COUNT = 850
SST_COUNT = 50
PV_COUNT = 50
VIP_COUNT = 50
RECORD_LOCAL_CELLS = (0, 6, 66)


def add_pyr_sst_stub_axon(sections: list[Any]) -> list[Any]:
    axon0 = ms.section("axon_stub_0", "axon")
    axon0.connect("soma_0", 0.5)
    axon0.L_um = 20.0
    axon0.nseg = 1
    axon0.diam_um = 0.5 * (3.0 + 1.75)

    axon1 = ms.section("axon_stub_1", "axon")
    axon1.connect(axon0, 1.0)
    axon1.L_um = 30.0
    axon1.nseg = 1
    axon1.diam_um = 0.5 * (1.75 + 1.0)

    myelin = ms.section("myelin_stub_0", "myelin")
    myelin.connect(axon1, 1.0)
    myelin.L_um = 100.0
    myelin.nseg = 1
    myelin.diam_um = 500.0
    return [*sections, axon0, axon1, myelin]


def add_vip_stub_axon(sections: list[Any]) -> list[Any]:
    axon0 = ms.section("vip_axon_stub_0", "axon")
    axon0.connect("soma_0", 1.0)
    axon0.L_um = 30.0
    axon0.nseg = 1
    axon0.diam_um = 1.1062632630369478

    axon1 = ms.section("vip_axon_stub_1", "axon")
    axon1.connect(axon0, 1.0)
    axon1.L_um = 30.0
    axon1.nseg = 1
    axon1.diam_um = 0.3140589549560489
    return [*sections, axon0, axon1]


def compute_section_nseg(
    sections: list[Any],
    target_seg_length_um: float = 40.0,
    *,
    skip_label: str | None = None,
) -> None:
    for sec in sections:
        if skip_label is not None and str(sec.label) == str(skip_label):
            continue
        base = int(float(sec.L_um) / float(target_seg_length_um))
        sec.nseg = max(1, 1 + 2 * base)


def compute_apic_exp_distribution(
    sections: list[Any],
    *,
    reference_gbar: float,
    exp_offset: float,
    exp_rate: float,
    exp_shift: float,
    exp_scale: float,
    apic_label: str = "apic",
    origin_label: str = "soma",
) -> dict[str, list[float]]:
    apic_sections: list[int] = []
    names: list[str] = []
    labels: list[str] = []
    nsegs: list[int] = []
    lengths: list[float] = []
    parents: list[Any] = []
    parentxs: list[float] = []

    for idx, sec in enumerate(sections):
        names.append(str(sec.name))
        labels.append(str(sec.label))
        nsegs.append(max(1, int(sec.nseg)))
        lengths.append(max(0.0, float(sec.L_um)))
        parents.append(sec.parent)
        parentxs.append(1.0 if sec.parent is None else float(sec.parentx))
        if str(sec.label) == apic_label:
            apic_sections.append(idx)
    if not apic_sections:
        return {}

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    origin_idx = apic_sections[0]
    for idx in apic_sections:
        parent_label = None
        parent = parents[idx]
        if parent is not None:
            parent_idx = name_to_idx.get(str(parent))
            if parent_idx is not None:
                parent_label = labels[parent_idx]
        if parent_label != apic_label:
            origin_idx = idx
            if parent_label == origin_label:
                break

    apic_children: dict[int, list[int]] = {idx: [] for idx in apic_sections}
    for idx in apic_sections:
        parent = parents[idx]
        if parent is None:
            continue
        parent_idx = name_to_idx.get(str(parent))
        if parent_idx in apic_children:
            apic_children[parent_idx].append(idx)

    start_dist: dict[int, float] = {origin_idx: 0.0}
    stack = [origin_idx]
    while stack:
        idx = stack.pop()
        base = start_dist[idx]
        parent_length = lengths[idx]
        for child_idx in apic_children[idx]:
            start_dist[child_idx] = base + parent_length * parentxs[child_idx]
            stack.append(child_idx)

    max_len = 0.0
    for idx in apic_sections:
        if apic_children[idx]:
            continue
        max_len = max(max_len, start_dist.get(idx, 0.0) + lengths[idx])
    if max_len <= 0.0:
        max_len = lengths[origin_idx]

    values_by_name: dict[str, list[float]] = {}
    for idx in apic_sections:
        nseg = nsegs[idx]
        seg_len = lengths[idx] / float(nseg)
        base = start_dist.get(idx, 0.0)
        values: list[float] = []
        for j in range(nseg):
            dist = base + seg_len * float(nseg if j == nseg - 1 else j + 0.5)
            dist_norm = dist / max_len if max_len > 0.0 else 0.0
            value = (exp_offset + exp_scale * math.exp(exp_rate * (dist_norm - exp_shift))) * reference_gbar
            values.append(value)
        values_by_name[names[idx]] = values
    return values_by_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the HL23 four-cell-type model through MIND_Sim.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--duration-ms", "--tstop", dest="tstop", type=float, default=50.0)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "mind_sim_hl23.npz",
    )
    args = parser.parse_args()

    mod_dir = DEFAULT_MOD_DIR.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    mech_lib = mod_dir / "x86_64" / "libcorenrnmech.so"
    mod_files = sorted(mod_dir.glob("*.mod"))
    if not mod_files:
        raise FileNotFoundError(f"no MOD files found in {mod_dir}")
    newest_mod = max(path.stat().st_mtime for path in mod_files)
    if mech_lib.exists() and mech_lib.stat().st_mtime >= newest_mod:
        print(f"[mech] cache hit: {mech_lib}", flush=True)
    else:
        print(f"[mech] compiling with mind_nrnivmodl: {mod_dir}", flush=True)
        subprocess.run(["mind_nrnivmodl", str(mod_dir)], check=True)

    print(
        "[build] "
        f"counts=PYR:{PYR_COUNT} SST:{SST_COUNT} PV:{PV_COUNT} VIP:{VIP_COUNT} "
        f"dt=0.025ms tstop={args.tstop}ms threads={args.threads} device=cpu",
        flush=True,
    )

    total_t0 = time.perf_counter()
    build_t0 = time.perf_counter()
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(int(args.threads))
    sim.set_dt(0.025)
    sim.celsius = 6.3
    sim.load_mech(str(mod_dir))

    pyr_section_list = ms.load_swc_section_list(str(ASSET_DIR / "HL23PYR.swc"))
    pyr_section_list.delete_label("axon")
    pyr_sections = list(pyr_section_list.to_list())
    pyr_sections = add_pyr_sst_stub_axon(pyr_sections)
    compute_section_nseg(pyr_sections, skip_label="myelin")
    pyr_apic_values = compute_apic_exp_distribution(
        pyr_sections,
        reference_gbar=0.000148,
        exp_offset=-0.8696,
        exp_rate=3.6161,
        exp_shift=0.0,
        exp_scale=2.0870,
    )
    pyr_apic_names = [str(sec.name) for sec in pyr_sections if str(sec.label) == "apic"]
    pyr_apic_indices = [idx for idx, name in enumerate(pyr_apic_names) if name in pyr_apic_values]
    pyr_apic_gbar = [pyr_apic_values[pyr_apic_names[idx]] for idx in pyr_apic_indices]

    sst_section_list = ms.load_swc_section_list(str(ASSET_DIR / "HL23SST.swc"))
    sst_section_list.delete_label("axon")
    sst_sections = list(sst_section_list.to_list())
    sst_sections = add_pyr_sst_stub_axon(sst_sections)
    compute_section_nseg(sst_sections, skip_label="myelin")

    pv_sections = list(ms.load_swc_section_list(str(ASSET_DIR / "HL23PV.swc")).to_list())
    compute_section_nseg(pv_sections)

    vip_section_list = ms.load_swc_section_list(str(ASSET_DIR / "HL23VIP.swc"))
    vip_section_list.delete_label("axon")
    vip_sections = list(vip_section_list.to_list())
    vip_sections = add_vip_stub_axon(vip_sections)
    compute_section_nseg(vip_sections)

    sim.build_morphology(
        [
            {"name": "PYR", "num_cells": PYR_COUNT, "sections": pyr_sections},
            {"name": "SST", "num_cells": SST_COUNT, "sections": sst_sections},
            {"name": "PV", "num_cells": PV_COUNT, "sections": pv_sections},
            {"name": "VIP", "num_cells": VIP_COUNT, "sections": vip_sections},
        ]
    )

    network = sim.network()
    spike_source_count = 0

    for cell in sim.population("PYR"):
        cell.v_init = -80.0
        cell.group("all").Ra = 100.0
        cell.group("all").cm = 1.0
        cell.group("dend").cm = 2.0
        cell.group("apic").cm = 2.0
        cell.group("myelin").cm = 0.02

        for label in ("soma", "dend", "axon", "apic"):
            cell.group(label).insert("pas", e=-80.0, g=0.0000954)
        cell.group("soma").insert("Ih", gbar=0.000148)
        cell.group("dend").insert("Ih", gbar=0.000000709)
        cell.group("axon").insert("Ih")
        cell.group("apic").insert(
            "Ih",
            gbar=cell.group("apic").segment_values(pyr_apic_indices, pyr_apic_gbar),
        )

        soma = cell.group("soma")
        soma.ena = 50.0
        soma.ek = -85.0
        soma.cai = 1.0e-4
        soma.cao = 2.0
        soma.insert("SK", gbar=0.000853)
        soma.insert("CaDynamics", gamma=0.0005, decay=20.0)
        soma.insert("Ca_LVA", gbar=0.00296)
        soma.insert("Ca_HVA", gbar=0.00155)
        soma.insert("K_T", gbar=0.0605)
        soma.insert("K_P", gbar=0.000208)
        soma.insert("Kv3_1", gbar=0.0424)
        soma.insert("NaTg", gbar=0.272, vshiftm=13.0, vshifth=15.0, slopem=7.0)
        soma.insert("Im", gbar=0.000306)

        axon = cell.group("axon")
        axon.ena = 50.0
        axon.ek = -85.0
        axon.cai = 1.0e-4
        axon.cao = 2.0
        axon.insert("SK", gbar=0.0145)
        axon.insert("Ca_LVA", gbar=0.0439)
        axon.insert("Ca_HVA", gbar=0.000306)
        axon.insert("K_T", gbar=0.0424)
        axon.insert("K_P", gbar=0.338)
        axon.insert("Nap", gbar=0.00842)
        axon.insert("Kv3_1", gbar=0.941)
        axon.insert("NaTg", gbar=1.38, vshifth=10.0, slopem=9.0)
        axon.insert("CaDynamics", gamma=0.0005, decay=226.0)
        axon.insert("Im", gbar=0.0)
        cell.group("soma")[0](0.5).insert(
            "IClamp",
            **{"del": 5.0, "dur": 80.0, "amp": 2.0},
        )
        seg = cell.group("soma")[0](0.5)
        network.register_spike_source(int(cell.gid), seg._ref_v, 0.0)
        spike_source_count += 1

    for cell in sim.population("SST"):
        cell.v_init = -80.0
        cell.group("all").Ra = 100.0
        cell.group("all").cm = 1.0
        cell.group("myelin").cm = 0.02

        for label in ("soma", "dend", "axon"):
            cell.group(label).insert("pas", e=-81.5, g=0.0000232)
        cell.group("soma").insert("Ih", gbar=0.0000431)
        cell.group("dend").insert("Ih", gbar=0.0000949)
        cell.group("axon").insert("Ih")

        soma = cell.group("soma")
        soma.ena = 50.0
        soma.ek = -85.0
        soma.cai = 1.0e-4
        soma.cao = 2.0
        soma.insert("SK", gbar=0.0)
        soma.insert("CaDynamics", gamma=0.0005, decay=465.0)
        soma.insert("Ca_LVA", gbar=0.00314)
        soma.insert("Ca_HVA", gbar=0.00355)
        soma.insert("K_T", gbar=0.0)
        soma.insert("K_P", gbar=0.0111)
        soma.insert("Kv3_1", gbar=0.871)
        soma.insert("NaTg", gbar=0.127, vshiftm=13.0, vshifth=15.0, slopem=7.0)
        soma.insert("Im", gbar=0.000158)

        axon = cell.group("axon")
        axon.ena = 50.0
        axon.ek = -85.0
        axon.cai = 1.0e-4
        axon.cao = 2.0
        axon.insert("SK", gbar=0.00113)
        axon.insert("Ca_LVA", gbar=0.0627)
        axon.insert("Ca_HVA", gbar=0.00145)
        axon.insert("K_T", gbar=0.023)
        axon.insert("K_P", gbar=0.0295)
        axon.insert("Nap", gbar=0.000444)
        axon.insert("Kv3_1", gbar=0.984)
        axon.insert("NaTg", gbar=0.343, vshifth=10.0, slopem=9.0)
        axon.insert("CaDynamics", gamma=0.0005, decay=469.0)
        axon.insert("Im", gbar=0.000317)
        cell.group("soma")[0](0.5).insert(
            "IClamp",
            **{"del": 5.0, "dur": 80.0, "amp": 2.0},
        )
        seg = cell.group("soma")[0](0.5)
        network.register_spike_source(int(cell.gid), seg._ref_v, 0.0)
        spike_source_count += 1

    for cell in sim.population("PV"):
        cell.v_init = -80.0
        allsec = cell.group("all")
        allsec.Ra = 100.0
        allsec.cm = 2.0
        allsec.insert("pas", e=-83.92924122901199, g=0.00011830111773572024)
        allsec.insert("Ih", gbar=2.7671764064314368e-05)

        soma = cell.group("soma")
        soma.ena = 50.0
        soma.ek = -85.0
        soma.cai = 1.0e-4
        soma.cao = 2.0
        soma.insert(
            "NaTg",
            gbar=0.49958525078702043,
            vshiftm=0.0,
            vshifth=10.0,
            slopem=9.0,
            slopeh=6.0,
        )
        soma.insert("Nap", gbar=0.008795461417521086)
        soma.insert("K_P", gbar=9.606092478937705e-06)
        soma.insert("K_T", gbar=0.0011701702607527396)
        soma.insert("Kv3_1", gbar=2.9921080101237565)
        soma.insert("Im", gbar=0.04215865946497755)
        soma.insert("SK", gbar=3.7265770903193036e-06)
        soma.insert("Ca_HVA", gbar=0.00017953651378188165)
        soma.insert("Ca_LVA", gbar=0.09250008555398015)
        soma.insert("CaDynamics", gamma=0.0005, decay=531.0255920416845)

        axon = cell.group("axon")
        axon.ena = 50.0
        axon.ek = -85.0
        axon.cai = 1.0e-4
        axon.cao = 2.0
        axon.insert(
            "NaTg",
            gbar=0.10914576408883477,
            vshiftm=0.0,
            vshifth=10.0,
            slopem=9.0,
            slopeh=6.0,
        )
        axon.insert("Nap", gbar=0.001200899579358837)
        axon.insert("K_P", gbar=0.6854776593761795)
        axon.insert("K_T", gbar=0.07603372775662909)
        axon.insert("Kv3_1", gbar=2.988867483754507)
        axon.insert("Im", gbar=0.029587905136596156)
        axon.insert("SK", gbar=0.5121938998281017)
        axon.insert("Ca_HVA", gbar=0.002961469262723619)
        axon.insert("Ca_LVA", gbar=5.9457835817342756e-05)
        axon.insert("CaDynamics", gamma=0.0005, decay=163.03538024059918)
        cell.group("soma")[0](0.5).insert(
            "IClamp",
            **{"del": 5.0, "dur": 80.0, "amp": 2.0},
        )
        seg = cell.group("soma")[0](0.5)
        network.register_spike_source(int(cell.gid), seg._ref_v, 0.0)
        spike_source_count += 1

    for cell in sim.population("VIP"):
        cell.v_init = -80.0
        allsec = cell.group("all")
        allsec.Ra = 100.0
        allsec.cm = 2.0
        allsec.insert("pas", e=-79.74132024971513, g=2.5756438955642182e-05)
        allsec.insert("Ih", gbar=4.274951616063423e-05)

        soma = cell.group("soma")
        soma.ena = 50.0
        soma.ek = -85.0
        soma.cai = 1.0e-4
        soma.cao = 2.0
        soma.insert(
            "NaTg",
            gbar=0.11491205828369114,
            vshiftm=13.0,
            vshifth=15.0,
            slopem=7.0,
            slopeh=6.0,
        )
        soma.insert("Nap", gbar=0.0001895305240694194)
        soma.insert("K_P", gbar=0.0009925418924114282)
        soma.insert("K_T", gbar=0.009051981253674193)
        soma.insert("Kv3_1", gbar=0.31215653649208114)
        soma.insert("SK", gbar=0.1655502166633749)
        soma.insert("Im", gbar=0.0003679378262289559)
        soma.insert("Ca_HVA", gbar=4.384846294634834e-05)
        soma.insert("Ca_LVA", gbar=0.0034472458995879864)
        soma.insert("CaDynamics", gamma=0.0005, decay=25.159166441555044)

        axon = cell.group("axon")
        axon.ena = 50.0
        axon.ek = -85.0
        axon.cai = 1.0e-4
        axon.cao = 2.0
        axon.insert(
            "NaTg",
            gbar=0.20112200814143477,
            vshiftm=0.0,
            vshifth=10.0,
            slopem=9.0,
            slopeh=6.0,
        )
        axon.insert("Nap", gbar=0.0006248906854665301)
        axon.insert("K_P", gbar=0.26489876414660096)
        axon.insert("K_T", gbar=0.014364427062274185)
        axon.insert("Kv3_1", gbar=0.0011201608191112877)
        axon.insert("SK", gbar=0.7027792087501376)
        axon.insert("Im", gbar=0.00013891465461042372)
        axon.insert("Ca_HVA", gbar=2.819397237794038e-05)
        axon.insert("Ca_LVA", gbar=0.010354001513952075)
        axon.insert("CaDynamics", gamma=0.0005, decay=75.78875619470153)
        cell.group("soma")[0](0.5).insert(
            "IClamp",
            **{"del": 5.0, "dur": 80.0, "amp": 2.0},
        )
        seg = cell.group("soma")[0](0.5)
        network.register_spike_source(int(cell.gid), seg._ref_v, 0.0)
        spike_source_count += 1

    sim.build_microcircuit()
    v_vectors: dict[int, Any] = {}
    for name in ("PYR", "SST", "PV", "VIP"):
        population = sim.population(name)
        for local_index in RECORD_LOCAL_CELLS:
            if local_index >= len(population):
                print(f"[record] skip {name}[{local_index}] because count={len(population)}", flush=True)
                continue
            cell = population[local_index]
            v_vectors[int(cell.gid)] = ms.Vector().record(cell.group("soma")[0](0.5)._ref_v)
    if not v_vectors:
        raise RuntimeError("no soma vectors selected for recording")
    t_vec = ms.Vector().record(sim._ref_t)
    build_t1 = time.perf_counter()

    init_t0 = time.perf_counter()
    sim.finitialize(-80.0)
    init_t1 = time.perf_counter()

    run_t0 = time.perf_counter()
    sim.run(float(args.tstop))
    run_t1 = time.perf_counter()
    run_s = run_t1 - run_t0
    total_t1 = time.perf_counter()

    build_s = build_t1 - build_t0
    init_s = init_t1 - init_t0
    pre_run_s = build_s + init_s
    sum_s = total_t1 - total_t0
    print(
        "[timing] "
        f"build_s={build_s:.6f} finitialize_s={init_s:.6f} "
        f"pre_run_s={pre_run_s:.6f} run_s={run_s:.6f} sum_s={sum_s:.6f}",
        flush=True,
    )
    spike_times = np.asarray(sim.spike_times(), dtype=float)
    spike_gids = np.asarray(sim.spike_gids(), dtype=int)
    if spike_times.size:
        order = np.lexsort((spike_gids, spike_times))
        spike_times = spike_times[order]
        spike_gids = spike_gids[order]
    first_spike = float(spike_times[0]) if spike_times.size else float("nan")
    print(
        f"[spikes] count={spike_times.size} first_ms={first_spike:.6f} "
        "threshold_mV=0.0",
        flush=True,
    )

    payload: dict[str, Any] = {
        "engine": "mind_sim",
        "dt": 0.025,
        "tstop": float(args.tstop),
        "stim_delay": 5.0,
        "stim_dur": 80.0,
        "stim_amp": 2.0,
        "spike_threshold": 0.0,
        "build_s": build_s,
        "finitialize_s": init_s,
        "pre_run_s": pre_run_s,
        "run_s": run_s,
        "sum_s": sum_s,
        "num_cells_pyr": PYR_COUNT,
        "num_cells_sst": SST_COUNT,
        "num_cells_pv": PV_COUNT,
        "num_cells_vip": VIP_COUNT,
        "spike_source_count": spike_source_count,
        "spike_times": spike_times,
        "spike_gids": spike_gids,
        "spike_count": int(spike_times.size),
        "t": np.asarray(t_vec.to_python(), dtype=float),
        "record_gids": np.asarray(sorted(v_vectors), dtype=int),
    }
    for gid, vector in sorted(v_vectors.items()):
        payload[str(gid)] = np.asarray(vector.to_python(), dtype=float)
    np.savez(output, **payload)
    print(f"[output] saved traces: {output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
