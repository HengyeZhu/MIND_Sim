#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np


POP_NAMES = ("PYR", "SST", "PV", "VIP")

_REF_POP_SIZES = {
    "PV": 70,
    "PYR": 800,
    "SST": 50,
    "VIP": 80,
}

_REF_CONN_COUNTS = {
    ("PV", "PV"): 26175,
    ("PV", "PYR"): 89216,
    ("PV", "SST"): 2896,
    ("PV", "VIP"): 1295,
    ("PYR", "PV"): 41360,
    ("PYR", "PYR"): 287706,
    ("PYR", "SST"): 61952,
    ("PYR", "VIP"): 23592,
    ("SST", "PV"): 8632,
    ("SST", "PYR"): 90708,
    ("SST", "SST"): 1236,
    ("SST", "VIP"): 1225,
    ("VIP", "PV"): 6248,
    ("VIP", "PYR"): 0,
    ("VIP", "SST"): 12780,
    ("VIP", "VIP"): 2352,
}


def _effective_lambda(pre: str, post: str) -> float:
    count = float(_REF_CONN_COUNTS[(pre, post)])
    denom = float(_REF_POP_SIZES[pre] * _REF_POP_SIZES[post])
    return count / denom if denom > 0.0 else 0.0


EFFECTIVE_SYNAPSES_PER_PAIR = {
    key: _effective_lambda(*key) for key in _REF_CONN_COUNTS
}


@dataclass(frozen=True)
class SynapseSpec:
    kind: str
    gbase_uS: float
    fast_tau_r_ms: float
    fast_tau_d_ms: float
    erev_mV: float
    slow_tau_r_ms: float = 0.0
    slow_tau_d_ms: float = 0.0
    slow_weight_ratio: float = 0.0


SYNAPSE_SPECS: Dict[Tuple[str, str], SynapseSpec] = {
    ("PV", "PV"): SynapseSpec("inh", 0.00033, 1.0, 10.0, -80.0),
    ("PV", "PYR"): SynapseSpec("inh", 0.00291, 1.0, 10.0, -80.0),
    ("PV", "SST"): SynapseSpec("inh", 0.00033, 1.0, 10.0, -80.0),
    ("PV", "VIP"): SynapseSpec("inh", 0.00034, 1.0, 10.0, -80.0),
    ("PYR", "PV"): SynapseSpec("exc", 0.000337, 0.3, 3.0, 0.0, slow_tau_r_ms=2.0, slow_tau_d_ms=65.0, slow_weight_ratio=1.0),
    ("PYR", "PYR"): SynapseSpec("exc", 0.0002482, 0.3, 3.0, 0.0, slow_tau_r_ms=2.0, slow_tau_d_ms=65.0, slow_weight_ratio=1.0),
    ("PYR", "SST"): SynapseSpec("exc", 0.00038, 0.3, 3.0, 0.0, slow_tau_r_ms=2.0, slow_tau_d_ms=65.0, slow_weight_ratio=1.0),
    ("PYR", "VIP"): SynapseSpec("exc", 0.00031, 0.3, 3.0, 0.0, slow_tau_r_ms=2.0, slow_tau_d_ms=65.0, slow_weight_ratio=1.0),
    ("SST", "PV"): SynapseSpec("inh", 0.00033, 1.0, 10.0, -80.0),
    ("SST", "PYR"): SynapseSpec("inh", 0.00124, 1.0, 10.0, -80.0),
    ("SST", "SST"): SynapseSpec("inh", 0.00034, 1.0, 10.0, -80.0),
    ("SST", "VIP"): SynapseSpec("inh", 0.00046, 1.0, 10.0, -80.0),
    ("VIP", "PV"): SynapseSpec("inh", 0.00034, 1.0, 10.0, -80.0),
    ("VIP", "PYR"): SynapseSpec("inh", 0.0, 1.0, 10.0, -80.0),
    ("VIP", "SST"): SynapseSpec("inh", 0.00036, 1.0, 10.0, -80.0),
    ("VIP", "VIP"): SynapseSpec("inh", 0.00034, 1.0, 10.0, -80.0),
}


def iter_sampled_contacts(
    pop_counts: Mapping[str, int],
    *,
    rng: np.random.Generator,
    scale: float = 1.0,
    allow_autapse: bool = True,
) -> Iterator[Tuple[str, int, str, int, SynapseSpec]]:
    scale_f = max(0.0, float(scale))
    for pre in POP_NAMES:
        n_pre = int(pop_counts.get(pre, 0))
        if n_pre <= 0:
            continue
        for post in POP_NAMES:
            n_post = int(pop_counts.get(post, 0))
            if n_post <= 0:
                continue
            spec = SYNAPSE_SPECS[(pre, post)]
            lam = float(EFFECTIVE_SYNAPSES_PER_PAIR[(pre, post)]) * scale_f
            if lam <= 0.0 or spec.gbase_uS <= 0.0:
                continue
            for pre_idx in range(n_pre):
                for post_idx in range(n_post):
                    if (not allow_autapse) and pre == post and pre_idx == post_idx:
                        continue
                    k = int(rng.poisson(lam))
                    for _ in range(k):
                        yield pre, pre_idx, post, post_idx, spec


def iter_sampled_source_gid_contacts(
    pop_cells: Mapping[str, Sequence[Any]],
    pop_soma_locs: Mapping[str, Sequence[Any]],
    *,
    rng: np.random.Generator,
    scale: float = 1.0,
    allow_autapse: bool = True,
) -> Iterator[Tuple[int, Any, SynapseSpec]]:
    pop_counts = {name: len(pop_cells.get(name, ())) for name in POP_NAMES}
    for pre_name, pre_local, post_name, post_local, spec in iter_sampled_contacts(
        pop_counts,
        rng=rng,
        scale=scale,
        allow_autapse=allow_autapse,
    ):
        pre_cells = pop_cells.get(pre_name, ())
        post_cells = pop_cells.get(post_name, ())
        post_locs = pop_soma_locs.get(post_name, ())
        if pre_local < 0 or pre_local >= len(pre_cells):
            raise RuntimeError(f"presyn local index out of range for {pre_name}: {pre_local}")
        if post_local < 0 or post_local >= len(post_cells):
            raise RuntimeError(f"postsyn local index out of range for {post_name}: {post_local}")
        if post_local >= len(post_locs):
            raise RuntimeError(f"postsyn soma loc index out of range for {post_name}: {post_local}")
        pre_cell = pre_cells[pre_local]
        yield int(pre_cell.gid), post_locs[post_local], spec


def iter_sampled_gid_contacts(
    pop_cells: Mapping[str, Sequence[Any]],
    pop_soma_locs: Mapping[str, Sequence[Any]],
    *,
    rng: np.random.Generator,
    scale: float = 1.0,
    allow_autapse: bool = True,
) -> Iterator[Tuple[int, Any, SynapseSpec]]:
    yield from iter_sampled_source_gid_contacts(
        pop_cells,
        pop_soma_locs,
        rng=rng,
        scale=scale,
        allow_autapse=allow_autapse,
    )


def estimate_contact_count(pop_counts: Mapping[str, int], *, scale: float = 1.0) -> float:
    total = 0.0
    scale_f = max(0.0, float(scale))
    for pre in POP_NAMES:
        n_pre = float(int(pop_counts.get(pre, 0)))
        if n_pre <= 0.0:
            continue
        for post in POP_NAMES:
            n_post = float(int(pop_counts.get(post, 0)))
            if n_post <= 0.0:
                continue
            total += n_pre * n_post * float(EFFECTIVE_SYNAPSES_PER_PAIR[(pre, post)]) * scale_f
    return total


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _sort_spikes(times_ms: np.ndarray, gids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times_ms.size == 0:
        return np.asarray(times_ms, dtype=float), np.asarray(gids, dtype=np.int32)
    order = np.argsort(times_ms, kind="mergesort")
    return np.asarray(times_ms[order], dtype=float), np.asarray(gids[order], dtype=np.int32)


def _collect_spikes_from_sim(sim: object, num_total: int) -> tuple[np.ndarray, np.ndarray]:
    times_parts: list[np.ndarray] = []
    gids_parts: list[np.ndarray] = []
    for gid in range(int(num_total)):
        spikes = np.asarray(sim.get_spk_by_gid(int(gid)), dtype=float)
        if spikes.size == 0:
            continue
        times_parts.append(spikes)
        gids_parts.append(np.full(spikes.shape, int(gid), dtype=np.int32))
    if not times_parts:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=np.int32)
    times = np.concatenate(times_parts)
    gids = np.concatenate(gids_parts)
    return _sort_spikes(times, gids)


def find_tutorial_root(start: Path) -> Path:
    here = start.resolve()
    if here.is_file():
        here = here.parent
    for root in (here, *here.parents):
        has_layout = (root / "neurons_tools").is_dir()
        has_markers = (
            (root / "result").is_dir()
            or (root / "plot_tools").is_dir()
            or (root / "tutorial_tools").is_dir()
            or (root / "neurons.sh").is_file()
        )
        if has_layout and has_markers:
            return root
    raise RuntimeError(f"tutorial root not found from {start}")


def pick_existing_dir(candidates: List[Path], *, label: str) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise RuntimeError(f"{label} not found; tried: {tried}")


def resolve_hl23_asset_dir(repo_root: Path) -> Path:
    env_dir = os.environ.get("MIND_SIM_HL23_ASSET_DIR", "").strip()
    candidates: List[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append((repo_root / "examples" / "hl23multi" / "hl23_minimal").resolve())
    return pick_existing_dir(candidates, label="HL23 asset directory")


def resolve_hl23_mod_dir(asset_dir: Path) -> Path:
    env_dir = os.environ.get("MIND_SIM_HL23_MOD_DIR", "").strip()
    candidates: List[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append((asset_dir / "mod").resolve())
    return pick_existing_dir(candidates, label="HL23 NMODL metadata directory")


def resolve_hl23_mechanism_path(mod_dir: Path) -> Path:
    env_path = os.environ.get("MIND_SIM_HL23_MECH_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"HL23 mechanism path does not exist: {path}")
        return path

    mod_sources = sorted(mod_dir.glob("*.mod"))
    if not mod_sources:
        raise RuntimeError(f"HL23 MOD directory contains no .mod files: {mod_dir}")
    return mod_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = REPO_ROOT

import mind_sim as ms


MICROCIRCUIT_DIR = resolve_hl23_asset_dir(REPO_ROOT)
HL23_MOD_DIR = resolve_hl23_mod_dir(MICROCIRCUIT_DIR)
HL23_MECH_PATH = resolve_hl23_mechanism_path(HL23_MOD_DIR)

DEFAULT_PYR_SWC = MICROCIRCUIT_DIR / "HL23PYR.swc"
DEFAULT_SST_SWC = MICROCIRCUIT_DIR / "HL23SST.swc"
DEFAULT_PV_SWC = MICROCIRCUIT_DIR / "HL23PV.swc"
DEFAULT_VIP_SWC = MICROCIRCUIT_DIR / "HL23VIP.swc"

PYR_SWC_FILE = DEFAULT_PYR_SWC
SST_SWC_FILE = DEFAULT_SST_SWC
PV_SWC_FILE = DEFAULT_PV_SWC
VIP_SWC_FILE = DEFAULT_VIP_SWC

NUM_CELLS_ALL = os.environ.get("MIND_SIM_HL23_NUM_CELLS", "").strip()
NUM_CELLS_PYR = _env_int("MIND_SIM_HL23_NUM_CELLS_PYR", int(NUM_CELLS_ALL) if NUM_CELLS_ALL else 100)
NUM_CELLS_SST = _env_int("MIND_SIM_HL23_NUM_CELLS_SST", int(NUM_CELLS_ALL) if NUM_CELLS_ALL else 100)
NUM_CELLS_PV = _env_int("MIND_SIM_HL23_NUM_CELLS_PV", int(NUM_CELLS_ALL) if NUM_CELLS_ALL else 100)
NUM_CELLS_VIP = _env_int("MIND_SIM_HL23_NUM_CELLS_VIP", int(NUM_CELLS_ALL) if NUM_CELLS_ALL else 100)

DT_MS = _env_float("MIND_SIM_HL23_DT_MS", 0.025)
TSTOP_MS = _env_float("MIND_SIM_HL23_TSTOP_MS", 600.0)
CELSIUS_C = _env_float("MIND_SIM_HL23_CELSIUS_C", 6.3)
TARGET_SEG_LENGTH_UM = _env_float("MIND_SIM_HL23_TARGET_SEG_LENGTH_UM", 40.0)

STIM_DELAY_MS = _env_float("MIND_SIM_HL23_STIM_DELAY_MS", 200.0)
STIM_DUR_MS = _env_float("MIND_SIM_HL23_STIM_DUR_MS", 150.0)
STIM_AMP_NA = _env_float("MIND_SIM_HL23_STIM_AMP_NA", 2.0)
ENABLE_SYNAPSES = _env_bool("MIND_SIM_HL23_ENABLE_SYNAPSES", True)
ALLOW_AUTAPSE = _env_bool("MIND_SIM_HL23_ALLOW_AUTAPSE", True)
SYNAPSE_SEED = _env_int("MIND_SIM_HL23_CONN_SEED", 12345)
SYNAPSE_SCALE = _env_float("MIND_SIM_HL23_CONN_SCALE", 1.0)
SYNAPSE_DELAY_MS = _env_float("MIND_SIM_HL23_CONN_DELAY_MS", 0.5)
SPIKE_THRESHOLD_MV = _env_float("MIND_SIM_HL23_SPIKE_THRESHOLD_MV", 0.0)

DEVICE = os.environ.get("MIND_SIM_HL23_DEVICE", "cpu").strip() or "cpu"
V_INIT_MV = -80.0
SAVE_OUTPUT = True
SAVE_TRACES = _env_path(
    "MIND_SIM_HL23_SAVE_TRACES",
    (REPO_ROOT / "result" / "hl23_mind_sim.npz").resolve(),
)


def compute_section_nseg(
    sections: list,
    target_seg_length_um: float,
    *,
    skip_label: str | None = None,
) -> None:
    for sec in sections:
        if skip_label is not None and str(getattr(sec, "label", "")) == str(skip_label):
            continue
        length_um = float(sec.L_um)
        base = int(length_um / float(target_seg_length_um))
        sec.nseg = max(1, 1 + 2 * base)


def _section_name(section: object) -> str:
    return str(getattr(section, "name"))


def _section_label(section: object) -> str:
    return str(getattr(section, "label", ""))


def _section_parent_name(section: object) -> str | None:
    parent = getattr(section, "parent", None)
    if parent is None:
        return None
    return str(parent)


def compute_apic_ih_distribution(
    sections: list,
    *,
    apic_label: str,
    soma_gbar: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
) -> list[list[float]]:
    if not sections:
        return []

    apic_sections = [i for i, sec in enumerate(sections) if _section_label(sec) == str(apic_label)]
    if not apic_sections:
        return []

    distance_layout = ms.build_section_distance_layout(sections)
    origin_name = _section_name(sections[apic_sections[0]])
    apic_children = {idx: [] for idx in apic_sections}
    name_to_idx = {_section_name(sec): i for i, sec in enumerate(sections)}
    for idx in apic_sections:
        parent_name = _section_parent_name(sections[idx])
        if parent_name is None:
            continue
        parent_idx = name_to_idx.get(parent_name)
        if parent_idx in apic_children:
            apic_children[parent_idx].append(idx)

    max_len = 0.0
    for idx in apic_sections:
        if apic_children[idx]:
            continue
        dist_end = float(distance_layout.distance(origin_name, 0.0, _section_name(sections[idx]), 1.0))
        if not math.isfinite(dist_end):
            raise RuntimeError(f"failed to resolve apical distance for section {_section_name(sections[idx])!r}")
        if dist_end > max_len:
            max_len = dist_end
    if max_len <= 0.0:
        raise RuntimeError("apical Ih distribution resolved a non-positive longest branch distance")

    out: list[list[float]] = []
    for idx in apic_sections:
        sec = sections[idx]
        nseg = int(sec.nseg)
        if nseg <= 0:
            continue
        values: list[float] = []
        for j in range(nseg):
            x = 1.0 if j == nseg - 1 else (float(j) + 0.5) / float(nseg)
            dist_um = float(distance_layout.distance(origin_name, 0.0, _section_name(sec), x))
            if not math.isfinite(dist_um):
                raise RuntimeError(f"failed to resolve apical segment distance for section {_section_name(sec)!r}")
            dist_norm = dist_um / max_len if max_len > 0.0 else 0.0
            value = (p1 + p4 * math.exp(p2 * (dist_norm - p3))) * float(soma_gbar)
            values.append(value)
        out.append(values)
    return out


def main() -> int:
    pyr_swc = Path(PYR_SWC_FILE).resolve()
    sst_swc = Path(SST_SWC_FILE).resolve()
    pv_swc = Path(PV_SWC_FILE).resolve()
    vip_swc = Path(VIP_SWC_FILE).resolve()
    save_traces = Path(SAVE_TRACES).resolve()
    save_traces.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[build] "
        f"pyr_swc={pyr_swc} pyr_cells={NUM_CELLS_PYR} | "
        f"sst_swc={sst_swc} sst_cells={NUM_CELLS_SST} | "
        f"pv_swc={pv_swc} pv_cells={NUM_CELLS_PV} | "
        f"vip_swc={vip_swc} vip_cells={NUM_CELLS_VIP} | "
        f"dt={DT_MS}ms tstop={TSTOP_MS}ms v_init={V_INIT_MV}mV "
        f"target_seg_length_um={TARGET_SEG_LENGTH_UM} celsius={CELSIUS_C}",
        flush=True,
    )
    build_t0 = time.perf_counter()

    sim = ms.Sim()
    sim.set_spike_output_enabled(False)
    sim.set_device(DEVICE)
    sim.set_dt(float(DT_MS))
    if int(sim.load_mech_metadata(str(HL23_MECH_PATH))) != 0:
        raise RuntimeError(f"failed to load HL23 MOD mechanisms: {HL23_MECH_PATH}")
    print(f"Loaded HL23 MOD mechanisms: {HL23_MECH_PATH}", flush=True)

    morph_biophys_t0 = time.perf_counter()
    morph_templates: List[dict] = []
    pyr_apic_ih_gbar: list[list[float]] = []

    if NUM_CELLS_PYR > 0:
        pyr_sections = ms.load_swc_section_list(str(pyr_swc))
        pyr_sections.delete_label("axon")

        axon0 = ms.section(name="axon_stub_0", label="axon")
        axon0.connect("soma_0", 0.5)
        axon0.L_um = 20.0
        axon0.nseg = 1
        axon0.diam_um = 0.5 * (3.0 + 1.75)

        axon1 = ms.section(name="axon_stub_1", label="axon")
        axon1.connect(axon0, 1.0)
        axon1.L_um = 30.0
        axon1.nseg = 1
        axon1.diam_um = 0.5 * (1.75 + 1.0)

        myelin = ms.section(name="myelin_stub_0", label="myelin")
        myelin.connect(axon1, 1.0)
        myelin.L_um = 100.0
        myelin.nseg = 1
        myelin.diam_um = 500.0

        pyr_sections.append(axon0)
        pyr_sections.append(axon1)
        pyr_sections.append(myelin)
        compute_section_nseg(pyr_sections, float(TARGET_SEG_LENGTH_UM), skip_label="myelin")
        pyr_apic_ih_gbar = compute_apic_ih_distribution(
            pyr_sections,
            apic_label="apic",
            soma_gbar=0.000148,
            p1=-0.8696,
            p2=3.6161,
            p3=0.0,
            p4=2.0870,
        )
        morph_templates.append(
            {
                "name": "PYR",
                "num_cells": int(NUM_CELLS_PYR),
                "sections": pyr_sections,
            }
        )

    if NUM_CELLS_SST > 0:
        sst_sections = ms.load_swc_section_list(str(sst_swc))
        sst_sections.delete_label("axon")

        axon0 = ms.section(name="axon_stub_0", label="axon")
        axon0.connect("soma_0", 0.5)
        axon0.L_um = 20.0
        axon0.nseg = 1
        axon0.diam_um = 0.5 * (3.0 + 1.75)

        axon1 = ms.section(name="axon_stub_1", label="axon")
        axon1.connect(axon0, 1.0)
        axon1.L_um = 30.0
        axon1.nseg = 1
        axon1.diam_um = 0.5 * (1.75 + 1.0)

        myelin = ms.section(name="myelin_stub_0", label="myelin")
        myelin.connect(axon1, 1.0)
        myelin.L_um = 100.0
        myelin.nseg = 1
        myelin.diam_um = 500.0

        sst_sections.append(axon0)
        sst_sections.append(axon1)
        sst_sections.append(myelin)
        compute_section_nseg(sst_sections, float(TARGET_SEG_LENGTH_UM), skip_label="myelin")
        morph_templates.append(
            {
                "name": "SST",
                "num_cells": int(NUM_CELLS_SST),
                "sections": sst_sections,
            }
        )

    if NUM_CELLS_PV > 0:
        pv_sections = ms.load_swc_section_list(str(pv_swc))
        compute_section_nseg(pv_sections, float(TARGET_SEG_LENGTH_UM))
        morph_templates.append(
            {
                "name": "PV",
                "num_cells": int(NUM_CELLS_PV),
                "sections": pv_sections,
            }
        )

    if NUM_CELLS_VIP > 0:
        vip_sections = ms.load_swc_section_list(str(vip_swc))
        vip_sections.delete_label("axon")

        axon0 = ms.section(name="vip_axon_stub_0", label="axon")
        axon0.connect("soma_0", 1.0)
        axon0.L_um = 30.0
        axon0.diam_um = 1.1062632630369478

        axon1 = ms.section(name="vip_axon_stub_1", label="axon")
        axon1.connect(axon0, 1.0)
        axon1.L_um = 30.0
        axon1.diam_um = 0.3140589549560489

        vip_sections.append(axon0)
        vip_sections.append(axon1)
        compute_section_nseg(vip_sections, float(TARGET_SEG_LENGTH_UM))
        morph_templates.append(
            {
                "name": "VIP",
                "num_cells": int(NUM_CELLS_VIP),
                "sections": vip_sections,
            }
        )

    if not morph_templates:
        raise RuntimeError("No morphology templates generated.")

    print(f"[MIND_Sim] sim.build_morphology (device={DEVICE}) ...", flush=True)
    sim.build_morphology(morph_templates)
    sim.celsius = float(CELSIUS_C)
    network = sim.network()

    pyr_population = None
    sst_population = None
    pv_population = None
    vip_population = None
    if NUM_CELLS_PYR > 0:
        population = sim.population("PYR")
        pyr_population = population
        for cell in population:
            cell.v_init = float(V_INIT_MV)
            pop_soma = cell.group("soma")
            pop_dend = cell.group("dend")
            pop_axon = cell.group("axon")
            pop_apic = cell.group("apic")
            pop_all = cell.group("all")
            pop_myelin = cell.group("myelin")

            pop_all.Ra = 100.0
            pop_all.cm = 1.0
            pop_dend.cm = 2.0
            pop_apic.cm = 2.0
            pop_myelin.cm = 0.02
            pop_soma.insert("pas", e=-80.0, g=0.0000954)
            pop_dend.insert("pas", e=-80.0, g=0.0000954)
            pop_axon.insert("pas", e=-80.0, g=0.0000954)
            pop_apic.insert("pas", e=-80.0, g=0.0000954)

            pop_soma.insert("Ih", gbar=0.000148)
            pop_dend.insert("Ih", gbar=0.000000709)
            pop_axon.insert("Ih")
            if pyr_apic_ih_gbar:
                pop_apic.insert(
                    "Ih",
                    gbar=pop_apic.segment_values(
                        list(range(len(pyr_apic_ih_gbar))),
                        pyr_apic_ih_gbar,
                    ),
                )

            pop_soma.insert("SK", gbar=0.000853, ek=-85.0)
            pop_soma.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=20.0,
            )
            pop_soma.insert("Ca_LVA", gbar=0.00296)
            pop_soma.insert("Ca_HVA", gbar=0.00155)
            pop_soma.insert("K_T", gbar=0.0605)
            pop_soma.insert("K_P", gbar=0.000208)
            pop_soma.insert("Kv3_1", gbar=0.0424)
            pop_soma.insert("NaTg", gbar=0.272, vshiftm=13.0, vshifth=15.0, slopem=7.0, ena=50.0)
            pop_soma.insert("Im", gbar=0.000306)

            pop_axon.insert("SK", gbar=0.0145, ek=-85.0)
            pop_axon.insert("Ca_LVA", gbar=0.0439)
            pop_axon.insert("Ca_HVA", gbar=0.000306)
            pop_axon.insert("K_T", gbar=0.0424)
            pop_axon.insert("K_P", gbar=0.338)
            pop_axon.insert("Nap", gbar=0.00842, ena=50.0)
            pop_axon.insert("Kv3_1", gbar=0.941)
            pop_axon.insert("NaTg", gbar=1.38, vshifth=10.0, slopem=9.0)
            pop_axon.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=226.0,
            )
            pop_axon.insert("Im", gbar=0.0)

    if NUM_CELLS_SST > 0:
        population = sim.population("SST")
        sst_population = population
        for cell in population:
            cell.v_init = float(V_INIT_MV)
            pop_soma = cell.group("soma")
            pop_dend = cell.group("dend")
            pop_axon = cell.group("axon")
            pop_all = cell.group("all")
            pop_myelin = cell.group("myelin")

            pop_all.Ra = 100.0
            pop_all.cm = 1.0
            pop_myelin.cm = 0.02
            pop_soma.insert("pas", e=-81.5, g=0.0000232)
            pop_dend.insert("pas", e=-81.5, g=0.0000232)
            pop_axon.insert("pas", e=-81.5, g=0.0000232)

            pop_soma.insert("Ih", gbar=0.0000431)
            pop_dend.insert("Ih", gbar=0.0000949)
            pop_axon.insert("Ih")

            pop_soma.insert("SK", gbar=0.0, ek=-85.0)
            pop_soma.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=465.0,
            )
            pop_soma.insert("Ca_LVA", gbar=0.00314)
            pop_soma.insert("Ca_HVA", gbar=0.00355)
            pop_soma.insert("K_T", gbar=0.0)
            pop_soma.insert("K_P", gbar=0.0111)
            pop_soma.insert("Kv3_1", gbar=0.871)
            pop_soma.insert("NaTg", gbar=0.127, vshiftm=13.0, vshifth=15.0, slopem=7.0, ena=50.0)
            pop_soma.insert("Im", gbar=0.000158)

            pop_axon.insert("SK", gbar=0.00113, ek=-85.0)
            pop_axon.insert("Ca_LVA", gbar=0.0627)
            pop_axon.insert("Ca_HVA", gbar=0.00145)
            pop_axon.insert("K_T", gbar=0.023)
            pop_axon.insert("K_P", gbar=0.0295)
            pop_axon.insert("Nap", gbar=0.000444, ena=50.0)
            pop_axon.insert("Kv3_1", gbar=0.984)
            pop_axon.insert("NaTg", gbar=0.343, vshifth=10.0, slopem=9.0)
            pop_axon.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=469.0,
            )
            pop_axon.insert("Im", gbar=0.000317)

    if NUM_CELLS_PV > 0:
        population = sim.population("PV")
        pv_population = population
        for cell in population:
            cell.v_init = float(V_INIT_MV)
            pop_all = cell.group("all")
            pop_soma = cell.group("soma")
            pop_axon = cell.group("axon")

            pop_all.Ra = 100.0
            pop_all.cm = 2.0
            pop_all.insert("pas", e=-83.92924122901199, g=0.00011830111773572024)
            pop_all.insert("Ih", gbar=2.7671764064314368e-05)

            pop_soma.insert(
                "NaTg",
                gbar=0.49958525078702043,
                ena=50.0,
                vshiftm=0.0,
                vshifth=10.0,
                slopem=9.0,
                slopeh=6.0,
            )
            pop_soma.insert("Nap", gbar=0.008795461417521086)
            pop_soma.insert("K_P", gbar=9.606092478937705e-06, ek=-85.0)
            pop_soma.insert("K_T", gbar=0.0011701702607527396)
            pop_soma.insert("Kv3_1", gbar=2.9921080101237565, vshift=0.0)
            pop_soma.insert("Im", gbar=0.04215865946497755)
            pop_soma.insert("SK", gbar=3.7265770903193036e-06)
            pop_soma.insert("Ca_HVA", gbar=0.00017953651378188165)
            pop_soma.insert("Ca_LVA", gbar=0.09250008555398015)
            pop_soma.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=531.0255920416845,
            )

            pop_axon.insert(
                "NaTg",
                gbar=0.10914576408883477,
                ena=50.0,
                vshiftm=0.0,
                vshifth=10.0,
                slopem=9.0,
                slopeh=6.0,
            )
            pop_axon.insert("Nap", gbar=0.001200899579358837)
            pop_axon.insert("K_P", gbar=0.6854776593761795, ek=-85.0)
            pop_axon.insert("K_T", gbar=0.07603372775662909)
            pop_axon.insert("Kv3_1", gbar=2.988867483754507, vshift=0.0)
            pop_axon.insert("Im", gbar=0.029587905136596156)
            pop_axon.insert("SK", gbar=0.5121938998281017)
            pop_axon.insert("Ca_HVA", gbar=0.002961469262723619)
            pop_axon.insert("Ca_LVA", gbar=5.9457835817342756e-05)
            pop_axon.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=163.03538024059918,
            )

    if NUM_CELLS_VIP > 0:
        population = sim.population("VIP")
        vip_population = population
        for cell in population:
            cell.v_init = float(V_INIT_MV)
            pop_all = cell.group("all")
            pop_soma = cell.group("soma")
            pop_axon = cell.group("axon")

            pop_all.Ra = 100.0
            pop_all.cm = 2.0
            pop_all.insert("pas", e=-79.74132024971513, g=2.5756438955642182e-05)
            pop_all.insert("Ih", gbar=4.274951616063423e-05)

            pop_soma.insert(
                "NaTg",
                gbar=0.11491205828369114,
                ena=50.0,
                vshiftm=13.0,
                vshifth=15.0,
                slopem=7.0,
                slopeh=6.0,
            )
            pop_soma.insert("Nap", gbar=0.0001895305240694194)
            pop_soma.insert("K_P", gbar=0.0009925418924114282, ek=-85.0)
            pop_soma.insert("K_T", gbar=0.009051981253674193)
            pop_soma.insert("Kv3_1", gbar=0.31215653649208114, vshift=0.0)
            pop_soma.insert("SK", gbar=0.1655502166633749)
            pop_soma.insert("Im", gbar=0.0003679378262289559)
            pop_soma.insert("Ca_HVA", gbar=4.384846294634834e-05)
            pop_soma.insert("Ca_LVA", gbar=0.0034472458995879864)
            pop_soma.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=25.159166441555044,
            )

            pop_axon.insert(
                "NaTg",
                gbar=0.20112200814143477,
                ena=50.0,
                vshiftm=0.0,
                vshifth=10.0,
                slopem=9.0,
                slopeh=6.0,
            )
            pop_axon.insert("Nap", gbar=0.0006248906854665301)
            pop_axon.insert("K_P", gbar=0.26489876414660096, ek=-85.0)
            pop_axon.insert("K_T", gbar=0.014364427062274185)
            pop_axon.insert("Kv3_1", gbar=0.0011201608191112877, vshift=0.0)
            pop_axon.insert("SK", gbar=0.7027792087501376)
            pop_axon.insert("Im", gbar=0.00013891465461042372)
            pop_axon.insert("Ca_HVA", gbar=2.819397237794038e-05)
            pop_axon.insert("Ca_LVA", gbar=0.010354001513952075)
            pop_axon.insert(
                "CaDynamics",
                gamma=0.0005,
                decay=75.78875619470153,
            )

    for population in (pyr_population, sst_population, pv_population, vip_population):
        if population is None:
            continue
        for cell in population:
            soma = cell.group("soma")
            soma[0](0.5).insert(
                "IClamp",
                **{"del": float(STIM_DELAY_MS)},
                dur=float(STIM_DUR_MS),
                amp=float(STIM_AMP_NA),
            )
    morph_biophys_t1 = time.perf_counter()
    morph_biophys_s = morph_biophys_t1 - morph_biophys_t0

    pyr_cells = list(pyr_population) if pyr_population is not None else []
    sst_cells = list(sst_population) if sst_population is not None else []
    pv_cells = list(pv_population) if pv_population is not None else []
    vip_cells = list(vip_population) if vip_population is not None else []

    num_pyr = len(pyr_cells)
    num_sst = len(sst_cells)
    num_pv = len(pv_cells)
    num_vip = len(vip_cells)
    num_total = int(num_pyr + num_sst + num_pv + num_vip)

    pop_counts = {
        "PYR": num_pyr,
        "SST": num_sst,
        "PV": num_pv,
        "VIP": num_vip,
    }

    pyr_gid_begin = pyr_population.gid_begin if pyr_population is not None else 0
    sst_gid_begin = sst_population.gid_begin if sst_population is not None else num_pyr
    pv_gid_begin = pv_population.gid_begin if pv_population is not None else (num_pyr + num_sst)
    vip_gid_begin = vip_population.gid_begin if vip_population is not None else (num_pyr + num_sst + num_pv)

    synapse_count = 0
    synapse_t0 = time.perf_counter()
    gid_source_count = 0
    sampled_contacts = 0
    pyr_soma_locs = [cell.group("soma")[0](0.5) for cell in pyr_cells]
    sst_soma_locs = [cell.group("soma")[0](0.5) for cell in sst_cells]
    pv_soma_locs = [cell.group("soma")[0](0.5) for cell in pv_cells]
    vip_soma_locs = [cell.group("soma")[0](0.5) for cell in vip_cells]

    pop_soma_locs = {
        "PYR": pyr_soma_locs,
        "SST": sst_soma_locs,
        "PV": pv_soma_locs,
        "VIP": vip_soma_locs,
    }
    pop_cells = {
        "PYR": pyr_cells,
        "SST": sst_cells,
        "PV": pv_cells,
        "VIP": vip_cells,
    }

    for cell, soma_loc in zip(pyr_cells, pyr_soma_locs):
        network.register_gid_source(
            int(cell.gid),
            soma_loc._ref_v,
            threshold=float(SPIKE_THRESHOLD_MV),
        )
        gid_source_count += 1
    for cell, soma_loc in zip(sst_cells, sst_soma_locs):
        network.register_gid_source(
            int(cell.gid),
            soma_loc._ref_v,
            threshold=float(SPIKE_THRESHOLD_MV),
        )
        gid_source_count += 1
    for cell, soma_loc in zip(pv_cells, pv_soma_locs):
        network.register_gid_source(
            int(cell.gid),
            soma_loc._ref_v,
            threshold=float(SPIKE_THRESHOLD_MV),
        )
        gid_source_count += 1
    for cell, soma_loc in zip(vip_cells, vip_soma_locs):
        network.register_gid_source(
            int(cell.gid),
            soma_loc._ref_v,
            threshold=float(SPIKE_THRESHOLD_MV),
        )
        gid_source_count += 1

    if ENABLE_SYNAPSES:
        expected = estimate_contact_count(pop_counts, scale=float(SYNAPSE_SCALE))
        rng = np.random.default_rng(int(SYNAPSE_SEED))
        for pre_gid, post_loc, spec in iter_sampled_gid_contacts(
            pop_cells,
            pop_soma_locs,
            rng=rng,
            scale=float(SYNAPSE_SCALE),
            allow_autapse=bool(ALLOW_AUTAPSE),
        ):
            sampled_contacts += 1
            syn_fast = post_loc.insert(
                "Exp2Syn",
                tau1=float(spec.fast_tau_r_ms),
                tau2=float(spec.fast_tau_d_ms),
                e=float(spec.erev_mV),
            )
            network.gid_connect(
                pre_gid,
                syn_fast,
                delay=float(SYNAPSE_DELAY_MS),
                weight=float(spec.gbase_uS),
            )
            synapse_count += 1

            if (
                float(spec.slow_weight_ratio) > 0.0
                and float(spec.slow_tau_d_ms) > float(spec.slow_tau_r_ms)
            ):
                syn_slow = post_loc.insert(
                    "Exp2Syn",
                    tau1=float(spec.slow_tau_r_ms),
                    tau2=float(spec.slow_tau_d_ms),
                    e=float(spec.erev_mV),
                )
                network.gid_connect(
                    pre_gid,
                    syn_slow,
                    delay=float(SYNAPSE_DELAY_MS),
                    weight=float(spec.gbase_uS) * float(spec.slow_weight_ratio),
                )
                synapse_count += 1
        print(
            "[synapse] "
            f"enabled=1 seed={int(SYNAPSE_SEED)} "
            f"scale={float(SYNAPSE_SCALE):.6g} "
            f"autapse={int(bool(ALLOW_AUTAPSE))} "
            f"gid_sources={gid_source_count} "
            f"sampled_contacts={sampled_contacts} "
            f"expected_contacts~{expected:.1f} "
            f"built_event_targets={synapse_count}",
            flush=True,
        )
    else:
        print(f"[synapse] enabled=0 gid_sources={gid_source_count}", flush=True)
    synapse_t1 = time.perf_counter()
    synapse_netcon_s = synapse_t1 - synapse_t0

    network_finalize_t0 = time.perf_counter()
    print("[MIND_Sim] sim.build_microcircuit ...", flush=True)
    if int(sim.build_microcircuit()) != 0:
        raise RuntimeError("sim.build_microcircuit failed")
    network_finalize_t1 = time.perf_counter()
    network_finalize_s = network_finalize_t1 - network_finalize_t0


















    recorders: List[object] = []
    record_labels: List[str] = []
    if pyr_population is not None:
        if len(pyr_population) > 0:
            vec = ms.Vector().record(pyr_population[0].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PYR[0]")
            print("[record] template=PYR local=0", flush=True)
        if len(pyr_population) > 6:
            vec = ms.Vector().record(pyr_population[6].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PYR[6]")
            print("[record] template=PYR local=6", flush=True)
        if len(pyr_population) > 66:
            vec = ms.Vector().record(pyr_population[66].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PYR[66]")
            print("[record] template=PYR local=66", flush=True)
    if sst_population is not None:
        if len(sst_population) > 0:
            vec = ms.Vector().record(sst_population[0].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("SST[0]")
            print("[record] template=SST local=0", flush=True)
        if len(sst_population) > 6:
            vec = ms.Vector().record(sst_population[6].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("SST[6]")
            print("[record] template=SST local=6", flush=True)
        if len(sst_population) > 66:
            vec = ms.Vector().record(sst_population[66].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("SST[66]")
            print("[record] template=SST local=66", flush=True)
    if pv_population is not None:
        if len(pv_population) > 0:
            vec = ms.Vector().record(pv_population[0].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PV[0]")
            print("[record] template=PV local=0", flush=True)
        if len(pv_population) > 6:
            vec = ms.Vector().record(pv_population[6].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PV[6]")
            print("[record] template=PV local=6", flush=True)
        if len(pv_population) > 66:
            vec = ms.Vector().record(pv_population[66].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("PV[66]")
            print("[record] template=PV local=66", flush=True)
    if vip_population is not None:
        if len(vip_population) > 0:
            vec = ms.Vector().record(vip_population[0].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("VIP[0]")
            print("[record] template=VIP local=0", flush=True)
        if len(vip_population) > 6:
            vec = ms.Vector().record(vip_population[6].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("VIP[6]")
            print("[record] template=VIP local=6", flush=True)
        if len(vip_population) > 66:
            vec = ms.Vector().record(vip_population[66].group("soma")[0](0.5)._ref_v)
            recorders.append(vec)
            record_labels.append("VIP[66]")
            print("[record] template=VIP local=66", flush=True)
    if not recorders:
        raise RuntimeError("No valid record cells selected from template populations.")
    tvec = ms.Vector().record(sim._ref_t)
    build_t1 = time.perf_counter()

    print(f"[MIND_Sim] frontend model build elapsed={build_t1 - build_t0:.3f} s", flush=True)

    init_t0 = time.perf_counter()
    sim.finitialize(float(V_INIT_MV))
    init_t1 = time.perf_counter()
    print(f"[MIND_Sim] finitialize elapsed={init_t1 - init_t0:.3f} s", flush=True)

    run_t0 = time.perf_counter()
    sim.run(float(TSTOP_MS))
    run_t1 = time.perf_counter()
    print(
        f"[MIND_Sim] run() elapsed: {run_t1 - run_t0:.6f} s (tstop={TSTOP_MS} ms, dt={DT_MS} ms)",
        flush=True,
    )

    traces = np.vstack([np.asarray(vec.to_python(), dtype=float) for vec in recorders])
    spike_times_ms, spike_gids = _collect_spikes_from_sim(
        sim,
        num_total,
    )
    payload: Dict[str, object] = {
        "dt": float(DT_MS),
        "t": np.asarray(tvec.to_python(), dtype=float),
        "traces": traces,
        "record_labels": np.asarray(record_labels, dtype="U32"),
        "spike_times_ms": np.asarray(spike_times_ms, dtype=float),
        "spike_gids": np.asarray(spike_gids, dtype=np.int32),
        "pop_names": np.asarray(["PYR", "SST", "PV", "VIP"], dtype="U8"),
        "pop_offsets": np.asarray(
            [pyr_gid_begin, sst_gid_begin, pv_gid_begin, vip_gid_begin],
            dtype=np.int32,
        ),
        "pop_counts": np.asarray(
            [num_pyr, num_sst, num_pv, num_vip],
            dtype=np.int32,
        ),
        "runtime_backend": np.asarray(f"mind_sim_core_neuron_{DEVICE}"),
    }
    print(
        f"[spike] events={int(spike_times_ms.size)} "
        f"active_cells={int(np.unique(spike_gids).size) if spike_gids.size else 0}",
        flush=True,
    )
    for trace, label in zip(traces, record_labels):
        if SAVE_OUTPUT:
            print(
                f"[trace] {label} points={trace.size} v[0:5]={trace[:5]}",
                flush=True,
            )

    save_traces.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_traces, **payload)
    print(f"[ok] saved MIND_Sim traces to {save_traces}", flush=True)

    build_s = build_t1 - build_t0
    init_s = init_t1 - init_t0
    run_s = run_t1 - run_t0
    pre_run_total_s = max(0.0, run_t0 - build_t0)
    python_frontend_build_s = build_s
    network_build_s = max(0.0, python_frontend_build_s - morph_biophys_s)
    total_s = max(0.0, run_t1 - build_t0)
    print(
        "[timing] "
        f"template_export_s=0.000 "
        f"build_s={build_s:.3f} "
        f"neuron_build_s={build_s:.3f} "
        f"frontend_morph_biophys_s={morph_biophys_s:.3f} "
        f"frontend_synapse_netcon_s={synapse_netcon_s:.3f} "
        f"frontend_network_finalize_s={network_finalize_s:.3f} "
        f"frontend_subtotal_s={(morph_biophys_s + synapse_netcon_s + network_finalize_s):.3f} "
        f"frontend_network_build_s={network_build_s:.3f} "
        f"python_frontend_build_s={python_frontend_build_s:.3f} "
        f"python_asset_load_s=0.000 "
        f"finitialize_s={init_s:.3f} "
        f"frontend_build_plus_finitialize_s={pre_run_total_s:.3f} "
        f"full_build_plus_finitialize_s={pre_run_total_s:.3f} "
        f"pre_run_total_s={pre_run_total_s:.3f} "
        f"python_pre_run_total_s={pre_run_total_s:.3f} "
        f"run_s={run_s:.3f} "
        f"sum_s={total_s:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
