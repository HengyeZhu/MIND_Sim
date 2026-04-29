#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
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

# Late-bound in `main()` after environment is finalized.
h = None  # type: ignore
coreneuron = None  # type: ignore


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _sort_spikes(times_ms: np.ndarray, gids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if times_ms.size == 0:
        return times_ms.astype(float), gids.astype(np.int32)
    order = np.argsort(times_ms, kind="mergesort")
    return np.asarray(times_ms[order], dtype=float), np.asarray(gids[order], dtype=np.int32)


def _hoc_vector_to_numpy(vec: Any, dtype) -> np.ndarray:
    try:
        return np.asarray(vec.to_python(), dtype=dtype)
    except Exception:
        pass
    try:
        return np.asarray(vec, dtype=dtype)
    except Exception:
        pass
    size = int(vec.size()) if hasattr(vec, "size") else 0
    return np.asarray([vec.x[i] for i in range(size)], dtype=dtype)


def resolve_hl23_asset_dir(repo_root: Path) -> Path:
    env_dir = os.environ.get("MIND_SIM_HL23_ASSET_DIR", "").strip()
    candidates: List[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append(repo_root / "examples" / "hl23multi" / "hl23_minimal")
    return pick_existing_dir(candidates, label="HL23 asset directory")


REPO_ROOT = Path(__file__).resolve().parents[2]
MICROCIRCUIT_DIR = resolve_hl23_asset_dir(REPO_ROOT)


DEFAULT_PYR_TEMPLATE_HOC = MICROCIRCUIT_DIR / "NeuronTemplate.hoc"
DEFAULT_SST_TEMPLATE_HOC = MICROCIRCUIT_DIR / "NeuronTemplate.hoc"
DEFAULT_PV_TEMPLATE_HOC = MICROCIRCUIT_DIR / "NeuronTemplate_full.hoc"
DEFAULT_VIP_TEMPLATE_HOC = MICROCIRCUIT_DIR / "NeuronTemplate.hoc"
DEFAULT_PYR_SWC = MICROCIRCUIT_DIR / "HL23PYR.swc"
DEFAULT_SST_SWC = MICROCIRCUIT_DIR / "HL23SST.swc"
DEFAULT_PV_SWC = MICROCIRCUIT_DIR / "HL23PV.swc"
DEFAULT_VIP_SWC = MICROCIRCUIT_DIR / "HL23VIP.swc"
DEFAULT_PYR_BIOPHYS_HOC = MICROCIRCUIT_DIR / "biophys_HL23PYR.hoc"
DEFAULT_SST_BIOPHYS_HOC = MICROCIRCUIT_DIR / "biophys_HL23SST.hoc"
DEFAULT_PV_BIOPHYS_HOC = MICROCIRCUIT_DIR / "biophys_HL23PV.hoc"
DEFAULT_VIP_BIOPHYS_HOC = MICROCIRCUIT_DIR / "biophys_HL23VIP.hoc"
DEFAULT_MOD_DIR = MICROCIRCUIT_DIR / "mod"
DEFAULT_EXPORT_VIA_NEURON = MICROCIRCUIT_DIR / "export_via_neuron.py"


def try_load_local_mechs(search_dirs: List[str]) -> str | None:
    candidates: List[str] = []
    for d in search_dirs:
        if not d:
            continue
        base = os.path.abspath(d)
        candidates.extend(
            [
                os.path.join(base, "x86_64", "libnrnmech.so"),
                os.path.join(base, "x86_64", ".libs", "libnrnmech.so"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            try:
                h.nrn_load_dll(path)
                print(f"Loaded mod library: {path}")
                return path
            except Exception as exc:
                print(f"Warning: failed to load {path}: {exc}")
    return None


def ensure_mechs(hoc_path_any: str, mod_dir: str | None, *, force_rebuild: bool = False) -> str | None:
    hoc_dir = os.path.dirname(os.path.abspath(hoc_path_any))
    search_dirs = [hoc_dir]
    if mod_dir:
        search_dirs.append(os.path.dirname(os.path.abspath(mod_dir)))

    if not force_rebuild:
        lib = try_load_local_mechs(search_dirs)
        if lib:
            return lib

    if not mod_dir:
        print("Warning: mod_dir not provided; continuing without loading MOD library")
        return None

    mod_dir_abs = os.path.abspath(mod_dir)
    build_dir = os.path.dirname(mod_dir_abs)
    corenrn_lib = Path(build_dir) / "x86_64" / "libcorenrnmech.so"
    try:
        print(f"Compiling MOD directory: {mod_dir_abs}")
        subprocess.run(["nrnivmodl", mod_dir_abs], cwd=build_dir, check=True)
        if not corenrn_lib.is_file():
            subprocess.run(["nrnivmodl", "-coreneuron", mod_dir_abs], cwd=build_dir, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Warning: nrnivmodl failed: {exc}")

    return try_load_local_mechs([build_dir])


def _maybe_prepare_corenrnlib(mod_dir: str | None, *, require: bool) -> str | None:
    if os.environ.get("CORENEURONLIB", "").strip():
        return os.environ.get("CORENEURONLIB")

    candidates: List[Path] = []
    if mod_dir:
        mod_dir_abs = Path(mod_dir).expanduser().resolve()
        build_dir = mod_dir_abs.parent
        candidates.extend(
            [
                build_dir / "x86_64" / "libcorenrnmech.so",
                build_dir / "x86_64" / "libcorenrnmech_internal.so",
            ]
        )

    for path in candidates:
        if path.is_file():
            os.environ["CORENEURONLIB"] = str(path)
            print(f"[coreneuron] CORENEURONLIB={path}", flush=True)
            return str(path)

    if require and mod_dir:
        mod_dir_abs = os.path.abspath(mod_dir)
        build_dir = os.path.dirname(mod_dir_abs)
        try:
            subprocess.run(["nrnivmodl", "-coreneuron", mod_dir_abs], cwd=build_dir, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        built = Path(build_dir) / "x86_64" / "libcorenrnmech.so"
        if built.is_file():
            os.environ["CORENEURONLIB"] = str(built)
            print(f"[coreneuron] CORENEURONLIB={built}", flush=True)
            return str(built)

    return None


def _is_outdated(output: Path, *, inputs: List[Path]) -> bool:
    if not output.exists():
        return True
    try:
        out_mtime = output.stat().st_mtime
    except OSError:
        return True
    for dep in inputs:
        try:
            if dep.stat().st_mtime > out_mtime:
                return True
        except OSError:
            return True
    return False


def _ensure_exported_celltemplate_hoc(
    *,
    cell_name: str,
    swc_file: Path,
    biophys_hoc: Path,
    template_hoc: Path,
    output_hoc: Path,
    python: str,
) -> float:
    export_script = DEFAULT_EXPORT_VIA_NEURON
    inputs = [swc_file, biophys_hoc, template_hoc, export_script]
    if not _is_outdated(output_hoc, inputs=inputs):
        print(f"[hoc] cache hit: {output_hoc}", flush=True)
        return 0.0

    output_hoc.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-B",
        str(export_script),
        "--cell-name",
        cell_name,
        "--swc-file",
        str(swc_file),
        "--biophys-file",
        str(biophys_hoc),
        "--template-hoc",
        str(template_hoc),
        "--output-hoc",
        str(output_hoc),
        "--skip-neuroml",
    ]
    print(f"[hoc] exporting template: {cell_name} -> {output_hoc}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    global coreneuron, h
    default_enable_synapses = _env_bool("MIND_SIM_HL23_ENABLE_SYNAPSES", True)
    default_allow_autapse = _env_bool("MIND_SIM_HL23_ALLOW_AUTAPSE", True)
    default_synapse_seed = int(os.environ.get("MIND_SIM_HL23_CONN_SEED", "12345"))
    default_synapse_scale = float(os.environ.get("MIND_SIM_HL23_CONN_SCALE", "1.0"))
    default_synapse_delay = float(os.environ.get("MIND_SIM_HL23_CONN_DELAY_MS", "0.5"))
    default_spike_threshold = float(os.environ.get("MIND_SIM_HL23_SPIKE_THRESHOLD_MV", "0.0"))
    default_force_mod_rebuild = _env_bool("MIND_SIM_HL23_FORCE_MOD_REBUILD", False)

    parser = argparse.ArgumentParser(
        description="Run HL23PYR+HL23SST+HL23PV+HL23VIP baseline with selectable runtime backend"
    )
    parser.add_argument("--pyr-template-hoc-file", default=str(DEFAULT_PYR_TEMPLATE_HOC))
    parser.add_argument("--sst-template-hoc-file", default=str(DEFAULT_SST_TEMPLATE_HOC))
    parser.add_argument("--pv-template-hoc-file", default=str(DEFAULT_PV_TEMPLATE_HOC))
    parser.add_argument("--vip-template-hoc-file", default=str(DEFAULT_VIP_TEMPLATE_HOC))
    parser.add_argument("--pyr-swc-file", default=str(DEFAULT_PYR_SWC))
    parser.add_argument("--sst-swc-file", default=str(DEFAULT_SST_SWC))
    parser.add_argument("--pv-swc-file", default=str(DEFAULT_PV_SWC))
    parser.add_argument("--vip-swc-file", default=str(DEFAULT_VIP_SWC))
    parser.add_argument("--pyr-biophys-hoc-file", default=str(DEFAULT_PYR_BIOPHYS_HOC))
    parser.add_argument("--sst-biophys-hoc-file", default=str(DEFAULT_SST_BIOPHYS_HOC))
    parser.add_argument("--pv-biophys-hoc-file", default=str(DEFAULT_PV_BIOPHYS_HOC))
    parser.add_argument("--vip-biophys-hoc-file", default=str(DEFAULT_VIP_BIOPHYS_HOC))
    parser.add_argument("--mod-dir", default=str(DEFAULT_MOD_DIR))
    parser.add_argument("--hoc-cache-dir", default=None)
    parser.add_argument("--num-cells", type=int, default=None, help="Alias: set all counts")
    parser.add_argument("--num-cells-pyr", type=int, default=1)
    parser.add_argument("--num-cells-sst", type=int, default=1)
    parser.add_argument("--num-cells-pv", type=int, default=1)
    parser.add_argument("--num-cells-vip", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tstop", type=float, default=600.0)
    parser.add_argument("--v-init", type=float, default=-80.0)
    parser.add_argument("--celsius", type=float, default=6.3)
    parser.add_argument("--stim-delay", type=float, default=200.0)
    parser.add_argument("--stim-dur", type=float, default=150.0)
    parser.add_argument("--stim-amp", type=float, default=2.0)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-traces", default=None)
    parser.add_argument("--export-path", default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--coreneuron-file-mode",
        choices=["auto", "true", "false"],
        default="auto",
        help="CoreNEURON transfer mode; auto uses file-mode when CORENRN shared libs are unavailable.",
    )
    parser.set_defaults(enable_synapses=default_enable_synapses)
    parser.add_argument("--enable-synapses", dest="enable_synapses", action="store_true")
    parser.add_argument("--disable-synapses", dest="enable_synapses", action="store_false")
    parser.set_defaults(allow_autapse=default_allow_autapse)
    parser.add_argument("--allow-autapse", dest="allow_autapse", action="store_true")
    parser.add_argument("--disallow-autapse", dest="allow_autapse", action="store_false")
    parser.add_argument("--synapse-seed", type=int, default=default_synapse_seed)
    parser.add_argument("--synapse-scale", type=float, default=default_synapse_scale)
    parser.add_argument("--synapse-delay-ms", type=float, default=default_synapse_delay)
    parser.add_argument("--spike-threshold-mv", type=float, default=default_spike_threshold)
    parser.set_defaults(force_mod_rebuild=default_force_mod_rebuild)
    parser.add_argument("--force-mod-rebuild", dest="force_mod_rebuild", action="store_true")
    parser.add_argument("--no-force-mod-rebuild", dest="force_mod_rebuild", action="store_false")
    args = parser.parse_args()

    if args.num_cells is not None:
        args.num_cells_pyr = int(args.num_cells)
        args.num_cells_sst = int(args.num_cells)
        args.num_cells_pv = int(args.num_cells)
        args.num_cells_vip = int(args.num_cells)

    num_total = int(args.num_cells_pyr + args.num_cells_sst + args.num_cells_pv + args.num_cells_vip)
    if num_total < 1:
        raise SystemExit("need at least 1 total cell (pyr+sst+pv+vip)")

    pyr_swc_abs = os.path.abspath(args.pyr_swc_file)
    sst_swc_abs = os.path.abspath(args.sst_swc_file)
    pv_swc_abs = os.path.abspath(args.pv_swc_file)
    vip_swc_abs = os.path.abspath(args.vip_swc_file)
    pyr_bio_hoc_abs = os.path.abspath(args.pyr_biophys_hoc_file)
    sst_bio_hoc_abs = os.path.abspath(args.sst_biophys_hoc_file)
    pv_bio_hoc_abs = os.path.abspath(args.pv_biophys_hoc_file)
    vip_bio_hoc_abs = os.path.abspath(args.vip_biophys_hoc_file)
    pyr_template_hoc_abs = os.path.abspath(args.pyr_template_hoc_file)
    sst_template_hoc_abs = os.path.abspath(args.sst_template_hoc_file)
    pv_template_hoc_abs = os.path.abspath(args.pv_template_hoc_file)
    vip_template_hoc_abs = os.path.abspath(args.vip_template_hoc_file)
    mod_dir = os.path.abspath(args.mod_dir) if args.mod_dir else None

    for path, label in [
        (pyr_swc_abs, "PYR SWC"),
        (sst_swc_abs, "SST SWC"),
        (pv_swc_abs, "PV SWC"),
        (vip_swc_abs, "VIP SWC"),
        (pyr_bio_hoc_abs, "PYR biophys hoc"),
        (sst_bio_hoc_abs, "SST biophys hoc"),
        (pv_bio_hoc_abs, "PV biophys hoc"),
        (vip_bio_hoc_abs, "VIP biophys hoc"),
        (pyr_template_hoc_abs, "PYR template hoc"),
        (sst_template_hoc_abs, "SST template hoc"),
        (pv_template_hoc_abs, "PV template hoc"),
        (vip_template_hoc_abs, "VIP template hoc"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    _maybe_prepare_corenrnlib(mod_dir, require=True)

    from neuron import coreneuron as _coreneuron, h as _h  # type: ignore

    coreneuron = _coreneuron
    h = _h

    ensure_mechs(pyr_template_hoc_abs, mod_dir, force_rebuild=bool(args.force_mod_rebuild))
    h.load_file("stdrun.hoc")
    h.celsius = float(args.celsius)

    python = sys.executable
    if args.hoc_cache_dir:
        cache_dir = Path(args.hoc_cache_dir).expanduser().resolve()
    elif args.save_traces:
        cache_dir = Path(args.save_traces).expanduser().resolve().parent / "hoc_templates"
    else:
        cache_dir = Path.cwd() / "hoc_templates"

    pyr_template_out = cache_dir / "HL23PYR.axon.hoc"
    sst_template_out = cache_dir / "HL23SST.axon.hoc"
    pv_template_out = cache_dir / "HL23PV.full.hoc"
    vip_template_out = cache_dir / "HL23VIP.bpo.hoc"

    print(
        "[build] "
        f"pyr_swc={pyr_swc_abs} pyr_cells={args.num_cells_pyr} | "
        f"sst_swc={sst_swc_abs} sst_cells={args.num_cells_sst} | "
        f"pv_swc={pv_swc_abs} pv_cells={args.num_cells_pv} | "
        f"vip_swc={vip_swc_abs} vip_cells={args.num_cells_vip} | "
        f"dt={args.dt}ms tstop={args.tstop}ms v_init={args.v_init}mV celsius={args.celsius}",
        flush=True,
    )
    build_t0 = time.perf_counter()
    morph_biophys_s = 0.0
    synapse_netcon_s = 0.0
    post_syn_setup_s = 0.0

    export_t0 = time.perf_counter()
    export_pyr_s = _ensure_exported_celltemplate_hoc(
        cell_name="HL23PYR",
        swc_file=Path(pyr_swc_abs),
        biophys_hoc=Path(pyr_bio_hoc_abs),
        template_hoc=Path(pyr_template_hoc_abs),
        output_hoc=pyr_template_out,
        python=python,
    )
    export_sst_s = _ensure_exported_celltemplate_hoc(
        cell_name="HL23SST",
        swc_file=Path(sst_swc_abs),
        biophys_hoc=Path(sst_bio_hoc_abs),
        template_hoc=Path(sst_template_hoc_abs),
        output_hoc=sst_template_out,
        python=python,
    )
    export_pv_s = _ensure_exported_celltemplate_hoc(
        cell_name="HL23PV",
        swc_file=Path(pv_swc_abs),
        biophys_hoc=Path(pv_bio_hoc_abs),
        template_hoc=Path(pv_template_hoc_abs),
        output_hoc=pv_template_out,
        python=python,
    )
    export_vip_s = _ensure_exported_celltemplate_hoc(
        cell_name="HL23VIP",
        swc_file=Path(vip_swc_abs),
        biophys_hoc=Path(vip_bio_hoc_abs),
        template_hoc=Path(vip_template_hoc_abs),
        output_hoc=vip_template_out,
        python=python,
    )
    export_t1 = time.perf_counter()

    morph_t0 = time.perf_counter()
    h.xopen(str(pyr_template_out))
    h.xopen(str(sst_template_out))
    h.xopen(str(pv_template_out))
    h.xopen(str(vip_template_out))
    pyr_cls = getattr(h, "HL23PYR")
    sst_cls = getattr(h, "HL23SST")
    pv_cls = getattr(h, "HL23PV")
    vip_cls = getattr(h, "HL23VIP")

    pc = h.ParallelContext()
    pyr_cells: List[Any] = []
    sst_cells: List[Any] = []
    pv_cells: List[Any] = []
    vip_cells: List[Any] = []
    iclamps = []
    syn_keepalive: List[Any] = []
    stim_delay = float(args.stim_delay)
    stim_amp = float(args.stim_amp)
    stim_dur = float(args.stim_dur)
    for gid in range(num_total):
        if gid < int(args.num_cells_pyr):
            cell = pyr_cls()
            pyr_cells.append(cell)
        elif gid < int(args.num_cells_pyr + args.num_cells_sst):
            cell = sst_cls()
            sst_cells.append(cell)
        elif gid < int(args.num_cells_pyr + args.num_cells_sst + args.num_cells_pv):
            cell = pv_cls()
            pv_cells.append(cell)
        else:
            cell = vip_cls()
            vip_cells.append(cell)

        soma = cell.soma[0]
        soma_mid = soma(0.5)
        stim = h.IClamp(soma_mid)
        stim.delay = stim_delay
        stim.amp = stim_amp
        stim.dur = 1e9 if (stim_dur == 0.0 and stim_amp == 0.0) else stim_dur

        pc.set_gid2node(gid, int(pc.id()))
        nc = h.NetCon(soma_mid._ref_v, None, sec=soma)
        nc.threshold = float(args.spike_threshold_mv)
        pc.cell(gid, nc)

        iclamps.append(stim)
    morph_t1 = time.perf_counter()
    morph_biophys_s = morph_t1 - morph_t0

    pop_counts = {
        "PYR": int(args.num_cells_pyr),
        "SST": int(args.num_cells_sst),
        "PV": int(args.num_cells_pv),
        "VIP": int(args.num_cells_vip),
    }
    pop_cells = {
        "PYR": pyr_cells,
        "SST": sst_cells,
        "PV": pv_cells,
        "VIP": vip_cells,
    }
    pop_gid_offset = {
        "PYR": 0,
        "SST": int(args.num_cells_pyr),
        "PV": int(args.num_cells_pyr + args.num_cells_sst),
        "VIP": int(args.num_cells_pyr + args.num_cells_sst + args.num_cells_pv),
    }

    synapse_count = 0
    syn_t0 = time.perf_counter()
    if args.enable_synapses:
        rng = np.random.default_rng(int(args.synapse_seed))
        expected = estimate_contact_count(pop_counts, scale=float(args.synapse_scale))
        for pre_name, pre_local, post_name, post_local, spec in iter_sampled_contacts(
            pop_counts,
            rng=rng,
            scale=float(args.synapse_scale),
            allow_autapse=bool(args.allow_autapse),
        ):
            pre_gid = int(pop_gid_offset[pre_name] + pre_local)
            post_cell = pop_cells[post_name][post_local]
            post_loc = post_cell.soma[0](0.5)

            syn_fast = h.Exp2Syn(post_loc)
            syn_fast.tau1 = float(spec.fast_tau_r_ms)
            syn_fast.tau2 = float(spec.fast_tau_d_ms)
            syn_fast.e = float(spec.erev_mV)
            nc_fast = pc.gid_connect(pre_gid, syn_fast)
            nc_fast.threshold = float(args.spike_threshold_mv)
            nc_fast.delay = float(args.synapse_delay_ms)
            nc_fast.weight[0] = float(spec.gbase_uS)
            syn_keepalive.extend([syn_fast, nc_fast])
            synapse_count += 1

            if float(spec.slow_weight_ratio) > 0.0 and float(spec.slow_tau_d_ms) > float(spec.slow_tau_r_ms):
                syn_slow = h.Exp2Syn(post_loc)
                syn_slow.tau1 = float(spec.slow_tau_r_ms)
                syn_slow.tau2 = float(spec.slow_tau_d_ms)
                syn_slow.e = float(spec.erev_mV)
                nc_slow = pc.gid_connect(pre_gid, syn_slow)
                nc_slow.threshold = float(args.spike_threshold_mv)
                nc_slow.delay = float(args.synapse_delay_ms)
                nc_slow.weight[0] = float(spec.gbase_uS) * float(spec.slow_weight_ratio)
                syn_keepalive.extend([syn_slow, nc_slow])
                synapse_count += 1
        print(
            "[synapse] "
            f"enabled=1 seed={int(args.synapse_seed)} "
            f"scale={float(args.synapse_scale):.6g} "
            f"autapse={int(bool(args.allow_autapse))} "
            f"expected_contacts~{expected:.1f} "
            f"built_point_processes={synapse_count}",
            flush=True,
        )
    else:
        print("[synapse] enabled=0", flush=True)
    syn_t1 = time.perf_counter()
    synapse_netcon_s = syn_t1 - syn_t0

    setup_t0 = time.perf_counter()
    pc.setup_transfer()
    pc.set_maxstep(10)
    setup_t1 = time.perf_counter()
    post_syn_setup_s = setup_t1 - setup_t0
    build_t1 = time.perf_counter()
    template_s = export_t1 - export_t0
    print(
        f"[hoc] export_s={template_s:.3f} "
        f"(pyr={export_pyr_s:.3f} sst={export_sst_s:.3f} pv={export_pv_s:.3f} vip={export_vip_s:.3f})",
        flush=True,
    )
    print(f"Cell built. build/init elapsed: {build_t1 - build_t0:.3f} s", flush=True)

    record_sites: List[Any] = []
    record_labels: List[str] = []
    if len(pyr_cells) > 0:
        record_sites.append(pyr_cells[0].soma[0](0.5))
        record_labels.append("PYR[0]")
    if len(pyr_cells) > 6:
        record_sites.append(pyr_cells[6].soma[0](0.5))
        record_labels.append("PYR[6]")
    if len(pyr_cells) > 66:
        record_sites.append(pyr_cells[66].soma[0](0.5))
        record_labels.append("PYR[66]")
    if len(sst_cells) > 0:
        record_sites.append(sst_cells[0].soma[0](0.5))
        record_labels.append("SST[0]")
    if len(sst_cells) > 6:
        record_sites.append(sst_cells[6].soma[0](0.5))
        record_labels.append("SST[6]")
    if len(sst_cells) > 66:
        record_sites.append(sst_cells[66].soma[0](0.5))
        record_labels.append("SST[66]")
    if len(pv_cells) > 0:
        record_sites.append(pv_cells[0].soma[0](0.5))
        record_labels.append("PV[0]")
    if len(pv_cells) > 6:
        record_sites.append(pv_cells[6].soma[0](0.5))
        record_labels.append("PV[6]")
    if len(pv_cells) > 66:
        record_sites.append(pv_cells[66].soma[0](0.5))
        record_labels.append("PV[66]")
    if len(vip_cells) > 0:
        record_sites.append(vip_cells[0].soma[0](0.5))
        record_labels.append("VIP[0]")
    if len(vip_cells) > 6:
        record_sites.append(vip_cells[6].soma[0](0.5))
        record_labels.append("VIP[6]")
    if len(vip_cells) > 66:
        record_sites.append(vip_cells[66].soma[0](0.5))
        record_labels.append("VIP[66]")
    if not record_sites:
        raise SystemExit("no valid record cells selected from template populations")

    export_load_s = 0.0
    init_s = 0.0
    run_s = 0.0
    run_start_t = 0.0
    run_end_t = 0.0
    traces = None
    t_axis = None
    spike_times_ms = np.zeros((0,), dtype=float)
    spike_gids = np.zeros((0,), dtype=np.int32)
    spike_tvec = h.Vector()
    spike_gidvec = h.Vector()
    pc.spike_record(-1, spike_tvec, spike_gidvec)

    cn_file_mode: bool
    if str(args.coreneuron_file_mode) == "auto":
        cn_shared = True
        try:
            from neuron import config as _nrn_config  # type: ignore

            cn_shared = bool(_nrn_config.arguments.get("CORENRN_ENABLE_SHARED", True))
        except Exception:
            cn_shared = True
        cn_file_mode = not cn_shared
    else:
        cn_file_mode = str(args.coreneuron_file_mode) == "true"
    print(
        f"[coreneuron] file_mode={int(cn_file_mode)} "
        f"(requested={args.coreneuron_file_mode})",
        flush=True,
    )

    h.cvode.cache_efficient(1)
    h.dt = float(args.dt)
    h.tstop = float(args.tstop)
    hoc_recorders = [h.Vector().record(site._ref_v) for site in record_sites]
    tvec = h.Vector().record(h._ref_t)
    with coreneuron(
        enable=True,
        verbose=0,
        gpu=(str(args.device) == "gpu"),
        num_gpus=1 if str(args.device) == "gpu" else 0,
        cell_permute=2 if str(args.device) == "gpu" else 1,
        file_mode=cn_file_mode,
    ):
        init_t0 = time.perf_counter()
        h.finitialize(args.v_init)
        init_t1 = time.perf_counter()
        run_t0 = time.perf_counter()
        try:
            pc.psolve(args.tstop)
        except Exception as exc:
            if cn_file_mode:
                raise RuntimeError(
                    "NEURON+CoreNEURON baseline requires CoreNEURON runtime via ParallelContext.psolve, "
                    "but CoreNEURON mechanism shared library is unavailable in this environment. "
                    "Please provide libcorenrnmech(.so) / libcorenrnmech_internal(.so), "
                    "or rebuild NEURON with CORENRN_ENABLE_SHARED=ON."
                ) from exc
            raise
        run_t1 = time.perf_counter()
    init_s = init_t1 - init_t0
    run_s = run_t1 - run_t0
    run_start_t = run_t0
    run_end_t = run_t1
    traces = np.vstack([_hoc_vector_to_numpy(vec, float) for vec in hoc_recorders])
    t_axis = _hoc_vector_to_numpy(tvec, float)
    spike_times_ms, spike_gids = _sort_spikes(
        _hoc_vector_to_numpy(spike_tvec, float),
        _hoc_vector_to_numpy(spike_gidvec, np.int32),
    )
    print(f"CoreNEURON run() elapsed: {run_s:.6f} s (tstop={args.tstop} ms, dt={args.dt} ms)")

    frontend_build_s = build_t1 - build_t0
    frontend_build_excl_hoc_s = max(0.0, frontend_build_s - template_s)
    network_build_s = max(0.0, frontend_build_excl_hoc_s - morph_biophys_s)
    pre_run_total_s = max(0.0, run_start_t - build_t0)
    pre_run_total_excl_hoc_s = max(0.0, pre_run_total_s - template_s)
    pre_run_total_excl_hoc_export_s = max(0.0, pre_run_total_excl_hoc_s - export_load_s)
    sum_s = max(0.0, run_end_t - build_t0)

    print(
        "[timing] "
        f"hoc_template_s={template_s:.3f} "
        f"build_s={frontend_build_s:.3f} "
        f"neuron_build_s={frontend_build_s:.3f} "
        f"frontend_morph_biophys_s={morph_biophys_s:.3f} "
        f"frontend_synapse_netcon_s={synapse_netcon_s:.3f} "
        f"frontend_post_syn_setup_s={post_syn_setup_s:.3f} "
        f"frontend_subtotal_s={(morph_biophys_s + synapse_netcon_s + post_syn_setup_s):.3f} "
        f"frontend_network_build_s={network_build_s:.3f} "
        f"frontend_build_excl_hoc_s={frontend_build_excl_hoc_s:.3f} "
        f"export_load_s={export_load_s:.3f} "
        f"finitialize_s={init_s:.3f} "
        f"frontend_build_plus_finitialize_s={frontend_build_s + init_s:.3f} "
        f"full_build_plus_finitialize_s={pre_run_total_s:.3f} "
        f"pre_run_total_s={pre_run_total_s:.3f} "
        f"pre_run_total_excl_hoc_s={pre_run_total_excl_hoc_s:.3f} "
        f"pre_run_total_excl_hoc_export_s={pre_run_total_excl_hoc_export_s:.3f} "
        f"run_s={run_s:.3f} "
        f"sum_s={sum_s:.3f}"
    )

    if args.save_output:
        assert traces is not None
        assert t_axis is not None
        print(
            f"[spike] events={int(spike_times_ms.size)} "
            f"active_cells={int(np.unique(spike_gids).size) if spike_gids.size else 0}",
            flush=True,
        )
        for label, v in zip(record_labels, traces):
            print(f"[Recorder {label}] points={v.size}; t[0:5]={t_axis[:5]}, v[0:5]={v[:5]}")
        if args.save_traces:
            save_path = os.path.abspath(args.save_traces)
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            payload: Dict[str, Any] = {"dt": args.dt, "traces": traces}
            payload["t"] = t_axis
            payload["record_labels"] = np.asarray(record_labels, dtype="U32")
            payload["spike_times_ms"] = np.asarray(spike_times_ms, dtype=float)
            payload["spike_gids"] = np.asarray(spike_gids, dtype=np.int32)
            payload["pop_names"] = np.asarray(["PYR", "SST", "PV", "VIP"], dtype="U8")
            payload["pop_offsets"] = np.asarray(
                [
                    pop_gid_offset["PYR"],
                    pop_gid_offset["SST"],
                    pop_gid_offset["PV"],
                    pop_gid_offset["VIP"],
                ],
                dtype=np.int32,
            )
            payload["pop_counts"] = np.asarray(
                [
                    pop_counts["PYR"],
                    pop_counts["SST"],
                    pop_counts["PV"],
                    pop_counts["VIP"],
                ],
                dtype=np.int32,
            )
            payload["runtime_backend"] = np.asarray("neuron_coreneuron")
            np.savez(save_path, **payload)
            print(f"Saved recorded traces to {save_path}")
    else:
        if args.save_traces:
            print("Skipping save_traces because --save-output not set (timing-only mode).")


if __name__ == "__main__":
    main()
