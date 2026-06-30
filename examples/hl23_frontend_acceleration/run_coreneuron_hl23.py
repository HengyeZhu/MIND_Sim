#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

PYR_COUNT = 850
SST_COUNT = 50
PV_COUNT = 50
VIP_COUNT = 50
SPIKE_THRESHOLD_MV = 0.0
CELL_TYPES = ("HL23PYR", "HL23SST", "HL23PV", "HL23VIP")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the HL23 model with CPU CoreNEURON.")
    parser.add_argument("--mechanism-lib", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--duration-ms", "--tstop", dest="tstop", type=float, default=50.0)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "coreneuron_hl23.npz",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mechanism_lib = args.mechanism_lib.resolve()
    if not mechanism_lib.is_file():
        raise FileNotFoundError(f"mechanism library not found: {mechanism_lib}")
    coreneuron_lib = mechanism_lib.parent / "libcorenrnmech.so"
    if not coreneuron_lib.is_file():
        raise FileNotFoundError(f"CoreNEURON mechanism library not found: {coreneuron_lib}")
    os.environ["CORENEURONLIB"] = str(coreneuron_lib)

    from neuron import h  # type: ignore
    from neuron import coreneuron  # type: ignore

    h.nrn_load_dll(str(mechanism_lib))
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.celsius = 6.3
    h.dt = 0.025
    h.tstop = float(args.tstop)

    coreneuron.enable = True
    coreneuron.gpu = False
    coreneuron.cell_permute = 1
    coreneuron.verbose = 0

    total_t0 = time.perf_counter()
    build_t0 = time.perf_counter()
    asset_dir = Path(__file__).resolve().parent / "assets"
    model_dir = asset_dir / "models"
    morphology_dir = asset_dir / "morphologies"
    hoc_files = [model_dir / "NeuronTemplate.hoc"]
    hoc_files.extend(model_dir / f"biophys_{cell_type}.hoc" for cell_type in CELL_TYPES)
    for path in hoc_files:
        if not path.is_file():
            raise FileNotFoundError(f"HOC file not found: {path}")
        print(f"[hoc] load: {path}", flush=True)
        h.xopen(str(path))
    for cell_type in CELL_TYPES:
        morphology = morphology_dir / f"{cell_type}.swc"
        if not morphology.is_file():
            raise FileNotFoundError(f"SWC morphology not found: {morphology}")

    print(
        "[build] "
        f"engine=coreneuron_cpu counts=PYR:{PYR_COUNT} SST:{SST_COUNT} PV:{PV_COUNT} VIP:{VIP_COUNT} "
        f"dt=0.025ms tstop={args.tstop}ms threads={args.threads}",
        flush=True,
    )

    pc = h.ParallelContext()
    pc.nthread(int(args.threads))
    total_cells = PYR_COUNT + SST_COUNT + PV_COUNT + VIP_COUNT
    cells = []
    clamps = []
    netcons = []

    def build_cell(cell_type: str):
        cell = h.NeuronTemplate(str(morphology_dir / f"{cell_type}.swc"))
        getattr(h, f"biophys_{cell_type}")(cell)
        return cell

    for gid in range(total_cells):
        if gid < PYR_COUNT:
            cell = build_cell("HL23PYR")
        elif gid < PYR_COUNT + SST_COUNT:
            cell = build_cell("HL23SST")
        elif gid < PYR_COUNT + SST_COUNT + PV_COUNT:
            cell = build_cell("HL23PV")
        else:
            cell = build_cell("HL23VIP")

        soma = cell.soma[0]
        stim = h.IClamp(soma(0.5))
        stim.delay = 5.0
        stim.dur = 80.0
        stim.amp = 2.0

        pc.set_gid2node(gid, int(pc.id()))
        soma.push()
        netcon = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        netcon.threshold = SPIKE_THRESHOLD_MV
        pc.cell(gid, netcon)
        h.pop_section()

        cells.append(cell)
        clamps.append(stim)
        netcons.append(netcon)

    pc.setup_transfer()
    pc.set_maxstep(10)

    v_vectors = {}
    for name, gid_offset, count in (
        ("PYR", 0, PYR_COUNT),
        ("SST", PYR_COUNT, SST_COUNT),
        ("PV", PYR_COUNT + SST_COUNT, PV_COUNT),
        ("VIP", PYR_COUNT + SST_COUNT + PV_COUNT, VIP_COUNT),
    ):
        for local_index in (0, 6, 66):
            if local_index >= count:
                print(f"[record] skip {name}[{local_index}] because count={count}", flush=True)
                continue
            gid = gid_offset + local_index
            soma = cells[gid].soma[0]
            v_vectors[gid] = h.Vector().record(soma(0.5)._ref_v)
    if not v_vectors:
        raise RuntimeError("no soma vectors selected for recording")
    t_vec = h.Vector().record(h._ref_t)

    spike_times_vector = h.Vector()
    spike_gids_vector = h.Vector()
    pc.spike_record(-1, spike_times_vector, spike_gids_vector)
    build_s = time.perf_counter() - build_t0

    init_t0 = time.perf_counter()
    h.finitialize(-80.0)
    init_s = time.perf_counter() - init_t0

    run_t0 = time.perf_counter()
    pc.psolve(float(args.tstop))
    run_s = time.perf_counter() - run_t0
    sum_s = time.perf_counter() - total_t0
    pre_run_s = build_s + init_s
    print(
        "[timing] "
        f"build_s={build_s:.6f} finitialize_s={init_s:.6f} "
        f"pre_run_s={pre_run_s:.6f} run_s={run_s:.6f} sum_s={sum_s:.6f}",
        flush=True,
    )
    spike_times = np.asarray(spike_times_vector, dtype=float)
    spike_gids = np.asarray(spike_gids_vector, dtype=int)
    if spike_times.size:
        order = np.lexsort((spike_gids, spike_times))
        spike_times = spike_times[order]
        spike_gids = spike_gids[order]
    first_spike = float(spike_times[0]) if spike_times.size else float("nan")
    print(
        f"[spikes] count={spike_times.size} first_ms={first_spike:.6f} "
        f"threshold_mV={SPIKE_THRESHOLD_MV}",
        flush=True,
    )

    payload = {
        "engine": "coreneuron_cpu",
        "dt": 0.025,
        "tstop": float(args.tstop),
        "stim_delay": 5.0,
        "stim_dur": 80.0,
        "stim_amp": 2.0,
        "spike_threshold": SPIKE_THRESHOLD_MV,
        "threads": int(args.threads),
        "build_s": build_s,
        "finitialize_s": init_s,
        "pre_run_s": pre_run_s,
        "run_s": run_s,
        "sum_s": sum_s,
        "num_cells_pyr": PYR_COUNT,
        "num_cells_sst": SST_COUNT,
        "num_cells_pv": PV_COUNT,
        "num_cells_vip": VIP_COUNT,
        "spike_source_count": total_cells,
        "spike_times": spike_times,
        "spike_gids": spike_gids,
        "spike_count": int(spike_times.size),
        "t": np.asarray(t_vec, dtype=float),
        "record_gids": np.asarray(sorted(v_vectors), dtype=int),
    }
    for gid, vector in sorted(v_vectors.items()):
        payload[str(gid)] = np.asarray(vector, dtype=float)
    np.savez(output, **payload)
    print(f"[output] saved traces: {output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
