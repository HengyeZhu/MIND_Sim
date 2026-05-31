#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from ca3_mind_sim_api import (
    BAS_COUNT,
    DEFAULT_RECORD,
    DT_MS,
    HERE,
    OLM_COUNT,
    PYR_COUNT,
    RESULT_DIR,
    SPIKE_THRESHOLD_MV,
    V_INIT_MV,
    parse_record,
    population_spike_counts,
    population_spike_rates,
)


MODEL_DIR = HERE / "modeldb_186768"


class Population:
    def __init__(self, cell_type, count: int, gid_begin: int, x: float, amp: float):
        from neuron import h  # type: ignore

        self.cell = []
        self.count = int(count)
        self.gid_begin = int(gid_begin)
        for local in range(self.count):
            cell = cell_type(x + local * 50.0, 0.0, 0.0, self.gid_begin + local)
            cell.somaInj.amp = amp
            cell.somaInj.dur = 1.0e9
            cell.somaInj.delay = 2.0 * h.dt
            self.cell.append(cell)


def make_conn(pre_n: int, post_n: int, conv: int) -> np.ndarray:
    out = np.zeros((post_n, conv), dtype=np.int32)
    for post in range(post_n):
        out[post, :] = random.sample(list(range(pre_n)), conv)
    return out


def connect_projection(src: Population, trg: Population, syn: str, delay: float, weight: float, conv: int) -> list[object]:
    from neuron import h  # type: ignore

    conns = []
    matrix = make_conn(src.count, trg.count, conv)
    for post_id, pres in enumerate(matrix):
        for pre_id in pres:
            pre = src.cell[int(pre_id)]
            post = trg.cell[int(post_id)]
            conns.append(
                h.NetCon(
                    pre.soma(0.5)._ref_v,
                    post.__dict__[syn].syn,
                    SPIKE_THRESHOLD_MV,
                    delay,
                    weight,
                    sec=pre.soma,
                )
            )
    return conns


def set_original_nmda_r(pyr: Population, bas: Population, olm: Population, *, olm_r: float, bas_r: float, pyr_bdend: float, pyr_adend3: float) -> None:
    for cell in olm.cell:
        cell.somaNMDA.syn.r = float(olm_r)
    for cell in bas.cell:
        cell.somaNMDA.syn.r = float(bas_r)
    for cell in pyr.cell:
        cell.BdendNMDA.syn.r = float(pyr_bdend)
        cell.Adend3NMDA.syn.r = float(pyr_adend3)


def add_background_netstims(args: argparse.Namespace, pyr: Population, bas: Population, olm: Population) -> list[object]:
    from neuron import h  # type: ignore

    keepalive: list[object] = []

    def make_netstims(po: Population, syn: str, weight: float, isi_ms: float, seed_start: int) -> int:
        seed = int(seed_start)
        for cell in po.cell:
            ns = h.NetStim()
            ns.interval = float(isi_ms)
            ns.noise = 1.0 if args.background_noise else 0.0
            ns.number = (1000.0 / float(isi_ms)) * float(args.duration_ms)
            ns.start = 0.0
            nc = h.NetCon(ns, cell.__dict__[syn].syn)
            nc.delay = 2.0 * h.dt
            nc.weight[0] = float(weight)
            keepalive.extend([ns, nc])
            if args.background_noise:
                rds = h.Random()
                if args.random_engine == "random123":
                    rds.Random123(seed, seed, 0)
                else:
                    rds.MCellRan4(seed, seed)
                rds.negexp(1)
                ns.noiseFromRandom(rds)
                keepalive.append(rds)
            seed += 1
        return seed

    if args.background_inputs:
        seed = int(args.seed)
        seed = make_netstims(pyr, "somaAMPAf", 0.05e-3, 1.0, seed)
        seed = make_netstims(pyr, "Adend3AMPAf", 0.05e-3, 1.0, seed)
        seed = make_netstims(pyr, "somaGABAf", 0.012e-3, 1.0, seed)
        seed = make_netstims(pyr, "Adend3GABAf", 0.012e-3, 1.0, seed)
        seed = make_netstims(pyr, "Adend3NMDA", 6.5e-3, 100.0, seed)
        seed = make_netstims(bas, "somaAMPAf", 0.02e-3, 1.0, seed)
        seed = make_netstims(bas, "somaGABAf", 0.2e-3, 1.0, seed)
        seed = make_netstims(olm, "somaAMPAf", 0.0625e-3, 1.0, seed)
        make_netstims(olm, "somaGABAf", 0.2e-3, 1.0, seed)

    if args.medial_septal:
        ns = h.NetStim()
        ns.interval = 150.0
        ns.noise = 0.0
        ns.number = (1000.0 / 150.0) * float(args.duration_ms)
        # The upstream code leaves NetStim.start at NEURON's default, 50 ms.
        keepalive.append(ns)
        for cell in bas.cell:
            nc = h.NetCon(ns, cell.somaGABAss.syn)
            nc.delay = 2.0 * h.dt
            nc.weight[0] = 1.6e-3 * float(args.ms_gain)
            keepalive.append(nc)
        for cell in olm.cell:
            nc = h.NetCon(ns, cell.somaGABAss.syn)
            nc.delay = 2.0 * h.dt
            nc.weight[0] = 1.6e-3 * float(args.ms_gain)
            keepalive.append(nc)
    return keepalive


def gid_lookup(pyr: Population, bas: Population, olm: Population, psr: Population | None, gid: int):
    if 0 <= gid < PYR_COUNT:
        return pyr.cell[gid]
    if PYR_COUNT <= gid < PYR_COUNT + BAS_COUNT:
        return bas.cell[gid - PYR_COUNT]
    if PYR_COUNT + BAS_COUNT <= gid < PYR_COUNT + BAS_COUNT + OLM_COUNT:
        return olm.cell[gid - PYR_COUNT - BAS_COUNT]
    if psr is not None and gid == PYR_COUNT + BAS_COUNT + OLM_COUNT:
        return psr.cell[0]
    raise KeyError(gid)


def section_ref(cell, sec: str, loc: float):
    return cell.__dict__[sec](float(loc))._ref_v


def run_original(args: argparse.Namespace) -> Path:
    from neuron import h  # type: ignore

    output = args.output or RESULT_DIR / f"ca3_original_{int(args.duration_ms)}ms.npz"
    output = Path(output).expanduser()
    if not output.is_absolute():
        output = (HERE / output).resolve()
    else:
        output = output.resolve()

    os.chdir(MODEL_DIR)
    sys.path.insert(0, str(MODEL_DIR))
    h.load_file("stdrun.hoc")
    h.tstop = float(args.duration_ms)
    h.dt = DT_MS
    h.steps_per_ms = 1.0 / h.dt
    h.v_init = V_INIT_MV

    from geom import Bwb, Ow, PyrAdr  # type: ignore

    records = parse_record(args.record)
    include_psr = bool(args.include_psr) or any(gid >= PYR_COUNT + BAS_COUNT + OLM_COUNT for gid, _, _ in records)
    t0 = time.perf_counter()
    pyr = Population(PyrAdr, PYR_COUNT, 0, 0.0, 50e-3)
    bas = Population(Bwb, BAS_COUNT, PYR_COUNT, 10.0, 0.0)
    olm = Population(Ow, OLM_COUNT, PYR_COUNT + BAS_COUNT, 20.0, -25e-3)
    psr = Population(PyrAdr, 1, PYR_COUNT + BAS_COUNT + OLM_COUNT, 0.0, 50e-3) if include_psr else None
    netcons = []
    spike_recorders = []
    spike_times_parts = []
    spike_gid_parts = []
    for pop in (pyr, bas, olm) + ((psr,) if psr is not None else ()):
        for cell in pop.cell:
            times = h.Vector()
            gids = h.Vector()
            nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
            nc.threshold = SPIKE_THRESHOLD_MV
            nc.record(times, gids, int(cell.id))
            spike_recorders.append(nc)
            spike_times_parts.append(times)
            spike_gid_parts.append(gids)

    if args.connections:
        random.seed(int(args.wseed))
        netcons.extend(connect_projection(pyr, bas, "somaNMDA", 2.0, 1.15 * 1.2e-3, 100))
        netcons.extend(connect_projection(pyr, olm, "somaNMDA", 2.0, 0.7e-3, 10))
        netcons.extend(connect_projection(pyr, pyr, "BdendNMDA", 2.0, 0.004e-3, 25))
        netcons.extend(connect_projection(pyr, bas, "somaAMPAf", 2.0, 0.3 * 1.2e-3, 100))
        netcons.extend(connect_projection(pyr, olm, "somaAMPAf", 2.0, 0.3 * 1.2e-3, 10))
        netcons.extend(connect_projection(pyr, pyr, "BdendAMPA", 2.0, 0.5 * 0.04e-3, 25))
        netcons.extend(connect_projection(bas, bas, "somaGABAf", 2.0, 3.0 * 1.5e-3, 60))
        netcons.extend(connect_projection(bas, pyr, "somaGABAf", 2.0, 4.0 * 0.18e-3, 50))
        netcons.extend(connect_projection(bas, olm, "somaGABAf", 2.0, 0.05 * 4.0 * 0.18e-3, 17))
        netcons.extend(connect_projection(olm, pyr, "Adend2GABAs", 2.0, 0.08 * 4.0 * 3.0 * 6.0e-3, 10))
        if psr is not None:
            netcons.extend(connect_projection(pyr, psr, "BdendNMDA", 2.0, 0.004e-3, 25))
            netcons.extend(connect_projection(pyr, psr, "BdendAMPA", 2.0, 0.5 * 0.04e-3, 25))

    stimulus_keepalive = add_background_netstims(args, pyr, bas, olm)
    set_original_nmda_r(
        pyr,
        bas,
        olm,
        olm_r=float(args.olm_soma_nmda),
        bas_r=float(args.bas_soma_nmda),
        pyr_bdend=float(args.pyr_bdend_nmda),
        pyr_adend3=float(args.pyr_adend3_nmda),
    )

    voltage_vectors = []
    for gid, sec, loc in records:
        cell = gid_lookup(pyr, bas, olm, psr, gid)
        voltage_vectors.append(h.Vector().record(section_ref(cell, sec, loc)))
    time_vector = h.Vector().record(h._ref_t)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    h.finitialize(V_INIT_MV)
    washin_applied = False
    if args.washin_washout:
        first_stop = min(float(args.washin_ms), float(args.duration_ms))
        if first_stop > 0.0:
            h.continuerun(first_stop)
        if float(args.duration_ms) > float(args.washin_ms):
            set_original_nmda_r(pyr, bas, olm, olm_r=0.0, bas_r=1.0, pyr_bdend=1.0, pyr_adend3=1.0)
            second_stop = min(float(args.washout_ms), float(args.duration_ms))
            if second_stop > first_stop:
                h.continuerun(second_stop)
        if float(args.duration_ms) > float(args.washout_ms):
            set_original_nmda_r(pyr, bas, olm, olm_r=1.0, bas_r=1.0, pyr_bdend=1.0, pyr_adend3=1.0)
            h.continuerun(float(args.duration_ms))
        washin_applied = True
    else:
        h.continuerun(float(args.duration_ms))
    run_s = time.perf_counter() - t1

    spike_times = []
    spike_gids = []
    for times, gids in zip(spike_times_parts, spike_gid_parts):
        t = np.asarray(times.to_python(), dtype=float)
        g = np.asarray(gids.to_python(), dtype=np.int32)
        if t.size:
            spike_times.append(t)
            spike_gids.append(g)
    if spike_times:
        all_t = np.concatenate(spike_times)
        all_g = np.concatenate(spike_gids)
        order = np.lexsort((all_g, all_t))
        all_t = all_t[order]
        all_g = all_g[order]
    else:
        all_t = np.zeros(0, dtype=float)
        all_g = np.zeros(0, dtype=np.int32)

    labels = np.asarray([f"{gid}:{sec}:{loc:g}" for gid, sec, loc in records], dtype=object)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        backend=np.asarray("neuron_original", dtype=object),
        duration_ms=np.asarray(float(args.duration_ms)),
        dt_ms=np.asarray(DT_MS),
        connections=np.asarray(bool(args.connections)),
        record_labels=labels,
        times_ms=np.asarray(time_vector.to_python(), dtype=float),
        voltages_mv=np.vstack([np.asarray(v.to_python(), dtype=float) for v in voltage_vectors]),
        record_sample_times_ms=np.asarray(time_vector.to_python(), dtype=float),
        record_initial_mv=np.asarray([float(np.asarray(v.to_python(), dtype=float)[0]) for v in voltage_vectors], dtype=float),
        record_final_mv=np.asarray([float(np.asarray(v.to_python(), dtype=float)[-1]) for v in voltage_vectors], dtype=float),
        spike_times_ms=all_t,
        spike_gids=all_g,
        population_spike_counts_json=json.dumps(population_spike_counts(all_g), sort_keys=True),
        population_rate_hz_json=json.dumps(population_spike_rates(all_g, float(args.duration_ms)), sort_keys=True),
        timing_s=np.asarray([build_s, run_s, build_s + run_s], dtype=float),
        metadata_json=json.dumps(
            {
                "model": "Original ModelDB 186768 CA3 microcircuit",
                "protocol": "paper_protocol" if (args.background_inputs or args.medial_septal or args.washin_washout) else "deterministic_micro",
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
                "connections": bool(args.connections),
                "wseed": int(args.wseed),
                "seed": int(args.seed),
                "random_engine": str(args.random_engine),
                "record": [str(x) for x in labels],
                "stimulus_object_count": len(stimulus_keepalive),
            },
            sort_keys=True,
        ),
    )
    print(f"output={output}")
    print(f"build_s={build_s:.6f}")
    print(f"run_s={run_s:.6f}")
    print(f"spikes={len(all_t)}")
    print(f"netcons={len(netcons)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic original-NEURON validation for ModelDB 186768 CA3.")
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--paper-protocol", action="store_true", help="Enable the upstream-style 5 s background/MS/wash protocol.")
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
    parser.add_argument("--random-engine", choices=("random123", "mcellran4"), default="random123")
    parser.add_argument("--include-psr", action="store_true")
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, default=None)
    argv = list(sys.argv[1:])
    if "-python" in argv:
        idx = argv.index("-python")
        del argv[idx]
        if idx < len(argv) and not argv[idx].startswith("-"):
            del argv[idx]
    args = parser.parse_args(argv)
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
    if args.random_engine == "mcellran4" and args.background_noise:
        print("warning: NEURON 9 requires Random123 for NetStim.noiseFromRandom(); use --random-engine random123 if MCellRan4 fails.")
    run_original(args)


if __name__ == "__main__":
    main()
