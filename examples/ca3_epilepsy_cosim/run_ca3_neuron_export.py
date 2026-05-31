#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "modeldb_186768"
SOURCE_COMMIT = "e60d5e99eb2a6836485b35cc3f8aae0085bee79f"

POPULATIONS = {
    "pyr": (0, 800),
    "bas": (800, 1000),
    "olm": (1000, 1200),
}


def install_random123_netstim_compat(network_module: object) -> None:
    hoc = network_module.h

    def init_netstims(self) -> None:
        for i in range(len(self.nrl)):
            rds = self.nrl[i]
            seed = int(self.nrlsead[i])
            rds.Random123(seed, seed, 0)
            rds.negexp(1)

    def make_netstims(self, po, syn, w, ISI, time_limit, sead):
        po.nssidx[syn] = len(self.nsl)
        po.ncsidx[syn] = len(self.ncl)
        for i in range(po.n):
            cel = po.cell[i]
            ns = hoc.NetStim()
            ns.interval = ISI
            ns.noise = 1
            ns.number = (1e3 / ISI) * time_limit
            ns.start = 0

            nc = hoc.NetCon(ns, cel.__dict__[syn].syn)
            nc.delay = hoc.dt * 2
            nc.weight[0] = w

            rds = hoc.Random()
            seed = int(sead)
            rds.Random123(seed, seed, 0)
            rds.negexp(1)
            ns.noiseFromRandom(rds)

            self.nsl.append(ns)
            self.ncl.append(nc)
            self.nrl.append(rds)
            self.nrlsead.append(seed)
            sead = seed + 1

        po.nseidx[syn] = len(self.nsl) - 1
        po.nceidx[syn] = len(self.ncl) - 1
        return sead

    network_module.Network.init_NetStims = init_netstims
    network_module.Network.make_NetStims = make_netstims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vendored CA3 epilepsy NEURON model and export training traces.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--duration-ms", type=float, default=3000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "ca3_baseline.npz")
    parser.add_argument("--rate-bin-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--ms-gain", type=float, default=1.0)
    parser.add_argument("--olm-soma-nmda", type=float, default=1.0)
    parser.add_argument("--bas-soma-nmda", type=float, default=1.0)
    parser.add_argument("--pyr-bdend-nmda", type=float, default=1.0)
    parser.add_argument("--pyr-adend3-nmda", type=float, default=1.0)
    argv = list(sys.argv[1:])
    if "-python" in argv:
        idx = argv.index("-python")
        del argv[idx]
        if idx < len(argv) and not argv[idx].startswith("-"):
            del argv[idx]
    return parser.parse_args(argv)


def population_rate(
    spike_times_ms: np.ndarray,
    spike_gids: np.ndarray,
    gid_start: int,
    gid_stop: int,
    duration_ms: float,
    bin_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, duration_ms + bin_ms, bin_ms, dtype=float)
    if edges.size < 2 or edges[-1] < duration_ms:
        edges = np.append(edges, duration_ms)
    mask = (spike_gids >= gid_start) & (spike_gids < gid_stop)
    counts, _ = np.histogram(spike_times_ms[mask], bins=edges)
    cell_count = max(1, gid_stop - gid_start)
    widths_s = np.diff(edges) / 1000.0
    rates_hz = counts.astype(float) / widths_s / float(cell_count)
    centers_ms = 0.5 * (edges[:-1] + edges[1:])
    return centers_ms, rates_hz


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = (HERE / output_path).resolve()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))

    try:
        from neuron import h  # type: ignore
    except ImportError as exc:
        raise SystemExit("NEURON Python module is required. Run this through NEURON's special binary.") from exc

    h("strdef simname, allfiles, simfiles, output_file, datestr, uname, osname, comment")
    h.simname = "ca3_epilepsy_modeldb_186768"
    h.allfiles = "geom.hoc pyinit.py geom.py network.py params.py run.py"
    h.simfiles = "pyinit.py geom.py network.py params.py run.py"
    h("runnum=1")
    h.datestr = "31may26"
    h.output_file = "outputs/ca3_export"
    h.uname = "x86_64"
    h.osname = "linux"
    h("templates_loaded=0")
    h("xwindows=0")

    h.xopen("nrnoc.hoc")
    h.xopen("init.hoc")

    h.tstop = float(args.duration_ms)
    h.dt = float(args.dt_ms)
    h.steps_per_ms = 1.0 / h.dt
    h.v_init = -65.0

    rseed_path = model_dir / "rseed.txt"
    old_rseed = rseed_path.read_text(encoding="utf-8") if rseed_path.exists() else None
    rseed_path.write_text(f"{int(args.seed)}\n{int(args.wseed)}\n{float(args.ms_gain)}\n", encoding="utf-8")
    try:
        import networkmsj  # type: ignore
    finally:
        if old_rseed is None:
            try:
                rseed_path.unlink()
            except FileNotFoundError:
                pass
        else:
            rseed_path.write_text(old_rseed, encoding="utf-8")

    install_random123_netstim_compat(networkmsj)
    net = networkmsj.net
    net.olm.set_r("somaNMDA", float(args.olm_soma_nmda))
    net.bas.set_r("somaNMDA", float(args.bas_soma_nmda))
    net.pyr.set_r("BdendNMDA", float(args.pyr_bdend_nmda))
    net.pyr.set_r("Adend3NMDA", float(args.pyr_adend3_nmda))
    net.set_noise_inputs(h.tstop)

    def init_netstims() -> None:
        net.init_NetStims()

    init_handler = h.FInitializeHandler(0, init_netstims)
    if init_handler is None:
        raise RuntimeError("failed to register NetStim initialization handler")
    h.run()

    net.setrastervecs()
    net.calc_lfp()

    spike_times_ms = np.asarray(net.mytimevec.to_python(), dtype=float)
    spike_gids = np.asarray(net.myidvec.to_python(), dtype=np.int32)
    order = np.argsort(spike_times_ms, kind="mergesort")
    spike_times_ms = spike_times_ms[order]
    spike_gids = spike_gids[order]

    lfp = np.asarray(net.lfp, dtype=float)
    time_ms = np.arange(lfp.size, dtype=float) * float(h.dt)

    rate_time_ms = None
    rates: dict[str, np.ndarray] = {}
    for name, (start, stop) in POPULATIONS.items():
        centers, rate = population_rate(
            spike_times_ms,
            spike_gids,
            start,
            stop,
            duration_ms=float(args.duration_ms),
            bin_ms=float(args.rate_bin_ms),
        )
        if rate_time_ms is None:
            rate_time_ms = centers
        rates[name] = rate

    metadata = {
        "source": "ModelDB 186768",
        "source_commit": SOURCE_COMMIT,
        "model": "CA3 epileptic activity network",
        "duration_ms": float(args.duration_ms),
        "dt_ms": float(h.dt),
        "rate_bin_ms": float(args.rate_bin_ms),
        "seed": int(args.seed),
        "wseed": int(args.wseed),
        "ms_gain": float(args.ms_gain),
        "nmda_ratios": {
            "olm_soma": float(args.olm_soma_nmda),
            "bas_soma": float(args.bas_soma_nmda),
            "pyr_bdend": float(args.pyr_bdend_nmda),
            "pyr_adend3": float(args.pyr_adend3_nmda),
        },
        "populations": POPULATIONS,
    }

    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time_ms=time_ms,
        lfp=lfp,
        spike_times_ms=spike_times_ms,
        spike_gids=spike_gids,
        rate_time_ms=np.asarray(rate_time_ms, dtype=float),
        rate_pyr_hz=rates["pyr"],
        rate_bas_hz=rates["bas"],
        rate_olm_hz=rates["olm"],
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
