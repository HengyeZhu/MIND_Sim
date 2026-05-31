#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from ca3_mind_sim_api import DEFAULT_RECORD, HERE, REPO_ROOT


def parse_values(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    return values


def value_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def add_common_runner_args(cmd: list[str], args: argparse.Namespace, combo: tuple[float, float, float, float]) -> None:
    olm, bas, pyr_bdend, pyr_adend3 = combo
    if args.paper_protocol:
        cmd.append("--paper-protocol")
    cmd.extend(["--duration-ms", f"{float(args.duration_ms):g}"])
    cmd.append("--connections" if args.connections else "--no-connections")
    cmd.append("--background-inputs" if args.background_inputs else "--no-background-inputs")
    cmd.append("--background-noise" if args.background_noise else "--no-background-noise")
    cmd.append("--medial-septal" if args.medial_septal else "--no-medial-septal")
    cmd.append("--washin-washout" if args.washin_washout else "--no-washin-washout")
    cmd.extend(
        [
            "--washin-ms",
            f"{float(args.washin_ms):g}",
            "--washout-ms",
            f"{float(args.washout_ms):g}",
            "--seed",
            str(int(args.seed)),
            "--wseed",
            str(int(args.wseed)),
            "--ms-gain",
            f"{float(args.ms_gain):g}",
            "--olm-soma-nmda",
            f"{float(olm):g}",
            "--bas-soma-nmda",
            f"{float(bas):g}",
            "--pyr-bdend-nmda",
            f"{float(pyr_bdend):g}",
            "--pyr-adend3-nmda",
            f"{float(pyr_adend3):g}",
            "--record",
            str(args.record),
        ]
    )


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an original-vs-MIND CA3 NMDA/AMPA ratio scan.")
    parser.add_argument("--values", type=parse_values, default=parse_values("0,1"))
    parser.add_argument("--olm-values", type=parse_values, default=None)
    parser.add_argument("--bas-values", type=parse_values, default=None)
    parser.add_argument("--pyr-bdend-values", type=parse_values, default=None)
    parser.add_argument("--pyr-adend3-values", type=parse_values, default=None)
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--paper-protocol", action="store_true")
    parser.add_argument("--connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--background-inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--medial-septal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ms-gain", type=float, default=1.0)
    parser.add_argument("--washin-washout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--washin-ms", type=float, default=1000.0)
    parser.add_argument("--washout-ms", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wseed", type=int, default=4321)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=os.environ.get("MIND_SIM_DEVICE", "cpu"))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MIND_SIM_NUM_THREADS", "1") or "1"))
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "nmda_scan")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--original-special", type=Path, default=HERE / "modeldb_186768" / "x86_64" / "special")
    parser.add_argument("--mind-python", default=sys.executable)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-mind", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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

    olm_values = args.olm_values or args.values
    bas_values = args.bas_values or args.values
    pyr_bdend_values = args.pyr_bdend_values or args.values
    pyr_adend3_values = args.pyr_adend3_values or args.values
    combos = list(itertools.product(olm_values, bas_values, pyr_bdend_values, pyr_adend3_values))

    output_dir = args.output_dir.expanduser().resolve()
    summary_path = (args.summary or output_dir / "ca3_nmda_scan_summary.json").expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_original and not args.original_special.exists():
        raise SystemExit(f"missing original NEURON executable: {args.original_special}")

    env = os.environ.copy()
    py_parts = [str(REPO_ROOT / "build"), str(REPO_ROOT / "src" / "python_api")]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    env.setdefault("MIND_SIM_CODEGEN_CACHE", "/tmp/mind_sim_codegen")

    summary = {
        "duration_ms": float(args.duration_ms),
        "combo_count": len(combos),
        "paper_protocol": bool(args.paper_protocol),
        "background_inputs": bool(args.background_inputs),
        "background_noise": bool(args.background_noise),
        "medial_septal": bool(args.medial_septal),
        "washin_washout": bool(args.washin_washout),
        "items": [],
    }

    for combo in combos:
        olm, bas, pyr_bdend, pyr_adend3 = combo
        tag = (
            f"olm{value_tag(olm)}_bas{value_tag(bas)}_"
            f"pyrb{value_tag(pyr_bdend)}_pyra{value_tag(pyr_adend3)}"
        )
        original_npz = output_dir / f"ca3_original_{tag}.npz"
        mind_npz = output_dir / f"ca3_mindsim_{tag}.npz"
        report_json = output_dir / f"ca3_compare_{tag}.json"

        if not args.skip_original:
            cmd = [str(args.original_special), "-python", str(HERE / "run_ca3_original_validation.py")]
            add_common_runner_args(cmd, args, combo)
            cmd.extend(["--output", str(original_npz)])
            run_command(cmd, env, args.dry_run)

        if not args.skip_mind:
            cmd = [str(args.mind_python), str(HERE / "ca3_mind_sim_api.py")]
            add_common_runner_args(cmd, args, combo)
            cmd.extend(["--device", args.device, "--num-threads", str(int(args.num_threads)), "--output", str(mind_npz)])
            run_command(cmd, env, args.dry_run)

        if not args.skip_compare and not args.dry_run:
            if original_npz.exists() and mind_npz.exists():
                cmd = [
                    str(args.mind_python),
                    str(HERE / "compare_ca3_validation.py"),
                    str(original_npz),
                    str(mind_npz),
                    "--bin-ms",
                    f"{float(args.bin_ms):g}",
                    "--output",
                    str(report_json),
                ]
                run_command(cmd, env, False)

        report = load_report(report_json)
        summary["items"].append(
            {
                "tag": tag,
                "olm_somaNMDA": float(olm),
                "bas_somaNMDA": float(bas),
                "pyr_BdendNMDA": float(pyr_bdend),
                "pyr_Adend3NMDA": float(pyr_adend3),
                "original": str(original_npz),
                "mind_sim": str(mind_npz),
                "report": str(report_json),
                "global_max_abs_mv": report.get("global_max_abs_mv"),
                "global_rms_mv": report.get("global_rms_mv"),
                "original_spikes": report.get("original_spikes"),
                "mind_sim_spikes": report.get("mind_sim_spikes"),
                "spike_gid_equal": report.get("spike_gid_equal"),
                "spike_time_max_abs_ms": report.get("spike_time_max_abs_ms"),
                "binned_spike_count_rms": report.get("binned_spike_count_rms"),
            }
        )

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if not args.dry_run:
        summary_path.write_text(text + "\n", encoding="utf-8")
        print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
