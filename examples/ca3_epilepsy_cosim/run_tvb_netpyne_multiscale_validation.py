#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TVB_ROOT = Path.home() / "tvb-root"
DEFAULT_TVB_MULTISCALE_ROOT = Path.home() / "tvb-multiscale"


def add_source_tree(tvb_root: Path, tvb_multiscale_root: Path) -> dict[str, str]:
    paths = [
        tvb_root / "tvb_library",
        tvb_root / "tvb_contrib",
        tvb_root / "tvb_storage",
        tvb_root / "tvb_framework",
        tvb_multiscale_root,
    ]
    for path in reversed(paths):
        if path.exists():
            sys.path.insert(0, str(path))
    return {
        "tvb_root": str(tvb_root),
        "tvb_multiscale_root": str(tvb_multiscale_root),
        "tvb_library_exists": str((tvb_root / "tvb_library").exists()),
        "tvb_multiscale_exists": str(tvb_multiscale_root.exists()),
    }


def patch_numpy_for_tvb_multiscale() -> None:
    # tvb-multiscale currently still references NumPy aliases removed by NumPy 2.
    for name, value in {"NAN": np.nan, "NaN": np.nan, "Inf": np.inf}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


@contextlib.contextmanager
def maybe_redirect_fds(enabled: bool, log_path: Path):
    if not enabled:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


@contextlib.contextmanager
def pushd(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def parse_proxy_inds(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def resolve_cli_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def array_stats(array: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        stats["finite"] = bool(np.isfinite(array).all())
        stats["min"] = float(np.nanmin(array))
        stats["max"] = float(np.nanmax(array))
    return stats


def summarize_results(results: list[Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        summary: dict[str, Any] = {"index": int(index), "type": type(item).__name__}
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            times = np.asarray(item[0], dtype=float)
            data = np.asarray(item[1], dtype=float)
            arrays[f"result_{index}_time_ms"] = times
            arrays[f"result_{index}_data"] = data
            summary["time"] = array_stats(times)
            summary["data"] = array_stats(data)
        else:
            try:
                arr = np.asarray(item)
            except Exception as exc:  # pragma: no cover - diagnostic only
                summary["array_error"] = f"{type(exc).__name__}: {exc}"
            else:
                arrays[f"result_{index}"] = arr
                summary["array"] = array_stats(arr)
        summaries.append(summary)
    return arrays, summaries


def summarize_netpyne() -> dict[str, Any]:
    try:
        from netpyne import sim as netpyne_sim
    except Exception as exc:  # pragma: no cover - depends on optional install
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    net = getattr(netpyne_sim, "net", None)
    cells = list(getattr(net, "allCells", []) or []) if net is not None else []
    pops = getattr(net, "allPops", {}) if net is not None else {}
    connection_count = 0
    for cell in cells:
        if isinstance(cell, dict):
            connection_count += len(cell.get("conns", []) or [])
    all_sim_data = getattr(netpyne_sim, "allSimData", {}) or {}
    spkt = np.asarray(all_sim_data.get("spkt", []), dtype=float)
    spkid = np.asarray(all_sim_data.get("spkid", []), dtype=np.int64)
    return {
        "available": True,
        "cells": int(len(cells)),
        "populations": int(len(pops)),
        "connections": int(connection_count),
        "spikes": int(spkt.size),
        "spike_time_min_ms": float(np.nanmin(spkt)) if spkt.size else None,
        "spike_time_max_ms": float(np.nanmax(spkt)) if spkt.size else None,
        "spike_gid_min": int(np.min(spkid)) if spkid.size else None,
        "spike_gid_max": int(np.max(spkid)) if spkid.size else None,
    }


def simulator_summary(simulator: Any) -> dict[str, Any]:
    integrator = getattr(simulator, "integrator", None)
    model = getattr(simulator, "model", None)
    return {
        "class": type(simulator).__name__,
        "model_class": type(model).__name__ if model is not None else None,
        "variables_of_interest": list(getattr(model, "variables_of_interest", ()) or ()),
        "number_of_nodes": int(getattr(simulator, "number_of_nodes", 0) or 0),
        "dt_ms": float(getattr(integrator, "dt", np.nan)) if integrator is not None else None,
        "simulation_length_ms": float(getattr(simulator, "simulation_length", np.nan)),
        "synchronization_time_ms": float(getattr(simulator, "synchronization_time", np.nan)),
        "proxy_inds": np.asarray(getattr(simulator, "proxy_inds", []), dtype=np.int64).tolist(),
        "out_proxy_count": int(np.asarray(getattr(simulator, "out_proxy_inds", [])).size),
        "exclusive": bool(getattr(simulator, "exclusive", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tvb-multiscale NetPyNE-TVB serial interface as an external cosim sanity check."
    )
    parser.add_argument("--tvb-root", type=Path, default=DEFAULT_TVB_ROOT)
    parser.add_argument("--tvb-multiscale-root", type=Path, default=DEFAULT_TVB_MULTISCALE_ROOT)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--population-order", type=int, default=2)
    parser.add_argument("--spiking-proxy-inds", default="0")
    parser.add_argument("--tvb-user-home", type=Path, default=Path("/tmp/tvb-user"))
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log", type=Path, default=HERE / "outputs" / "tvb_netpyne_multiscale.log")
    parser.add_argument("--workdir", type=Path, default=HERE / "outputs" / "tvb_netpyne_multiscale_workdir")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tvb_netpyne_multiscale_validation.npz")
    parser.add_argument("--report", type=Path, default=HERE / "outputs" / "tvb_netpyne_multiscale_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    if args.population_order <= 0:
        raise SystemExit("--population-order must be positive")

    os.environ.setdefault("TVB_USER_HOME", str(args.tvb_user_home.expanduser()))
    os.environ.setdefault("MPLBACKEND", "Agg")
    source_status = add_source_tree(args.tvb_root.expanduser(), args.tvb_multiscale_root.expanduser())
    patch_numpy_for_tvb_multiscale()
    proxy_inds = parse_proxy_inds(args.spiking_proxy_inds)

    output = resolve_cli_path(args.output)
    report_path = resolve_cli_path(args.report)
    log_path = resolve_cli_path(args.log)
    workdir = resolve_cli_path(args.workdir)

    start = time.perf_counter()
    with maybe_redirect_fds(bool(args.quiet), log_path), pushd(workdir):
        from examples.tvb_netpyne.example import default_example
        from tvb_multiscale.tvb_netpyne.config import Config

        results, simulator = default_example(
            simulation_length=float(args.duration_ms),
            population_order=int(args.population_order),
            spiking_proxy_inds=proxy_inds,
            plot_write=False,
            config=Config(output_base=str(workdir)),
        )
    elapsed_s = time.perf_counter() - start

    arrays, result_summary = summarize_results(results)
    netpyne_summary = summarize_netpyne()
    sim_summary = simulator_summary(simulator)
    finite_results = all(
        bool(item.get("data", item.get("array", {})).get("finite", True))
        for item in result_summary
    )
    passed = bool(result_summary and finite_results and netpyne_summary.get("available", False))
    report = {
        "passed": passed,
        "elapsed_s": float(elapsed_s),
        "source_status": source_status,
        "tvb_multiscale_workdir": str(workdir),
        "tvb_multiscale_reference": {
            "entrypoint": "examples.tvb_netpyne.example.default_example",
            "macro_model": "ReducedWongWangExcIOInhI",
            "spiking_builder": "DefaultExcIOInhIBuilder",
            "orchestrator": "TVBNetpyneSerialOrchestrator",
            "duration_ms_arg": float(args.duration_ms),
            "population_order": int(args.population_order),
            "spiking_proxy_inds": proxy_inds,
        },
        "simulator": sim_summary,
        "results": result_summary,
        "netpyne": netpyne_summary,
        "comparison_to_mind_vep_ca3": {
            "same_model": False,
            "numerical_precision_comparison_performed": False,
            "reason": (
                "This tvb-multiscale example uses ReducedWongWangExcIOInhI plus a default "
                "NetPyNE HH-soma network. The MIND Sim CA3 example uses a reduced VEP x-z "
                "macro scaffold plus the ModelDB 186768 CA3 PYR/BAS/OLM microcircuit and "
                "different bridge transforms."
            ),
            "required_for_precision_comparison": [
                "same macro equations and integrator",
                "same macro connectivity and delays",
                "same CA3 microcircuit implementation",
                "same TVB-to-micro and micro-to-TVB transforms",
                "same seeds and initial conditions",
            ],
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        report_json=np.asarray(json.dumps(report, sort_keys=True), dtype=object),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
