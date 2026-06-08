#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROIS = [
    "Left-CA3",
    "Right-CA3",
    "Left-subiculum",
    "Right-subiculum",
    "ctx-lh-entorhinal",
]


def load_macro(raw: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    if "macro_records" in raw:
        return (
            np.asarray(raw["macro_records"], dtype=float),
            [str(name) for name in np.asarray(raw["record_names"], dtype=object).tolist()],
        )
    return (
        np.asarray(raw["macro_exposures"], dtype=float),
        [str(name) for name in np.asarray(raw["exposure_names"], dtype=object).tolist()],
    )


def common_time_indices(time_a: np.ndarray, time_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounded_a = np.round(np.asarray(time_a, dtype=float), 9)
    rounded_b = np.round(np.asarray(time_b, dtype=float), 9)
    common, index_a, index_b = np.intersect1d(rounded_a, rounded_b, return_indices=True)
    return common.astype(float), index_a, index_b


def thin_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_points).astype(int))


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def metrics(diff: np.ndarray) -> tuple[float, float]:
    finite = finite_values(diff)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.max(np.abs(finite))), float(np.sqrt(np.mean(finite * finite)))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def choose_rois(labels: list[str], requested: list[str], count: int) -> list[str]:
    selected: list[str] = []
    for label in [*requested, *DEFAULT_ROIS]:
        if label in labels and label not in selected:
            selected.append(label)
        if len(selected) >= count:
            return selected

    if len(selected) < count:
        for index in np.linspace(0, len(labels) - 1, count * 2).astype(int):
            label = labels[int(index)]
            if label not in selected:
                selected.append(label)
            if len(selected) >= count:
                break
    return selected


def style_axis(ax) -> None:
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pair(
    output: Path,
    title: str,
    ylabel: str,
    time_ms: np.ndarray,
    mind: np.ndarray,
    tvb: np.ndarray,
    max_points: int,
) -> None:
    keep = thin_indices(time_ms.size, max_points)
    t = np.asarray(time_ms, dtype=float)[keep]
    mind_trace = np.asarray(mind, dtype=float)[keep]
    tvb_trace = np.asarray(tvb, dtype=float)[keep]
    diff = mind_trace - tvb_trace
    max_abs, rms = metrics(diff)

    fig, (ax_overlay, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if finite_values(mind_trace).size or finite_values(tvb_trace).size:
        ax_overlay.plot(t, mind_trace, label="MIND_Sim", color="#2563eb", linewidth=1.4)
        ax_overlay.plot(t, tvb_trace, label="TVB-NetPyNE", color="#f97316", linewidth=1.2, linestyle="--")
    else:
        ax_overlay.text(0.5, 0.5, "no finite samples", transform=ax_overlay.transAxes, ha="center", va="center")

    if finite_values(diff).size:
        ax_diff.plot(t, diff, color="#dc2626", linewidth=1.1)
        ax_diff.axhline(0.0, color="#64748b", linewidth=0.8)
    else:
        ax_diff.text(0.5, 0.5, "no finite diff samples", transform=ax_diff.transAxes, ha="center", va="center")

    ax_overlay.set_ylabel(ylabel)
    ax_diff.set_ylabel("diff")
    ax_diff.set_xlabel("time (ms)")
    ax_overlay.legend(loc="best", frameon=False)
    ax_overlay.set_title("overlay", loc="left", fontsize=11)
    ax_diff.set_title(f"MIND_Sim - TVB-NetPyNE   max_abs={max_abs:.6g}, rms={rms:.6g}", loc="left", fontsize=11)
    style_axis(ax_overlay)
    style_axis(ax_diff)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MIND_Sim vs TVB-NetPyNE CA3 co-simulation outputs.")
    parser.add_argument("mindsim", type=Path)
    parser.add_argument("tvb_netpyne", type=Path)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--roi", action="append", default=[], help="ROI label to plot. Can be passed multiple times.")
    parser.add_argument("--roi-count", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mind_raw = np.load(args.mindsim, allow_pickle=True)
    tvb_raw = np.load(args.tvb_netpyne, allow_pickle=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(label) for label in np.asarray(mind_raw["labels"], dtype=object).tolist()]
    rois = choose_rois(labels, args.roi, int(args.roi_count))
    mind_macro, mind_names = load_macro(mind_raw)
    tvb_macro, tvb_names = load_macro(tvb_raw)
    if mind_names != tvb_names:
        raise RuntimeError(f"macro variable mismatch: {mind_names} vs {tvb_names}")

    macro_time, mind_index, tvb_index = common_time_indices(mind_raw["time_ms"], tvb_raw["time_ms"])
    mind_macro = mind_macro[mind_index]
    tvb_macro = tvb_macro[tvb_index]

    for roi_label in rois:
        roi_index = labels.index(roi_label)
        for variable_index, variable in enumerate(mind_names):
            output = output_dir / f"{safe_name(roi_label)}_{safe_name(variable)}.{args.format}"
            plot_pair(
                output,
                f"{roi_label} / {variable}",
                variable,
                macro_time,
                mind_macro[:, roi_index, variable_index],
                tvb_macro[:, roi_index, variable_index],
                args.max_points,
            )
            print(output)

    if (
        "voltage_time" in mind_raw
        and "voltage" in mind_raw
        and "voltage_time" in tvb_raw
        and "voltage" in tvb_raw
        and np.asarray(mind_raw["voltage_time"]).size > 1
        and np.asarray(tvb_raw["voltage_time"]).size > 1
        and np.asarray(mind_raw["voltage"]).size > 1
        and np.asarray(tvb_raw["voltage"]).size > 1
    ):
        voltage_time, mind_voltage_index, tvb_voltage_index = common_time_indices(
            mind_raw["voltage_time"],
            tvb_raw["voltage_time"],
        )
        output = output_dir / f"voltage.{args.format}"
        plot_pair(
            output,
            "first PYR soma voltage",
            "mV",
            voltage_time,
            np.asarray(mind_raw["voltage"], dtype=float)[mind_voltage_index],
            np.asarray(tvb_raw["voltage"], dtype=float)[tvb_voltage_index],
            args.max_points,
        )
        print(output)


if __name__ == "__main__":
    main()
