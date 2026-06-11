#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare and plot CA3 all-in-one outputs.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plotdir", type=Path, required=True)
    parser.add_argument("--duration-ms", required=True)
    parser.add_argument("--mind-1", type=Path, required=True)
    parser.add_argument("--mind-4", type=Path, required=True)
    parser.add_argument("--tvb-neuron-1", type=Path, required=True)
    parser.add_argument("--tvb-neuron-4", type=Path, required=True)
    return parser.parse_args()


def as_labels(array: np.ndarray) -> list[str]:
    return [str(value) for value in array.tolist()]


def timing(record) -> dict[str, float]:
    values = np.asarray(record["timing_s"], dtype=float)
    return {
        "pre_run_s": float(values[0]),
        "run_s": float(values[1]),
    }


def finite_diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    diff = np.abs(left - right)
    valid = np.isfinite(diff) & ~(np.isnan(left) & np.isnan(right))
    if not np.any(valid):
        return 0.0, 0.0
    values = diff[valid]
    return float(np.max(values)), float(np.sqrt(np.mean(values * values)))


def array_max_detail(a: np.ndarray, b: np.ndarray, time: np.ndarray, labels: list[str]) -> dict[str, object]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    diff = np.abs(left - right)
    valid = np.isfinite(diff) & ~(np.isnan(left) & np.isnan(right))
    if not np.any(valid):
        return {
            "max_abs": 0.0,
            "rms": 0.0,
            "sample_index": None,
            "time_ms": None,
            "label": None,
        }
    masked = np.where(valid, diff, np.nan)
    index = np.unravel_index(int(np.nanargmax(masked)), diff.shape)
    values = diff[valid]
    return {
        "max_abs": float(diff[index]),
        "rms": float(np.sqrt(np.mean(values * values))),
        "sample_index": int(index[0]),
        "time_ms": float(time[index[0]]),
        "label": labels[int(index[1])],
        "mind_value": float(left[index]),
        "reference_value": float(right[index]),
    }


def voltage_trace(record, label: str) -> np.ndarray:
    labels = as_labels(record["voltage_labels"])
    return np.asarray(record["voltage_traces"], dtype=float)[:, labels.index(label)]


def spike_indices_from_voltage(voltage: np.ndarray) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=float)
    return np.flatnonzero((voltage[:-1] < 0.0) & (voltage[1:] >= 0.0)) + 1


def compare_pair(data, paths, mind_key: str, ref_key: str) -> dict[str, object]:
    mind = data[mind_key]
    ref = data[ref_key]
    roi_labels = as_labels(mind["labels"])
    voltage_labels = ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]

    result: dict[str, object] = {
        "mind_file": str(paths[mind_key]),
        "reference_file": str(paths[ref_key]),
        "macro_time_equal": bool(np.array_equal(mind["time_ms"], ref["time_ms"])),
        "macro_time_max_abs_ms": float(np.max(np.abs(mind["time_ms"] - ref["time_ms"]))),
        "voltage_time_equal": bool(np.array_equal(mind["voltage_trace_time"], ref["voltage_trace_time"])),
        "voltage_time_max_abs_ms": float(
            np.max(np.abs(mind["voltage_trace_time"] - ref["voltage_trace_time"]))
        ),
        "macro_x": array_max_detail(mind["macro_x"], ref["macro_x"], mind["time_ms"], roi_labels),
        "macro_z": array_max_detail(mind["macro_z"], ref["macro_z"], mind["time_ms"], roi_labels),
        "voltage": {},
        "spikes": {},
        "timing": {
            "mind": timing(mind),
            "reference": timing(ref),
        },
    }
    result["timing"]["speedup"] = (
        result["timing"]["reference"]["run_s"] / result["timing"]["mind"]["run_s"]
    )

    for voltage_label in voltage_labels:
        mind_voltage = voltage_trace(mind, voltage_label)
        ref_voltage = voltage_trace(ref, voltage_label)
        max_abs, rms = finite_diff_stats(mind_voltage, ref_voltage)
        mind_spikes = spike_indices_from_voltage(mind_voltage)
        ref_spikes = spike_indices_from_voltage(ref_voltage)
        spike_times_mind = np.asarray(mind["voltage_trace_time"], dtype=float)[mind_spikes]
        spike_times_ref = np.asarray(ref["voltage_trace_time"], dtype=float)[ref_spikes]
        result["voltage"][voltage_label] = {
            "max_abs_mV": max_abs,
            "rms_mV": rms,
        }
        raw_time_max = 0.0
        if mind_spikes.size == ref_spikes.size and mind_spikes.size > 0:
            raw_time_max = float(np.max(np.abs(spike_times_mind - spike_times_ref)))
        result["spikes"][voltage_label] = {
            "sample_indices_equal": bool(np.array_equal(mind_spikes, ref_spikes)),
            "mind_count": int(mind_spikes.size),
            "reference_count": int(ref_spikes.size),
            "raw_time_max_abs_ms": raw_time_max,
            "first_indices": [int(value) for value in mind_spikes[:10]],
            "last_indices": [int(value) for value in mind_spikes[-5:]],
        }

    if "adend3_voltage" in mind.files and "adend3_voltage" in ref.files:
        max_abs, rms = finite_diff_stats(mind["adend3_voltage"], ref["adend3_voltage"])
        result["voltage"]["PYR[0].Adend3"] = {
            "max_abs_mV": max_abs,
            "rms_mV": rms,
        }

    return result


def compare_same_backend(data, paths, left_key: str, right_key: str) -> dict[str, object]:
    left = data[left_key]
    right = data[right_key]
    roi_labels = as_labels(left["labels"])
    result = {
        "left_file": str(paths[left_key]),
        "right_file": str(paths[right_key]),
        "macro_x": array_max_detail(left["macro_x"], right["macro_x"], left["time_ms"], roi_labels),
        "macro_z": array_max_detail(left["macro_z"], right["macro_z"], left["time_ms"], roi_labels),
        "voltage": {},
        "spikes": {},
    }
    for voltage_label in ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]:
        left_voltage = voltage_trace(left, voltage_label)
        right_voltage = voltage_trace(right, voltage_label)
        max_abs, rms = finite_diff_stats(left_voltage, right_voltage)
        left_spikes = spike_indices_from_voltage(left_voltage)
        right_spikes = spike_indices_from_voltage(right_voltage)
        result["voltage"][voltage_label] = {"max_abs_mV": max_abs, "rms_mV": rms}
        result["spikes"][voltage_label] = {
            "sample_indices_equal": bool(np.array_equal(left_spikes, right_spikes)),
            "left_count": int(left_spikes.size),
            "right_count": int(right_spikes.size),
        }
    if "adend3_voltage" in left.files and "adend3_voltage" in right.files:
        max_abs, rms = finite_diff_stats(left["adend3_voltage"], right["adend3_voltage"])
        result["voltage"]["PYR[0].Adend3"] = {"max_abs_mV": max_abs, "rms_mV": rms}
    return result


def build_summary(args: argparse.Namespace, data, paths) -> dict[str, object]:
    comparisons = {
        "mind_vs_tvb_neuron_1thread": compare_pair(data, paths, "mind_1thread", "tvb_neuron_1thread"),
        "mind_vs_tvb_neuron_4thread": compare_pair(data, paths, "mind_4thread", "tvb_neuron_4thread"),
        "mind_1thread_vs_4thread": compare_same_backend(data, paths, "mind_1thread", "mind_4thread"),
        "tvb_neuron_1thread_vs_4thread": compare_same_backend(
            data, paths, "tvb_neuron_1thread", "tvb_neuron_4thread"
        ),
    }
    return {
        "duration_ms": float(args.duration_ms),
        "outputs": {label: str(path) for label, path in paths.items()},
        "comparisons": comparisons,
    }


def build_markdown(summary: dict[str, object], json_path: Path, plotdir: Path) -> list[str]:
    duration_ms = summary["duration_ms"]
    comparisons = summary["comparisons"]
    md_lines = [
        f"# CA3 all-in-one comparison ({duration_ms:g} ms)",
        "",
        "## Timing",
        "",
        "| Workflow | Threads | Pre-run | Run | Speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, thread in [
        ("mind_vs_tvb_neuron_1thread", 1),
        ("mind_vs_tvb_neuron_4thread", 4),
    ]:
        item = comparisons[name]
        md_lines.append(
            f"| MIND_Sim async | {thread} | "
            f"{item['timing']['mind']['pre_run_s']:.3f}s | "
            f"{item['timing']['mind']['run_s']:.3f}s | "
            f"{item['timing']['speedup']:.2f}x |"
        )
        md_lines.append(
            f"| TVB+NEURON | {thread} | "
            f"{item['timing']['reference']['pre_run_s']:.3f}s | "
            f"{item['timing']['reference']['run_s']:.3f}s | 1.00x |"
        )
    md_lines.extend([
        "",
        "## MIND_Sim vs TVB+NEURON precision",
        "",
        "| Threads | macro x max | macro z max | max voltage error | spike sample indices |",
        "| ---: | ---: | ---: | ---: | --- |",
    ])
    for name, thread in [
        ("mind_vs_tvb_neuron_1thread", 1),
        ("mind_vs_tvb_neuron_4thread", 4),
    ]:
        item = comparisons[name]
        voltage_max = max(metric["max_abs_mV"] for metric in item["voltage"].values())
        spike_equal = all(metric["sample_indices_equal"] for metric in item["spikes"].values())
        md_lines.append(
            f"| {thread} | {item['macro_x']['max_abs']:.6g} | "
            f"{item['macro_z']['max_abs']:.6g} | {voltage_max:.6g} mV | "
            f"{'exactly equal' if spike_equal else 'different'} |"
        )
    md_lines.extend([
        "",
        "## Generated files",
        "",
        f"- JSON summary: `{json_path}`",
        f"- Plot directory: `{plotdir}`",
    ])
    return md_lines


def save_macro_plot(data, plotdir: Path, thread: str, mind_key: str, ref_key: str) -> Path:
    mind = data[mind_key]
    ref = data[ref_key]
    labels = as_labels(mind["labels"])
    ca3 = labels.index("Left-CA3")
    time = np.asarray(mind["time_ms"], dtype=float)
    mind_x = np.asarray(mind["macro_x"], dtype=float)[:, ca3]
    ref_x = np.asarray(ref["macro_x"], dtype=float)[:, ca3]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, mind_x, label="MIND_Sim", linewidth=1.5)
    axes[0].plot(time, ref_x, "--", label="TVB+NEURON", linewidth=1.2)
    axes[0].set_ylabel("Left-CA3 x")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, mind_x - ref_x, color="black", linewidth=1.0)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Difference")
    axes[1].grid(True, alpha=0.25)
    path = plotdir / f"macro_left_ca3_x_{thread}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_voltage_plot(data, plotdir: Path, thread: str, mind_key: str, ref_key: str) -> Path:
    mind = data[mind_key]
    ref = data[ref_key]
    mind_t = np.asarray(mind["voltage_trace_time"], dtype=float)
    ref_t = np.asarray(ref["voltage_trace_time"], dtype=float)
    names = ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, names):
        axis.plot(mind_t, voltage_trace(mind, name), label="MIND_Sim", linewidth=1.0)
        axis.plot(ref_t, voltage_trace(ref, name), "--", label="TVB+NEURON", linewidth=0.9)
        axis.set_ylabel(f"{name}\n(mV)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time (ms)")
    path = plotdir / f"voltage_overlay_{thread}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_precision_plot(summary: dict[str, object], plotdir: Path) -> Path:
    labels = ["macro x", "macro z", "PYR soma", "BAS soma", "OLM soma", "PYR Adend3"]
    comparisons = summary["comparisons"]
    values = []
    for name in ["mind_vs_tvb_neuron_1thread", "mind_vs_tvb_neuron_4thread"]:
        item = comparisons[name]
        values.append(
            [
                item["macro_x"]["max_abs"],
                item["macro_z"]["max_abs"],
                item["voltage"]["PYR[0].soma"]["max_abs_mV"],
                item["voltage"]["BAS[0].soma"]["max_abs_mV"],
                item["voltage"]["OLM[0].soma"]["max_abs_mV"],
                item["voltage"]["PYR[0].Adend3"]["max_abs_mV"],
            ]
        )
    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels))
    width = 0.35

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.bar(x - width / 2, values[0], width, label="1 thread")
    axis.bar(x + width / 2, values[1], width, label="4 threads")
    axis.set_yscale("log")
    axis.set_ylabel("Max absolute error")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.legend(loc="best")
    axis.grid(True, axis="y", alpha=0.25)
    path = plotdir / "precision_max_abs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_timing_plot(summary: dict[str, object], plotdir: Path) -> Path:
    comparisons = summary["comparisons"]
    labels = ["MIND 1T", "TVB+NEURON 1T", "MIND 4T", "TVB+NEURON 4T"]
    values = [
        comparisons["mind_vs_tvb_neuron_1thread"]["timing"]["mind"]["run_s"],
        comparisons["mind_vs_tvb_neuron_1thread"]["timing"]["reference"]["run_s"],
        comparisons["mind_vs_tvb_neuron_4thread"]["timing"]["mind"]["run_s"],
        comparisons["mind_vs_tvb_neuron_4thread"]["timing"]["reference"]["run_s"],
    ]

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    axis.bar(labels, values, color=["#4477aa", "#cc6677", "#4477aa", "#cc6677"])
    axis.set_ylabel("Run time (s)")
    axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
    path = plotdir / "timing_run_seconds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_plots(data, summary: dict[str, object], plotdir: Path) -> list[Path]:
    return [
        save_macro_plot(data, plotdir, "1thread", "mind_1thread", "tvb_neuron_1thread"),
        save_macro_plot(data, plotdir, "4thread", "mind_4thread", "tvb_neuron_4thread"),
        save_voltage_plot(data, plotdir, "1thread", "mind_1thread", "tvb_neuron_1thread"),
        save_voltage_plot(data, plotdir, "4thread", "mind_4thread", "tvb_neuron_4thread"),
        save_precision_plot(summary, plotdir),
        save_timing_plot(summary, plotdir),
    ]


def main() -> None:
    args = parse_args()
    paths = {
        "mind_1thread": args.mind_1,
        "mind_4thread": args.mind_4,
        "tvb_neuron_1thread": args.tvb_neuron_1,
        "tvb_neuron_4thread": args.tvb_neuron_4,
    }
    for label, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing {label} output: {path}")

    data = {label: np.load(path, allow_pickle=True) for label, path in paths.items()}
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.plotdir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args, data, paths)
    json_path = args.outdir / f"compare_{args.duration_ms}ms.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = build_markdown(summary, json_path, args.plotdir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        plot_paths = save_plots(data, summary, args.plotdir)
    except Exception as exc:  # pragma: no cover - depends on local env
        md_lines.extend([
            "",
            "## Plotting",
            "",
            f"Plots were skipped because matplotlib could not be imported: `{exc}`",
        ])
        md_path = args.outdir / f"compare_{args.duration_ms}ms.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"summary_json={json_path}")
        print(f"summary_md={md_path}")
        print(f"plot_status=skipped: {exc}")
        return

    md_lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        md_lines.append(f"- `{path}`")

    md_path = args.outdir / f"compare_{args.duration_ms}ms.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"summary_json={json_path}")
    print(f"summary_md={md_path}")
    print(f"plot_dir={args.plotdir}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
