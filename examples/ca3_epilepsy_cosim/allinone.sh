#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DURATION_MS="${DURATION_MS:-1000}"
MIND_ENV="${MIND_ENV:-mind_sim}"
TVB_ENV="${TVB_ENV:-test}"
CONNECTIVITY_CSV="${CONNECTIVITY_CSV:-${SCRIPT_DIR}/data/synthetic_hybrid_ca3_connectivity.csv}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outputs/allinone_${DURATION_MS}ms}"
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
CLEAN_TVB_MOD="${CLEAN_TVB_MOD:-0}"

MIND_SCRIPT="${SCRIPT_DIR}/mind_sim/run_vep_ca3_cosim.py"
TVB_NEURON_SCRIPT="${SCRIPT_DIR}/neuron_tvb/run_tvb_neuron_ca3_cosim.py"
LOGDIR="${OUTDIR}/logs"
PLOTDIR="${OUTDIR}/plots"

MIND_1="${MIND_1:-${OUTDIR}/mind_${DURATION_MS}ms_1thread.npz}"
MIND_4="${MIND_4:-${OUTDIR}/mind_${DURATION_MS}ms_4thread.npz}"
TVB_1="${TVB_1:-${OUTDIR}/tvb_neuron_${DURATION_MS}ms_1thread.npz}"
TVB_4="${TVB_4:-${OUTDIR}/tvb_neuron_${DURATION_MS}ms_4thread.npz}"

mkdir -p "${OUTDIR}" "${LOGDIR}" "${PLOTDIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is required on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

run_mind() {
  local threads="$1"
  local output="$2"
  local log="${LOGDIR}/mind_${DURATION_MS}ms_${threads}thread.log"

  echo
  echo "== MIND_Sim async, ${threads} thread(s) =="
  conda activate "${MIND_ENV}"
  unset MIND_SIM_FORCE_SERIAL_PIPELINE
  python "${MIND_SCRIPT}" \
    --connectivity-csv "${CONNECTIVITY_CSV}" \
    --duration-ms "${DURATION_MS}" \
    --micro-threads "${threads}" \
    --output "${output}" 2>&1 | tee "${log}"
  conda deactivate
}

run_tvb_neuron() {
  local threads="$1"
  local output="$2"
  local log="${LOGDIR}/tvb_neuron_${DURATION_MS}ms_${threads}thread.log"

  echo
  echo "== TVB+NEURON, ${threads} thread(s) =="
  conda activate "${TVB_ENV}"
  if [[ "${CLEAN_TVB_MOD}" == "1" ]]; then
    rm -rf "${SCRIPT_DIR}/neuron_tvb/mod/x86_64" \
           "${SCRIPT_DIR}/neuron_tvb/mod/i686" \
           "${SCRIPT_DIR}/neuron_tvb/mod/aarch64" \
           "${SCRIPT_DIR}/neuron_tvb/mod/arm64"
  fi
  python "${TVB_NEURON_SCRIPT}" \
    --connectivity-csv "${CONNECTIVITY_CSV}" \
    --duration-ms "${DURATION_MS}" \
    --micro-threads "${threads}" \
    --output "${output}" \
    --workdir "${OUTDIR}/tvb_neuron_workdir_${threads}thread" 2>&1 | tee "${log}"
  conda deactivate
}

compare_and_plot() {
  echo
  echo "== Compare and plot =="
  conda activate "${TVB_ENV}"
  python - "${OUTDIR}" "${PLOTDIR}" "${DURATION_MS}" "${MIND_1}" "${MIND_4}" "${TVB_1}" "${TVB_4}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

outdir = Path(sys.argv[1])
plotdir = Path(sys.argv[2])
duration_ms = sys.argv[3]
paths = {
    "mind_1thread": Path(sys.argv[4]),
    "mind_4thread": Path(sys.argv[5]),
    "tvb_neuron_1thread": Path(sys.argv[6]),
    "tvb_neuron_4thread": Path(sys.argv[7]),
}

for label, path in paths.items():
    if not path.exists():
        raise SystemExit(f"missing {label} output: {path}")

data = {label: np.load(path, allow_pickle=True) for label, path in paths.items()}


def as_labels(array: np.ndarray) -> list[str]:
    return [str(value) for value in array.tolist()]


def timing(record) -> dict[str, float]:
    values = np.asarray(record["timing_s"], dtype=float)
    return {
        "pre_run_s": float(values[0]),
        "run_s": float(values[1]),
    }


def finite_diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    valid = np.isfinite(diff) & ~(np.isnan(a) & np.isnan(b))
    if not np.any(valid):
        return 0.0, 0.0
    values = diff[valid]
    return float(np.max(values)), float(np.sqrt(np.mean(values * values)))


def array_max_detail(a: np.ndarray, b: np.ndarray, time: np.ndarray, labels: list[str]) -> dict[str, object]:
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    valid = np.isfinite(diff) & ~(np.isnan(a) & np.isnan(b))
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
        "mind_value": float(a[index]),
        "reference_value": float(b[index]),
    }


def voltage_trace(record, label: str) -> np.ndarray:
    labels = as_labels(record["voltage_labels"])
    return np.asarray(record["voltage_traces"], dtype=float)[:, labels.index(label)]


def spike_indices_from_voltage(voltage: np.ndarray) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=float)
    return np.flatnonzero((voltage[:-1] < 0.0) & (voltage[1:] >= 0.0)) + 1


def compare_pair(label: str, mind_key: str, ref_key: str) -> dict[str, object]:
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


def compare_same_backend(label: str, left_key: str, right_key: str) -> dict[str, object]:
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


comparisons = {
    "mind_vs_tvb_neuron_1thread": compare_pair("1thread", "mind_1thread", "tvb_neuron_1thread"),
    "mind_vs_tvb_neuron_4thread": compare_pair("4thread", "mind_4thread", "tvb_neuron_4thread"),
    "mind_1thread_vs_4thread": compare_same_backend("mind", "mind_1thread", "mind_4thread"),
    "tvb_neuron_1thread_vs_4thread": compare_same_backend(
        "tvb_neuron", "tvb_neuron_1thread", "tvb_neuron_4thread"
    ),
}

summary = {
    "duration_ms": float(duration_ms),
    "outputs": {label: str(path) for label, path in paths.items()},
    "comparisons": comparisons,
}

outdir.mkdir(parents=True, exist_ok=True)
plotdir.mkdir(parents=True, exist_ok=True)
json_path = outdir / f"compare_{duration_ms}ms.json"
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

md_lines = [
    f"# CA3 all-in-one comparison ({duration_ms} ms)",
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - depends on local env
    md_lines.extend([
        "",
        "## Plotting",
        "",
        f"Plots were skipped because matplotlib could not be imported: `{exc}`",
    ])
    (outdir / f"compare_{duration_ms}ms.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"summary_json={json_path}")
    print(f"summary_md={outdir / f'compare_{duration_ms}ms.md'}")
    print(f"plot_status=skipped: {exc}")
    raise SystemExit(0)


def save_macro_plot(thread: str, mind_key: str, ref_key: str) -> Path:
    mind = data[mind_key]
    ref = data[ref_key]
    labels = as_labels(mind["labels"])
    ca3 = labels.index("Left-CA3")
    time = np.asarray(mind["time_ms"], dtype=float)
    mind_x = np.asarray(mind["macro_x"], dtype=float)[:, ca3]
    ref_x = np.asarray(ref["macro_x"], dtype=float)[:, ca3]

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


def save_voltage_plot(thread: str, mind_key: str, ref_key: str) -> Path:
    mind = data[mind_key]
    ref = data[ref_key]
    mind_t = np.asarray(mind["voltage_trace_time"], dtype=float)
    ref_t = np.asarray(ref["voltage_trace_time"], dtype=float)
    names = ["PYR[0].soma", "BAS[0].soma", "OLM[0].soma"]

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


def save_precision_plot() -> Path:
    labels = ["macro x", "macro z", "PYR soma", "BAS soma", "OLM soma", "PYR Adend3"]
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


def save_timing_plot() -> Path:
    labels = ["MIND 1T", "TVB+NEURON 1T", "MIND 4T", "TVB+NEURON 4T"]
    values = [
        comparisons["mind_vs_tvb_neuron_1thread"]["timing"]["mind"]["run_s"],
        comparisons["mind_vs_tvb_neuron_1thread"]["timing"]["reference"]["run_s"],
        comparisons["mind_vs_tvb_neuron_4thread"]["timing"]["mind"]["run_s"],
        comparisons["mind_vs_tvb_neuron_4thread"]["timing"]["reference"]["run_s"],
    ]
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


plot_paths = [
    save_macro_plot("1thread", "mind_1thread", "tvb_neuron_1thread"),
    save_macro_plot("4thread", "mind_4thread", "tvb_neuron_4thread"),
    save_voltage_plot("1thread", "mind_1thread", "tvb_neuron_1thread"),
    save_voltage_plot("4thread", "mind_4thread", "tvb_neuron_4thread"),
    save_precision_plot(),
    save_timing_plot(),
]

md_lines.extend(["", "## Plots", ""])
for path in plot_paths:
    md_lines.append(f"- `{path}`")

md_path = outdir / f"compare_{duration_ms}ms.md"
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"summary_json={json_path}")
print(f"summary_md={md_path}")
print(f"plot_dir={plotdir}")
for path in plot_paths:
    print(f"plot={path}")
PY
  conda deactivate
}

cd "${REPO_ROOT}"

echo "CA3 all-in-one workflow"
echo "duration_ms=${DURATION_MS}"
echo "connectivity_csv=${CONNECTIVITY_CSV}"
echo "outdir=${OUTDIR}"
echo "mind_env=${MIND_ENV}"
echo "tvb_env=${TVB_ENV}"

if [[ "${RUN_EXPERIMENTS}" == "1" ]]; then
  run_mind 1 "${MIND_1}"
  run_mind 4 "${MIND_4}"
  run_tvb_neuron 1 "${TVB_1}"
  run_tvb_neuron 4 "${TVB_4}"
else
  echo
  echo "== Experiments skipped because RUN_EXPERIMENTS=${RUN_EXPERIMENTS} =="
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  compare_and_plot
else
  echo
  echo "== Compare/plot skipped because RUN_COMPARE=${RUN_COMPARE} =="
fi

echo
echo "Done."
echo "Outputs: ${OUTDIR}"
