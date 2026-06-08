#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOLTAGE_KEY = "voltage"
VOLTAGE_TIME_KEY = "voltage_time"
VOLTAGE_TRACES_KEY = "voltage_traces"
VOLTAGE_TRACE_TIME_KEY = "voltage_trace_time"
VOLTAGE_LABELS_KEY = "voltage_labels"
DEFAULT_MACRO_ATOL = 1e-6
DEFAULT_VOLTAGE_ATOL = 1e-3
DEFAULT_SPIKE_THRESHOLD_MV = 0.0


def metadata(raw: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in raw:
        return {}
    try:
        return json.loads(str(raw["metadata_json"]))
    except json.JSONDecodeError:
        return {"metadata_parse_error": str(raw["metadata_json"])}


def timing(raw: np.lib.npyio.NpzFile, meta: dict) -> dict:
    if "timing_s" in raw:
        values = np.asarray(raw["timing_s"], dtype=float)
        if values.size >= 3:
            return {"pre_run_s": float(values[0]), "run_s": float(values[1]), "total_s": float(values[2])}
    if "pre_run_s" in meta and "run_s" in meta:
        pre_run_s = float(meta["pre_run_s"])
        run_s = float(meta["run_s"])
        return {"pre_run_s": pre_run_s, "run_s": run_s, "total_s": pre_run_s + run_s}
    return {}


def extract_macro_x(raw: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "macro_x" in raw:
        return np.asarray(raw["macro_x"], dtype=float)
    if "tvb_raw" not in raw:
        return None
    data = np.asarray(raw["tvb_raw"], dtype=float)
    if data.ndim == 4 and data.shape[1] >= 1:
        return data[:, 0, :, 0]
    if data.ndim == 3 and data.shape[1] >= 1:
        return data[:, 0, :]
    raise ValueError(f"unsupported tvb_raw shape {list(data.shape)}")


def extract_macro_exposures(raw: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, list[str] | None]:
    if "macro_records" in raw:
        records = np.asarray(raw["macro_records"], dtype=float)
        names = (
            [str(name) for name in np.asarray(raw["record_names"], dtype=object).tolist()]
            if "record_names" in raw
            else None
        )
        return records, names
    if "macro_exposures" in raw:
        exposures = np.asarray(raw["macro_exposures"], dtype=float)
        names = (
            [str(name) for name in np.asarray(raw["exposure_names"], dtype=object).tolist()]
            if "exposure_names" in raw
            else None
        )
        return exposures, names
    macro_x = extract_macro_x(raw)
    if macro_x is None:
        return None, None
    if "macro_z" in raw:
        macro_z = np.asarray(raw["macro_z"], dtype=float)
        if macro_z.shape == macro_x.shape:
            return np.stack((macro_x, macro_z), axis=2), ["x", "z"]
    return macro_x[:, :, None], ["x"]


def load_output(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    meta = metadata(raw)
    data: dict[str, np.ndarray] = {}

    time_key = "time_ms" if "time_ms" in raw else "tvb_time_ms" if "tvb_time_ms" in raw else None
    if time_key is not None:
        data["time_ms"] = np.asarray(raw[time_key], dtype=float)
    if "labels" in raw:
        data["labels"] = np.asarray(raw["labels"], dtype=object)

    macro_exposures, exposure_names = extract_macro_exposures(raw)
    if macro_exposures is not None:
        data["macro_exposures"] = macro_exposures
    if exposure_names is not None:
        data["exposure_names"] = np.asarray(exposure_names, dtype=object)

    if VOLTAGE_TIME_KEY in raw:
        data[VOLTAGE_TIME_KEY] = np.asarray(raw[VOLTAGE_TIME_KEY], dtype=float)
    if VOLTAGE_KEY in raw:
        data[VOLTAGE_KEY] = np.asarray(raw[VOLTAGE_KEY], dtype=float)
    if VOLTAGE_TRACE_TIME_KEY in raw:
        data[VOLTAGE_TRACE_TIME_KEY] = np.asarray(raw[VOLTAGE_TRACE_TIME_KEY], dtype=float)
    elif VOLTAGE_TIME_KEY in data:
        data[VOLTAGE_TRACE_TIME_KEY] = np.asarray(data[VOLTAGE_TIME_KEY], dtype=float)
    if VOLTAGE_LABELS_KEY in raw:
        data[VOLTAGE_LABELS_KEY] = np.asarray(raw[VOLTAGE_LABELS_KEY], dtype=object)
    elif VOLTAGE_KEY in data:
        data[VOLTAGE_LABELS_KEY] = np.asarray(["PYR[0].soma"], dtype=object)
    if VOLTAGE_TRACES_KEY in raw:
        data[VOLTAGE_TRACES_KEY] = np.asarray(raw[VOLTAGE_TRACES_KEY], dtype=float)
    elif VOLTAGE_KEY in data:
        data[VOLTAGE_TRACES_KEY] = np.asarray(data[VOLTAGE_KEY], dtype=float)[:, None]

    return {
        "path": str(path.resolve()),
        "keys": sorted(raw.keys()),
        "metadata": meta,
        "timing": timing(raw, meta),
        "data": data,
    }


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    finite = np.isfinite(diff)
    finite_diff = diff[finite]
    return {
        "shape": list(a.shape),
        "shape_match": True,
        "finite_pair_count": int(finite.sum()),
        "total_element_count": int(finite.size),
        "max_abs": float(np.max(np.abs(finite_diff))) if finite_diff.size else None,
        "rms": float(np.sqrt(np.mean(finite_diff * finite_diff))) if finite_diff.size else None,
        "exact_equal": bool(np.array_equal(a, b)),
        "has_nonfinite_pairs": bool(finite.sum() != finite.size),
    }


def time_aligned_metrics(a: np.ndarray, b: np.ndarray, time_a: np.ndarray, time_b: np.ndarray) -> dict:
    if a.shape == b.shape and time_a.shape == time_b.shape and np.array_equal(time_a, time_b):
        item = metrics(a, b)
        item["time_match"] = True
        item["comparison"] = "exact_time_grid"
        return item

    rounded_a = np.round(np.asarray(time_a, dtype=float), 9)
    rounded_b = np.round(np.asarray(time_b, dtype=float), 9)
    common, ind_a, ind_b = np.intersect1d(rounded_a, rounded_b, return_indices=True)
    item = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "time_shape_a": list(time_a.shape),
        "time_shape_b": list(time_b.shape),
        "time_match": False,
        "comparison": "common_rounded_time_samples",
        "common_sample_count": int(common.size),
    }
    if common.size == 0:
        item["shape_match"] = False
        return item
    item.update(metrics(a[ind_a], b[ind_b]))
    return item


def compare_trace(
    mind_data: dict[str, np.ndarray],
    tvbm_data: dict[str, np.ndarray],
    key: str,
    *,
    time_key: str,
) -> dict:
    if key not in mind_data or key not in tvbm_data:
        return {"present_in_both": False}
    if np.asarray(mind_data[key]).size <= 1 or np.asarray(tvbm_data[key]).size <= 1:
        return {
            "present_in_both": True,
            "compared": False,
            "reason": "insufficient_samples",
            "shape_a": list(np.asarray(mind_data[key]).shape),
            "shape_b": list(np.asarray(tvbm_data[key]).shape),
        }
    if time_key in mind_data and time_key in tvbm_data:
        item = time_aligned_metrics(mind_data[key], tvbm_data[key], mind_data[time_key], tvbm_data[time_key])
    else:
        item = metrics(mind_data[key], tvbm_data[key])
    item["present_in_both"] = True
    item["compared"] = True
    return item


def compare_macro_exposures(mind_data: dict[str, np.ndarray], tvbm_data: dict[str, np.ndarray]) -> dict:
    if "macro_exposures" not in mind_data or "macro_exposures" not in tvbm_data:
        return {"present_in_both": False}
    mind_names = [str(name) for name in mind_data.get("exposure_names", np.asarray([], dtype=object)).tolist()]
    tvbm_names = [str(name) for name in tvbm_data.get("exposure_names", np.asarray([], dtype=object)).tolist()]
    if mind_names != tvbm_names:
        return {
            "present_in_both": True,
            "shape_match": False,
            "exposure_names_match": False,
            "mindsim_exposure_names": mind_names,
            "tvb_multiscale_exposure_names": tvbm_names,
        }
    item = compare_trace(mind_data, tvbm_data, "macro_exposures", time_key="time_ms")
    item["exposure_names_match"] = True
    item["exposure_names"] = mind_names
    return item


def inferred_voltage_dt_ms(data: dict[str, np.ndarray], meta: dict) -> float | None:
    raw_dt = meta.get("dt_micro_ms")
    try:
        dt_ms = float(raw_dt)
        if np.isfinite(dt_ms) and dt_ms > 0.0:
            return dt_ms
    except (TypeError, ValueError):
        pass

    if VOLTAGE_TIME_KEY not in data:
        return None
    time_ms = np.asarray(data[VOLTAGE_TIME_KEY], dtype=float)
    diffs = np.diff(time_ms)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return float(np.round(np.median(diffs), 12))


def voltage_threshold_spikes_from_arrays(
    time_ms: np.ndarray,
    voltage: np.ndarray,
    meta: dict,
    *,
    threshold_mv: float,
) -> dict:
    voltage = np.asarray(voltage, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    sample_count = min(voltage.size, time_ms.size)
    if sample_count <= 1:
        return {
            "present": False,
            "reason": "insufficient_voltage_samples",
            "voltage_shape": list(voltage.shape),
            "time_shape": list(time_ms.shape),
        }

    voltage = voltage[:sample_count]
    time_ms = time_ms[:sample_count]
    crossing_indices = np.flatnonzero(
        (voltage[:-1] < float(threshold_mv)) & (voltage[1:] >= float(threshold_mv))
    ) + 1
    dt_ms = inferred_voltage_dt_ms({VOLTAGE_TIME_KEY: time_ms}, meta)
    if dt_ms is None:
        spike_times_ms = time_ms[crossing_indices]
        time_source = "recorded_voltage_time"
    else:
        spike_times_ms = crossing_indices.astype(float) * float(dt_ms)
        time_source = "sample_index_times_nominal_dt"

    return {
        "present": True,
        "threshold_mv": float(threshold_mv),
        "sample_count": int(sample_count),
        "dt_ms": float(dt_ms) if dt_ms is not None else None,
        "time_source": time_source,
        "time_start_ms": float(time_ms[0]),
        "time_end_ms": float(time_ms[-1]),
        "sample_indices": crossing_indices.astype(int),
        "times_ms": np.asarray(spike_times_ms, dtype=float),
    }


def voltage_threshold_spikes(
    data: dict[str, np.ndarray],
    meta: dict,
    *,
    threshold_mv: float,
) -> dict:
    if VOLTAGE_KEY not in data or VOLTAGE_TIME_KEY not in data:
        return {"present": False, "reason": "missing_voltage_or_voltage_time"}

    return voltage_threshold_spikes_from_arrays(
        np.asarray(data[VOLTAGE_TIME_KEY], dtype=float),
        np.asarray(data[VOLTAGE_KEY], dtype=float),
        meta,
        threshold_mv=threshold_mv,
    )


def compare_spike_records(mind_spikes: dict, tvbm_spikes: dict, *, threshold_mv: float) -> dict:
    if not mind_spikes.get("present", False) or not tvbm_spikes.get("present", False):
        return {
            "present_in_both": False,
            "source": "voltage_threshold_rising_crossings",
            "threshold_mv": float(threshold_mv),
            "mindsim": {key: value for key, value in mind_spikes.items() if key not in {"sample_indices", "times_ms"}},
            "tvb_multiscale": {key: value for key, value in tvbm_spikes.items() if key not in {"sample_indices", "times_ms"}},
        }

    mind_indices = np.asarray(mind_spikes["sample_indices"], dtype=int)
    tvbm_indices = np.asarray(tvbm_spikes["sample_indices"], dtype=int)
    mind_times = np.asarray(mind_spikes["times_ms"], dtype=float)
    tvbm_times = np.asarray(tvbm_spikes["times_ms"], dtype=float)
    mind_index_set = set(mind_indices.tolist())
    tvbm_index_set = set(tvbm_indices.tolist())
    common_count = min(mind_indices.size, tvbm_indices.size)
    prefix_mismatch = np.where(mind_indices[:common_count] != tvbm_indices[:common_count])[0]
    mind_only = sorted(mind_index_set - tvbm_index_set)[:20]
    tvbm_only = sorted(tvbm_index_set - mind_index_set)[:20]
    return {
        "present_in_both": True,
        "source": "voltage_threshold_rising_crossings",
        "threshold_mv": float(threshold_mv),
        "time_source": {
            "mindsim": mind_spikes["time_source"],
            "tvb_multiscale": tvbm_spikes["time_source"],
        },
        "mindsim_dt_ms": mind_spikes["dt_ms"],
        "tvb_multiscale_dt_ms": tvbm_spikes["dt_ms"],
        "mindsim_spike_count": int(mind_indices.size),
        "tvb_multiscale_spike_count": int(tvbm_indices.size),
        "same_count": bool(mind_indices.size == tvbm_indices.size),
        "exact_order_match": bool(
            mind_indices.size == tvbm_indices.size and np.array_equal(mind_indices, tvbm_indices)
        ),
        "exact_time_match": bool(
            mind_times.size == tvbm_times.size and np.array_equal(mind_times, tvbm_times)
        ),
        "prefix_mismatch_count": int(prefix_mismatch.size),
        "first_prefix_mismatch": int(prefix_mismatch[0]) if prefix_mismatch.size else None,
        "sample_index_intersection_count": int(len(mind_index_set & tvbm_index_set)),
        "mindsim_only_count": int(len(mind_index_set - tvbm_index_set)),
        "tvb_multiscale_only_count": int(len(tvbm_index_set - mind_index_set)),
        "mindsim_only_sample_indices": [int(index) for index in mind_only],
        "tvb_multiscale_only_sample_indices": [int(index) for index in tvbm_only],
        "mindsim_spike_times_ms_sample": [float(time) for time in mind_times[:20]],
        "tvb_multiscale_spike_times_ms_sample": [float(time) for time in tvbm_times[:20]],
    }


def compare_voltage_derived_spikes(mind: dict, tvbm: dict, *, threshold_mv: float) -> dict:
    mind_spikes = voltage_threshold_spikes(mind["data"], mind["metadata"], threshold_mv=threshold_mv)
    tvbm_spikes = voltage_threshold_spikes(tvbm["data"], tvbm["metadata"], threshold_mv=threshold_mv)
    return compare_spike_records(mind_spikes, tvbm_spikes, threshold_mv=threshold_mv)


def normalized_voltage_traces(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if VOLTAGE_TRACES_KEY not in data:
        return None
    traces = np.asarray(data[VOLTAGE_TRACES_KEY], dtype=float)
    if traces.ndim == 1:
        traces = traces[:, None]
    if traces.ndim != 2:
        return None
    return traces


def compare_voltage_traces(mind_data: dict[str, np.ndarray], tvbm_data: dict[str, np.ndarray]) -> dict:
    mind_traces = normalized_voltage_traces(mind_data)
    tvbm_traces = normalized_voltage_traces(tvbm_data)
    if (
        mind_traces is None
        or tvbm_traces is None
        or VOLTAGE_TRACE_TIME_KEY not in mind_data
        or VOLTAGE_TRACE_TIME_KEY not in tvbm_data
    ):
        return {"present_in_both": False}

    mind_labels = [str(label) for label in mind_data.get(VOLTAGE_LABELS_KEY, np.asarray([], dtype=object)).tolist()]
    tvbm_labels = [str(label) for label in tvbm_data.get(VOLTAGE_LABELS_KEY, np.asarray([], dtype=object)).tolist()]
    item = time_aligned_metrics(
        mind_traces,
        tvbm_traces,
        np.asarray(mind_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
        np.asarray(tvbm_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
    )
    item["present_in_both"] = True
    item["compared"] = True
    item["labels_match"] = bool(mind_labels == tvbm_labels)
    item["mindsim_labels"] = mind_labels
    item["tvb_multiscale_labels"] = tvbm_labels

    per_trace = {}
    if mind_labels == tvbm_labels and mind_traces.shape[1] == tvbm_traces.shape[1] == len(mind_labels):
        for index, label in enumerate(mind_labels):
            trace_item = time_aligned_metrics(
                mind_traces[:, index],
                tvbm_traces[:, index],
                np.asarray(mind_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
                np.asarray(tvbm_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
            )
            trace_item["present_in_both"] = True
            trace_item["compared"] = True
            per_trace[label] = trace_item
    item["per_trace"] = per_trace
    return item


def compare_voltage_derived_spikes_by_trace(mind: dict, tvbm: dict, *, threshold_mv: float) -> dict:
    mind_data = mind["data"]
    tvbm_data = tvbm["data"]
    mind_traces = normalized_voltage_traces(mind_data)
    tvbm_traces = normalized_voltage_traces(tvbm_data)
    if (
        mind_traces is None
        or tvbm_traces is None
        or VOLTAGE_TRACE_TIME_KEY not in mind_data
        or VOLTAGE_TRACE_TIME_KEY not in tvbm_data
    ):
        return {"present_in_both": False, "source": "voltage_threshold_rising_crossings"}

    mind_labels = [str(label) for label in mind_data.get(VOLTAGE_LABELS_KEY, np.asarray([], dtype=object)).tolist()]
    tvbm_labels = [str(label) for label in tvbm_data.get(VOLTAGE_LABELS_KEY, np.asarray([], dtype=object)).tolist()]
    item = {
        "present_in_both": True,
        "source": "voltage_threshold_rising_crossings",
        "threshold_mv": float(threshold_mv),
        "labels_match": bool(mind_labels == tvbm_labels),
        "mindsim_labels": mind_labels,
        "tvb_multiscale_labels": tvbm_labels,
        "per_trace": {},
    }
    if mind_labels != tvbm_labels or mind_traces.shape[1] != tvbm_traces.shape[1] or len(mind_labels) != mind_traces.shape[1]:
        item["exact_order_match_all"] = False
        item["exact_time_match_all"] = False
        return item

    exact_order_all = True
    exact_time_all = True
    for index, label in enumerate(mind_labels):
        mind_spikes = voltage_threshold_spikes_from_arrays(
            np.asarray(mind_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
            mind_traces[:, index],
            mind["metadata"],
            threshold_mv=threshold_mv,
        )
        tvbm_spikes = voltage_threshold_spikes_from_arrays(
            np.asarray(tvbm_data[VOLTAGE_TRACE_TIME_KEY], dtype=float),
            tvbm_traces[:, index],
            tvbm["metadata"],
            threshold_mv=threshold_mv,
        )
        trace_item = compare_spike_records(mind_spikes, tvbm_spikes, threshold_mv=threshold_mv)
        item["per_trace"][label] = trace_item
        exact_order_all = exact_order_all and bool(trace_item.get("exact_order_match", False))
        exact_time_all = exact_time_all and bool(trace_item.get("exact_time_match", False))
    item["exact_order_match_all"] = bool(exact_order_all)
    item["exact_time_match_all"] = bool(exact_time_all)
    return item


def passed(item: dict, atol: float) -> bool:
    max_abs = item.get("max_abs")
    return bool(item.get("shape_match", False) and max_abs is not None and max_abs <= float(atol))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CA3 cosim outputs by recorded macro exposures and voltage.")
    parser.add_argument("mindsim", type=Path)
    parser.add_argument("tvb_multiscale", type=Path)
    parser.add_argument("--macro-atol", type=float, default=DEFAULT_MACRO_ATOL)
    parser.add_argument("--voltage-atol", type=float, default=DEFAULT_VOLTAGE_ATOL)
    parser.add_argument("--spike-threshold-mv", type=float, default=DEFAULT_SPIKE_THRESHOLD_MV)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=Path("outputs/mindsim_vs_tvb_multiscale.json"))
    parser.add_argument("--no-fail", action="store_true", help="Write the report even when differences exceed tolerances.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mind = load_output(args.mindsim)
    tvbm = load_output(args.tvb_multiscale)
    mind_data = mind["data"]
    tvbm_data = tvbm["data"]

    macro_item = compare_macro_exposures(mind_data, tvbm_data)

    voltage_item = compare_trace(mind_data, tvbm_data, VOLTAGE_KEY, time_key=VOLTAGE_TIME_KEY)
    voltage_traces_item = compare_voltage_traces(mind_data, tvbm_data)
    spike_item = compare_voltage_derived_spikes(mind, tvbm, threshold_mv=float(args.spike_threshold_mv))
    spike_traces_item = compare_voltage_derived_spikes_by_trace(
        mind,
        tvbm,
        threshold_mv=float(args.spike_threshold_mv),
    )

    macro_pass = passed(macro_item, args.macro_atol)
    voltage_pass = True
    if voltage_item.get("present_in_both", False) and voltage_item.get("compared", True):
        voltage_pass = passed(voltage_item, args.voltage_atol)
    voltage_traces_pass = True
    if voltage_traces_item.get("present_in_both", False) and voltage_traces_item.get("compared", True):
        voltage_traces_pass = passed(voltage_traces_item, args.voltage_atol)
    spike_pass = True
    if spike_item.get("present_in_both", False):
        spike_pass = bool(spike_item.get("exact_order_match", False))
    spike_traces_pass = True
    if spike_traces_item.get("present_in_both", False):
        spike_traces_pass = bool(spike_traces_item.get("exact_order_match_all", False))
    report = {
        "mindsim": mind["path"],
        "tvb_multiscale": tvbm["path"],
        "pass_mode": "recorded_macro_records_voltage_derived_spikes_and_optional_voltage",
        "macro_atol": float(args.macro_atol),
        "voltage_atol": float(args.voltage_atol),
        "spike_threshold_mv": float(args.spike_threshold_mv),
        "schema": {
            "mindsim_keys": mind["keys"],
            "tvb_multiscale_keys": tvbm["keys"],
        },
        "timing": {
            "mindsim": mind["timing"],
            "tvb_multiscale": tvbm["timing"],
        },
        "numeric": {
            "macro_exposures": macro_item,
            "voltage": voltage_item,
            "voltage_traces": voltage_traces_item,
            "spikes": spike_item,
            "spikes_by_voltage_trace": spike_traces_item,
        },
        "passed_macro_exposures": bool(macro_pass),
        "passed_voltage": bool(voltage_pass),
        "passed_voltage_traces": bool(voltage_traces_pass),
        "passed_spikes": bool(spike_pass),
        "passed_spikes_by_voltage_trace": bool(spike_traces_pass),
        "passed": bool(macro_pass and voltage_pass and voltage_traces_pass and spike_pass and spike_traces_pass),
    }

    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
