#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOLTAGE_KEY = "voltage"
VOLTAGE_TIME_KEY = "voltage_time"
DEFAULT_MACRO_ATOL = 1e-6
DEFAULT_VOLTAGE_ATOL = 1e-3


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

    macro_exposures, exposure_names = extract_macro_exposures(raw)
    if macro_exposures is not None:
        data["macro_exposures"] = macro_exposures
    if exposure_names is not None:
        data["exposure_names"] = np.asarray(exposure_names, dtype=object)

    if VOLTAGE_TIME_KEY in raw:
        data[VOLTAGE_TIME_KEY] = np.asarray(raw[VOLTAGE_TIME_KEY], dtype=float)
    if VOLTAGE_KEY in raw:
        data[VOLTAGE_KEY] = np.asarray(raw[VOLTAGE_KEY], dtype=float)

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
    if time_key in mind_data and time_key in tvbm_data:
        return time_aligned_metrics(mind_data[key], tvbm_data[key], mind_data[time_key], tvbm_data[time_key])
    return metrics(mind_data[key], tvbm_data[key])


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


def passed(item: dict, atol: float) -> bool:
    max_abs = item.get("max_abs")
    return bool(item.get("shape_match", False) and max_abs is not None and max_abs <= float(atol))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CA3 cosim outputs by recorded macro exposures and voltage.")
    parser.add_argument("mindsim", type=Path)
    parser.add_argument("tvb_multiscale", type=Path)
    parser.add_argument("--macro-atol", type=float, default=DEFAULT_MACRO_ATOL)
    parser.add_argument("--voltage-atol", type=float, default=DEFAULT_VOLTAGE_ATOL)
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

    macro_pass = passed(macro_item, args.macro_atol)
    voltage_pass = passed(voltage_item, args.voltage_atol)
    report = {
        "mindsim": mind["path"],
        "tvb_multiscale": tvbm["path"],
        "pass_mode": "recorded_macro_exposures_and_voltage",
        "macro_atol": float(args.macro_atol),
        "voltage_atol": float(args.voltage_atol),
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
        },
        "passed_macro_exposures": bool(macro_pass),
        "passed_voltage": bool(voltage_pass),
        "passed": bool(macro_pass and voltage_pass),
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
