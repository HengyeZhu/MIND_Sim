from __future__ import annotations

import hashlib
import math
import os
import shlex
import subprocess
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import _native

_CODEGEN_ABI = "split_io_named_ports_v1"
_COMPILE_FLAGS = ("-std=c++20", "-O3", "-fPIC", "-shared")
_COMPILER_VERSION_CACHE: dict[str, str] = {}


def _cache_dir() -> Path:
    root = os.environ.get("MIND_SIM_CODEGEN_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "mind_sim" / "codegen"


def _compiler() -> str:
    return os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"


def _compiler_version(compiler: str) -> str:
    cached = _COMPILER_VERSION_CACHE.get(compiler)
    if cached is not None:
        return cached
    command = [*shlex.split(compiler), "--version"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "MIND_Sim code generation could not query the C++ compiler version:\n"
            f"command: {' '.join(shlex.quote(part) for part in command)}"
        ) from exc
    version = completed.stdout.splitlines()[0] if completed.stdout else ""
    _COMPILER_VERSION_CACHE[compiler] = version
    return version


def _require_integer_multiple(value: float, unit: float, what: str, unit_name: str) -> float:
    if value <= 0.0 or unit <= 0.0:
        raise ValueError(f"{what} requires positive values")
    ratio = value / unit
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{what} must be an integer multiple of {unit_name}")
    return float(steps) * unit


def _batch_window_from_min_delay(network: "Network", dt_macro: float) -> float:
    min_delay = float(network.min_positive_delay())
    if min_delay <= 0.0:
        raise ValueError("automatic batch_window requires at least one positive connectivity delay")
    steps = math.floor((min_delay / dt_macro) + 1e-9)
    if steps < 1:
        raise ValueError("minimum positive connectivity delay is smaller than dt_macro")
    return float(steps) * dt_macro


def _compile(kind: str, source: str) -> str:
    compiler = _compiler()
    compiler_version = _compiler_version(compiler)
    digest_input = "\n".join((_CODEGEN_ABI, kind, compiler, compiler_version, " ".join(_COMPILE_FLAGS), source))
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    build_dir = _cache_dir() / kind / digest
    stem = f"mind_sim_{kind}_{digest[:16]}"
    source_path = build_dir / f"{stem}.cpp"
    library_path = build_dir / f"lib{stem}.so"
    if library_path.exists():
        return str(library_path)
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    command = [
        *shlex.split(compiler),
        *_COMPILE_FLAGS,
        str(source_path),
        "-o",
        str(library_path),
    ]
    try:
        subprocess.run(command, check=True, cwd=build_dir, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "MIND_Sim code generation failed:\n"
            f"command: {' '.join(shlex.quote(part) for part in command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    return str(library_path)


def _mapping_values(name: str, values: Mapping[str, float] | None) -> dict[str, float]:
    out = dict(values or {})
    for key in out:
        if not isinstance(key, str):
            raise TypeError(f"{name} names must be strings")
    return out


def _merged_values(
    kind: str,
    names: Sequence[str],
    defaults: Mapping[str, float],
    overrides: Mapping[str, float] | None,
) -> list[float]:
    merged = dict(defaults)
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise KeyError(f"{kind} has unknown values: {', '.join(unknown)}")
    merged.update(overrides)
    missing = [name for name in names if name not in merged]
    if missing:
        raise KeyError(f"{kind} is missing values: {', '.join(missing)}")
    return [float(merged[name]) for name in names]


def _roi_index(roi) -> int:
    if isinstance(roi, ROI):
        return roi.index
    if hasattr(roi, "index"):
        return int(roi.index)
    return int(roi)


def _roi_indices(rois) -> list[int]:
    if rois == "all":
        raise ValueError("'all' must be expanded by Network")
    return [_roi_index(roi) for roi in rois]


def _gid_ranges(values) -> tuple[list[int], list[int]]:
    if hasattr(values, "gid_begin") and hasattr(values, "gid_end"):
        ranges = [(int(values.gid_begin), int(values.gid_end))]
    else:
        seq = list(values)
        if len(seq) == 2 and all(isinstance(item, int) for item in seq):
            ranges = [(int(seq[0]), int(seq[1]))]
        else:
            ranges = [(int(begin), int(end)) for begin, end in seq]
    begins = [begin for begin, _ in ranges]
    ends = [end for _, end in ranges]
    return begins, ends


def _input_port_base(value) -> int:
    if hasattr(value, "runtime_base"):
        return int(value.runtime_base)
    if hasattr(value, "runtime_index"):
        return int(value.runtime_index)
    raise TypeError("input port must be a mind_sim spike input or spike input group")


class RegionRule:
    def __init__(
        self,
        *,
        name: str,
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
        step: str,
    ):
        self.name = _native._codegen_kernel_name(name, "region rule")
        self.state = _mapping_values("state", state)
        self.params = _mapping_values("params", params)
        self.step = str(step)
        self._compiled: dict[tuple[tuple[str, ...], tuple[str, ...]], object] = {}

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    def _native_for(self, inputs: Sequence[str], exposures: Sequence[str]):
        key = (tuple(inputs), tuple(exposures))
        rule = self._compiled.get(key)
        if rule is None:
            source = _native._codegen_region_rule_source(
                list(inputs), list(exposures), self.state_names, self.param_names, self.step
            )
            path = _compile("region_rule", source)
            rule = _native._load_region_rule(
                self.name, path, len(inputs), len(exposures), len(self.state), len(self.params)
            )
            self._compiled[key] = rule
        return rule

    def state_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("RegionRule state", self.state_names, self.state, values)

    def param_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("RegionRule params", self.param_names, self.params, values)


class CouplingRule:
    def __init__(
        self,
        *,
        name: str,
        params: Mapping[str, float] | None = None,
        edge: str,
        finish: str = "",
    ):
        self.name = _native._codegen_kernel_name(name, "coupling rule")
        self.params = _mapping_values("params", params)
        self.edge = str(edge)
        self.finish = str(finish)
        self._compiled: dict[tuple[tuple[str, ...], tuple[str, ...], int], object] = {}

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    def _native_for_projection(
        self,
        inputs: Sequence[str],
        exposures: Sequence[str],
        *,
        roi_count: int,
    ):
        key = (tuple(inputs), tuple(exposures), int(roi_count))
        rule = self._compiled.get(key)
        if rule is None:
            source = _native._codegen_coupling_projection_rule_source(
                list(inputs),
                list(exposures),
                self.param_names,
                self.edge,
                self.finish,
                int(roi_count),
            )
            path = _compile("coupling_rule", source)
            rule = _native._load_coupling_rule(self.name, path, len(inputs), len(exposures), len(self.params))
            self._compiled[key] = rule
        return rule

    def param_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("CouplingRule params", self.param_names, self.params, values)


class MicroInputRule:
    def __init__(
        self,
        *,
        name: str,
        ports: Sequence[str],
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
        code: str,
    ):
        self.name = _native._codegen_kernel_name(name, "micro input rule")
        self.ports = list(ports)
        self.state = _mapping_values("state", state)
        self.params = _mapping_values("params", params)
        self.code = str(code)
        self._compiled: dict[tuple[tuple[str, ...], tuple[str, ...]], object] = {}

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    def _native_for(self, inputs: Sequence[str]):
        key = (tuple(inputs), tuple(self.ports))
        rule = self._compiled.get(key)
        if rule is None:
            source = _native._codegen_micro_input_rule_source(
                list(inputs), self.ports, self.state_names, self.param_names, self.code
            )
            path = _compile("micro_input_rule", source)
            rule = _native._load_micro_input_rule(
                self.name, path, len(inputs), len(self.state), len(self.params), len(self.ports)
            )
            self._compiled[key] = rule
        return rule

    def state_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("MicroInputRule state", self.state_names, self.state, values)

    def param_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("MicroInputRule params", self.param_names, self.params, values)


class MicroOutputRule:
    def __init__(
        self,
        *,
        name: str,
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
        spike: str,
        finish: str = "",
    ):
        self.name = _native._codegen_kernel_name(name, "micro output rule")
        self.state = _mapping_values("state", state)
        self.params = _mapping_values("params", params)
        self.spike = str(spike)
        self.finish = str(finish)
        self._compiled: dict[tuple[str, ...], object] = {}

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    def _native_for(self, exposures: Sequence[str]):
        key = tuple(exposures)
        rule = self._compiled.get(key)
        if rule is None:
            source = _native._codegen_micro_output_rule_source(
                list(exposures), self.state_names, self.param_names, self.spike, self.finish
            )
            path = _compile("micro_output_rule", source)
            rule = _native._load_micro_output_rule(self.name, path, len(exposures), len(self.state), len(self.params))
            self._compiled[key] = rule
        return rule

    def state_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("MicroOutputRule state", self.state_names, self.state, values)

    def param_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _merged_values("MicroOutputRule params", self.param_names, self.params, values)


class ROI:
    def __init__(self, network: "Network", index: int):
        self._network = network
        self.index = int(index)
        native_roi = network._native.roi(self.index)
        self.label = native_roi.label
        self._native_roi = native_roi

    def initial_output(self, values: Mapping[str, float]) -> "ROI":
        for name, value in values.items():
            self._network._native.set_initial_exposure_value(self._native_roi, name, float(value))
        return self

    def dc_input(self, values: Mapping[str, float]) -> "ROI":
        for name, value in values.items():
            self._network._native.set_dc_input_value(self._native_roi, name, float(value))
        return self

    def record(self) -> "ROI":
        self._network.record(rois=[self])
        return self

    def use(
        self,
        rule: RegionRule,
        *,
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "ROI":
        if not isinstance(rule, RegionRule):
            raise TypeError("ROI.use expects a RegionRule")
        self._network._native.use_region_rule(
            self._native_roi,
            rule._native_for(self._network.inputs, self._network.exposures),
            rule.state_values(state),
            rule.param_values(params),
        )
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.ROI index={self.index} label={self.label!r}>"


class MicroCircuit:
    def __init__(self, network: "Network", index: int, name: str):
        self._network = network
        self.index = int(index)
        self.name = str(name)

    def bind_roi(
        self,
        roi,
        *,
        gid_ranges,
        input: MicroInputRule,
        output: MicroOutputRule,
        input_state: Mapping[str, float] | None = None,
        input_params: Mapping[str, float] | None = None,
        input_ports: Mapping[str, object],
        output_state: Mapping[str, float] | None = None,
        output_params: Mapping[str, float] | None = None,
    ) -> "MicroCircuit":
        if not isinstance(input, MicroInputRule):
            raise TypeError("MicroCircuit.bind_roi input expects a MicroInputRule")
        if not isinstance(output, MicroOutputRule):
            raise TypeError("MicroCircuit.bind_roi output expects a MicroOutputRule")
        roi_obj = roi if isinstance(roi, ROI) else self._network.roi(roi)
        begins, ends = _gid_ranges(gid_ranges)
        port_values = dict(input_ports)
        missing_ports = [name for name in input.ports if name not in port_values]
        extra_ports = sorted(set(port_values) - set(input.ports))
        if missing_ports:
            raise KeyError(f"MicroCircuit.bind_roi is missing input ports: {', '.join(missing_ports)}")
        if extra_ports:
            raise KeyError(f"MicroCircuit.bind_roi has unknown input ports: {', '.join(extra_ports)}")
        input_port_bases = [_input_port_base(port_values[name]) for name in input.ports]
        self._network._native.bind_micro_roi(
            self.index,
            roi_obj._native_roi,
            begins,
            ends,
            input._native_for(self._network.inputs),
            input.state_values(input_state),
            input.param_values(input_params),
            input_port_bases,
            output._native_for(self._network.exposures),
            output.state_values(output_state),
            output.param_values(output_params),
        )
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.MicroCircuit index={self.index} name={self.name!r}>"


class Network:
    def __init__(self, *, connectivity=None, labels=None, weights=None, delays=None, inputs, exposures):
        self.inputs = list(inputs)
        self.exposures = list(exposures)
        if connectivity is not None:
            self._native = _native._Network(
                connectivity, self.inputs, self.exposures, list(range(connectivity.roi_count()))
            )
        else:
            labels = list(labels)
            self._native = _native._Network(
                labels, weights, delays, self.inputs, self.exposures, list(range(len(labels)))
            )

    def roi(self, roi) -> ROI:
        if isinstance(roi, str):
            return ROI(self, self._native.roi_index(roi))
        return ROI(self, _roi_index(roi))

    def rois(self) -> list[ROI]:
        return [ROI(self, roi.index) for roi in self._native.rois()]

    def roi_count(self) -> int:
        return self._native.roi_count()

    def min_positive_delay(self) -> float:
        return float(self._native.min_positive_delay())

    def all_rois(self) -> list[int]:
        return list(range(self.roi_count()))

    def rois_except(self, roi) -> list[int]:
        excluded = _roi_index(roi)
        return [index for index in range(self.roi_count()) if index != excluded]

    def record(self, *, rois="all", exposures="all") -> "Network":
        if exposures != "all" and list(exposures) != self.exposures:
            raise ValueError("MIND_Sim currently records all exposures for selected ROIs")
        recorded = self.all_rois() if rois == "all" else _roi_indices(rois)
        self._native.set_recorded_rois(recorded)
        return self

    def couple(
        self,
        *,
        sources,
        targets,
        rule: CouplingRule,
        params: Mapping[str, float] | None = None,
    ) -> "Network":
        if not isinstance(rule, CouplingRule):
            raise TypeError("Network.couple rule expects a CouplingRule")
        source_ids = self.all_rois() if sources == "all" else _roi_indices(sources)
        target_ids = self.all_rois() if targets == "all" else _roi_indices(targets)
        self._native.couple(
            source_ids,
            target_ids,
            rule._native_for_projection(
                self.inputs,
                self.exposures,
                roi_count=self.roi_count(),
            ),
            rule.param_values(params),
        )
        return self

    def couple_all(
        self,
        rule: CouplingRule,
        params: Mapping[str, float] | None = None,
    ) -> "Network":
        return self.couple(sources="all", targets="all", rule=rule, params=params)

    def couple_from(
        self,
        source,
        rule: CouplingRule,
        params: Mapping[str, float] | None = None,
    ) -> "Network":
        return self.couple(sources=[source], targets="all", rule=rule, params=params)

    def use_micro(self, name: str, micro) -> MicroCircuit:
        index = self._native.use_micro(str(name), micro)
        return MicroCircuit(self, index, str(name))


class Simulator:
    def __init__(
        self,
        network: Network,
        *,
        dt_micro: float,
        dt_macro: float,
        batch_window: float | None = None,
        record_micro_spikes: bool = True,
    ):
        if not isinstance(network, Network):
            raise TypeError("Simulator expects a mind_sim.Network")
        dt_micro = float(dt_micro)
        dt_macro = _require_integer_multiple(float(dt_macro), dt_micro, "dt_macro", "dt_micro")
        batch_window = (
            _batch_window_from_min_delay(network, dt_macro)
            if batch_window is None
            else _require_integer_multiple(float(batch_window), dt_macro, "batch_window", "dt_macro")
        )
        min_delay = float(network.min_positive_delay())
        if min_delay <= 0.0:
            raise ValueError("batch_window requires at least one positive connectivity delay")
        if batch_window > min_delay + 1e-9:
            raise ValueError("batch_window must not exceed the minimum positive connectivity delay")
        self._native = _native.Simulator(
            network._native,
            dt_micro,
            dt_macro,
            batch_window,
            bool(record_micro_spikes),
        )

    def run(self, t_stop: float):
        return self._native.run(float(t_stop))


class MacroSimulator:
    def __init__(self, network: Network, *, dt_macro: float):
        if not isinstance(network, Network):
            raise TypeError("MacroSimulator expects a mind_sim.Network")
        self._native = _native.MacroRuntime(network._native)
        self.dt_macro = float(dt_macro)

    def run(self, t_stop: float):
        return self._native.run(float(t_stop), self.dt_macro)


Connectivity = _native.Connectivity
