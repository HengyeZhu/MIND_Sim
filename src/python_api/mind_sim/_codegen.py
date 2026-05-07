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

_CODEGEN_ABI = "bare_mind_mod_double_v2_random"
_COMPILE_FLAGS = ("-std=c++20", "-O3", "-fPIC", "-shared")
_COMPILER_VERSION_CACHE: dict[str, str] = {}
_MOD_RULE_CACHE: dict[tuple[str, int, int], "ModRule"] = {}


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


def _compile_artifact(kind: str, source: str) -> tuple[str, str]:
    compiler = _compiler()
    compiler_version = _compiler_version(compiler)
    digest_input = "\n".join((_CODEGEN_ABI, kind, compiler, compiler_version, " ".join(_COMPILE_FLAGS), source))
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    build_dir = _cache_dir() / kind / digest
    stem = f"mind_sim_{kind}_{digest[:16]}"
    source_path = build_dir / f"{stem}.cpp"
    library_path = build_dir / f"lib{stem}.so"
    if library_path.exists():
        return str(library_path), str(source_path)
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
    return str(library_path), str(source_path)


def _compile(kind: str, source: str) -> str:
    return _compile_artifact(kind, source)[0]


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


def _strict_override_values(
    kind: str,
    names: Sequence[str],
    defaults: Mapping[str, float],
    overrides: Mapping[str, float] | None,
) -> list[float]:
    if overrides is None:
        return [float(defaults[name]) for name in names]
    values = dict(overrides)
    expected = set(names)
    got = set(values)
    missing = [name for name in names if name not in got]
    extra = sorted(got - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise KeyError(f"{kind} names must exactly match the mod declaration ({'; '.join(details)})")
    return [float(values[name]) for name in names]


def _random_provider_source(state_names: Sequence[str], uniform_body: str) -> str:
    lines = [
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "using std::abs;",
        "using std::ceil;",
        "using std::clamp;",
        "using std::exp;",
        "using std::fabs;",
        "using std::floor;",
        "using std::fmax;",
        "using std::fmin;",
        "using std::isfinite;",
        "using std::log;",
        "using std::max;",
        "using std::min;",
        "using std::pow;",
        "using std::round;",
        "using std::size_t;",
        "",
        'extern "C" double mind_random_uniform(double* state, int state_count, int index, int draw) {',
        "    (void)state_count;",
        "    (void)index;",
        "    (void)draw;",
    ]
    for offset, name in enumerate(state_names):
        if not isinstance(name, str):
            raise TypeError("random provider state names must be strings")
        kernel_name = _native._codegen_kernel_name(name, "random provider state")
        lines.append(f"    double& {kernel_name} = state[{offset}];")
    lines.append(str(uniform_body))
    lines.append("}")
    return "\n".join(lines)


class _CompiledRandomProvider:
    def __init__(self, *, state: Mapping[str, float], uniform: str):
        self.state = _mapping_values("random provider state", state)
        self.uniform = str(uniform)
        if not self.uniform.strip():
            raise ValueError("random provider uniform body must be non-empty")
        source = _random_provider_source(list(self.state), self.uniform)
        self.library_path = _compile("mind_random_provider", source)
        self._native_rule = _native._load_random_stream_rule(self.library_path, len(self.state))

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    def state_values(self) -> list[float]:
        return [float(self.state[name]) for name in self.state_names]


def _compile_random_provider(provider: object) -> _CompiledRandomProvider:
    if hasattr(provider, "_mind_random_provider"):
        provider = provider._mind_random_provider()
    if not isinstance(provider, Mapping):
        raise TypeError("random provider must be a mapping or expose _mind_random_provider()")
    if "state" not in provider or "uniform" not in provider:
        raise KeyError("random provider requires 'state' and 'uniform'")
    return _CompiledRandomProvider(state=provider["state"], uniform=provider["uniform"])


def _strict_random_values(
    kind: str,
    names: Sequence[str],
    providers: Mapping[str, object] | None,
) -> tuple[list[object], list[list[float]]]:
    values = dict(providers or {})
    expected = set(names)
    got = set(values)
    missing = [name for name in names if name not in got]
    extra = sorted(got - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise KeyError(f"{kind} names must exactly match the mod declaration ({'; '.join(details)})")

    rules: list[object] = []
    states: list[list[float]] = []
    for name in names:
        provider = _compile_random_provider(values[name])
        rules.append(provider._native_rule)
        states.append(provider.state_values())
    return rules, states


def _roi_index(roi) -> int:
    if isinstance(roi, ROI):
        return roi.index
    if hasattr(roi, "index"):
        return int(roi.index)
    return int(roi)


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


def _add_name(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def _require_names(context: str, names: Sequence[str], available: set[str]) -> None:
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"{context} missing names: {', '.join(missing)}")


def _offsets(schema_names: Sequence[str], names: Sequence[str], width: int) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(schema_names)}
    return [index_by_name[name] * int(width) for name in names]


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
        fields = _native._inspect_region_rule_fields(self.state_names, self.param_names, self.step)
        self.read = list(fields["inputs"])
        self.write = list(fields["exposures"])
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


class ModRule:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        source = self.path.read_text(encoding="utf-8")
        cpp_source = _native._translate_mind_mod_to_cpp(source, str(self.path))
        self.library_path, self.cpp_path = _compile_artifact("mind_mod_rule", cpp_source)
        spec = _native._inspect_mind_mod_library(self.library_path)
        self.kind = str(spec["kind"])
        self.mod_name = str(spec["name"])
        self.name = _native._codegen_kernel_name(self.mod_name, f"{self.kind} mod rule")
        self.read = list(spec["read"])
        self.write = list(spec["write"])
        self.emit = list(spec["emit"])
        self.random = list(spec["random"])
        self.params = dict(zip(list(spec["param_names"]), list(spec["param_defaults"])))
        self.state = dict(zip(list(spec["state_names"]), list(spec["state_defaults"])))
        self._native_rule: object | None = None

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    @property
    def ports(self) -> list[str]:
        return list(self.emit)

    @property
    def random_names(self) -> list[str]:
        return list(self.random)

    def _require_kind(self, expected: str) -> None:
        if self.kind != expected:
            raise TypeError(f"{self.path} is a {self.kind!r} mod rule, expected {expected!r}")

    def _native_coupling(self):
        self._require_kind("coupling")
        if self._native_rule is None:
            self._native_rule = _native._load_compiled_coupling_rule(self.library_path)
        return self._native_rule

    def _native_micro_input(self):
        self._require_kind("micro_input")
        if self._native_rule is None:
            self._native_rule = _native._load_compiled_micro_input_rule(self.library_path)
        return self._native_rule

    def _native_micro_output(self):
        self._require_kind("micro_output")
        if self._native_rule is None:
            self._native_rule = _native._load_compiled_micro_output_rule(self.library_path)
        return self._native_rule

    def state_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _strict_override_values(f"{self.mod_name} state", self.state_names, self.state, values)

    def param_values(self, values: Mapping[str, float] | None = None) -> list[float]:
        return _strict_override_values(f"{self.mod_name} params", self.param_names, self.params, values)

    def random_values(self, values: Mapping[str, object] | None = None) -> tuple[list[object], list[list[float]]]:
        return _strict_random_values(f"{self.mod_name} random", self.random_names, values)

    def __repr__(self) -> str:
        return f"<mind_sim.ModRule kind={self.kind!r} name={self.mod_name!r} path={str(self.path)!r}>"


def _load_mod_rule(path: str | Path) -> ModRule:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MOD_RULE_CACHE.get(key)
    if cached is None:
        cached = ModRule(resolved)
        _MOD_RULE_CACHE[key] = cached
    return cached


class ROI:
    def __init__(self, network: "Network", index: int, label: str):
        self._network = network
        self.index = int(index)
        self.label = str(label)
        self.name = self.label

    def initial_output(self, values: Mapping[str, float]) -> "ROI":
        self._network._set_initial_output(self.index, values)
        return self

    def dc_input(self, values: Mapping[str, float]) -> "ROI":
        self._network._set_dc_input(self.index, values)
        return self

    def record(self) -> "ROI":
        self._network.record(rois=[self])
        return self

    def connect(
        self,
        source,
        rule: str,
        *,
        params: Mapping[str, float] | None = None,
        state: Mapping[str, float] | None = None,
        random: Mapping[str, object] | None = None,
    ) -> "ROI":
        mod = self._network._mod_rule(rule)
        source_roi = self._network.roi(source)
        if mod.kind == "coupling":
            if random is not None:
                raise TypeError("coupling mod rules do not accept random providers")
            self._network._add_coupling(source_roi.index, self.index, mod, mod.param_values(params))
        elif mod.kind == "micro_input":
            self._network._set_micro_input_rule(
                self.index,
                source_roi.index,
                mod,
                mod.state_values(state),
                mod.param_values(params),
                mod.random_values(random),
            )
        elif mod.kind == "micro_output":
            if random is not None:
                raise TypeError("micro output mod rules do not accept random providers")
            self._network._set_micro_output_rule(
                self.index,
                source_roi.index,
                mod,
                mod.state_values(state),
                mod.param_values(params),
            )
        else:
            raise TypeError(f"unsupported mod rule kind: {mod.kind}")
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
        self._network._use_region_rule(
            self.index,
            rule,
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
        ports: Mapping[str, object] | None = None,
    ) -> "MicroCircuit":
        roi_obj = roi if isinstance(roi, ROI) else self._network.roi(roi)
        begins, ends = _gid_ranges(gid_ranges)
        port_values = dict(ports or {})
        for name in port_values:
            if not isinstance(name, str):
                raise TypeError("micro ROI port names must be strings")
        self._network._bind_micro_roi(self.index, roi_obj.index, begins, ends, port_values)
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.MicroCircuit index={self.index} name={self.name!r}>"


class Network:
    def __init__(self, *, connectivity=None, labels=None, weights=None, delays=None):
        if connectivity is not None:
            self._connectivity = connectivity
        else:
            if labels is None or weights is None or delays is None:
                raise TypeError("Network requires connectivity or labels/weights/delays")
            self._connectivity = _native.Connectivity(list(labels), weights, delays)
        native_rois = list(self._connectivity.rois())
        self._labels = [roi.label for roi in native_rois]
        self._recorded_rois = list(range(self._connectivity.roi_count()))
        self._micro_input_ports: dict[int, dict[str, object]] = {}
        self._initial_outputs: dict[int, dict[str, float]] = {}
        self._dc_inputs: dict[int, dict[str, float]] = {}
        self._region_owners: dict[int, tuple[RegionRule, list[float], list[float]]] = {}
        self._couplings: list[tuple[int, int, ModRule, list[float]]] = []
        self._micro_circuits: list[object] = []
        self._micro_bindings: dict[int, tuple[int, list[int], list[int]]] = {}
        self._micro_input_rules: dict[
            int,
            tuple[int, ModRule, list[float], list[float], tuple[list[object], list[list[float]]]],
        ] = {}
        self._micro_output_rules: dict[int, tuple[int, ModRule, list[float], list[float]]] = {}
        self._rois = [ROI(self, roi.index, roi.label) for roi in native_rois]
        self._roi_by_label = {roi.label: roi for roi in self._rois}
        self._mod_rules: dict[str, ModRule] = {}
        self._loaded_mod_metadata_paths: list[str] = []

    def roi(self, roi) -> ROI:
        if isinstance(roi, ROI):
            if roi._network is not self:
                raise ValueError("ROI belongs to a different Network")
            return roi
        if isinstance(roi, str):
            try:
                return self._roi_by_label[roi]
            except KeyError as exc:
                raise KeyError(f"unknown ROI: {roi}") from exc
        index = _roi_index(roi)
        if index < 0:
            index += len(self._rois)
        if index < 0 or index >= len(self._rois):
            raise IndexError("ROI index out of range")
        return self._rois[index]

    def rois(self) -> list[ROI]:
        return list(self._rois)

    def roi_count(self) -> int:
        return self._connectivity.roi_count()

    def min_positive_delay(self) -> float:
        return float(self._connectivity.min_positive_delay())

    def record(self, *, rois="all") -> "Network":
        recorded = (
            list(range(self.roi_count()))
            if rois == "all"
            else [self.roi(roi).index for roi in (rois if isinstance(rois, (list, tuple, set)) else [rois])]
        )
        self._recorded_rois = recorded
        return self

    def load_mod_metadata(self, path: str | Path) -> "Network":
        root = Path(path)
        if root.is_dir():
            mod_paths = sorted(root.glob("*.mod"))
            origin = str(root)
        else:
            mod_paths = [root]
            origin = str(root.parent)
        if not mod_paths:
            raise FileNotFoundError(f"no .mod files found in {origin}")
        for mod_path in mod_paths:
            rule = _load_mod_rule(mod_path)
            existing = self._mod_rules.get(rule.mod_name)
            if existing is not None and existing.path.resolve() != rule.path.resolve():
                raise RuntimeError(
                    f"MindMod rule {rule.mod_name!r} is declared by both "
                    f"{existing.path} and {rule.path}"
                )
            self._mod_rules[rule.mod_name] = rule
        self._loaded_mod_metadata_paths.append(str(Path(path).resolve()))
        return self

    def get_loaded_mod_metadata_paths(self) -> list[str]:
        return list(self._loaded_mod_metadata_paths)

    def use_micro(self, micro) -> MicroCircuit:
        index = len(self._micro_circuits)
        self._micro_circuits.append(micro)
        return MicroCircuit(self, index, f"micro_{index}")

    def _mod_rule(self, rule: str) -> ModRule:
        if not isinstance(rule, str):
            raise TypeError("ROI.connect expects a MindMod rule name string")
        try:
            return self._mod_rules[rule]
        except KeyError as exc:
            raise KeyError(
                f"unknown MindMod rule {rule!r}; load its .mod directory with network.load_mod_metadata(...)"
            ) from exc

    def _set_initial_output(self, roi: int, values: Mapping[str, float]) -> None:
        target = self._initial_outputs.setdefault(int(roi), {})
        for name, value in _mapping_values("initial output", values).items():
            target[name] = float(value)

    def _set_dc_input(self, roi: int, values: Mapping[str, float]) -> None:
        target = self._dc_inputs.setdefault(int(roi), {})
        for name, value in _mapping_values("dc input", values).items():
            target[name] = float(value)

    def _use_region_rule(self, roi: int, rule: RegionRule, state: list[float], params: list[float]) -> None:
        self._region_owners[int(roi)] = (rule, state, params)

    def _add_coupling(self, source: int, target: int, mod: ModRule, params: list[float]) -> None:
        self._couplings.append((int(source), int(target), mod, params))

    def _bind_micro_roi(
        self,
        circuit_index: int,
        roi: int,
        begins: list[int],
        ends: list[int],
        ports: Mapping[str, object],
    ) -> None:
        self._micro_bindings[int(roi)] = (int(circuit_index), list(begins), list(ends))
        self._micro_input_ports[int(roi)] = dict(ports)

    def _set_micro_input_rule(
        self,
        roi: int,
        source: int,
        mod: ModRule,
        state: list[float],
        params: list[float],
        random_streams: tuple[list[object], list[list[float]]],
    ) -> None:
        self._micro_input_rules[int(roi)] = (int(source), mod, state, params, random_streams)

    def _set_micro_output_rule(
        self,
        roi: int,
        source: int,
        mod: ModRule,
        state: list[float],
        params: list[float],
    ) -> None:
        self._micro_output_rules[int(roi)] = (int(source), mod, state, params)

    def _schema(self) -> tuple[list[str], list[str], dict[int, set[str]], dict[int, set[str]]]:
        inputs: list[str] = []
        exposures: list[str] = []
        accepted_by_roi: dict[int, set[str]] = {roi.index: set() for roi in self._rois}
        exposed_by_roi: dict[int, set[str]] = {roi.index: set() for roi in self._rois}

        for roi, (rule, _, _) in self._region_owners.items():
            for name in rule.read:
                _add_name(inputs, name)
                accepted_by_roi[roi].add(name)
            for name in rule.write:
                exposed_by_roi[roi].add(name)

        for roi, (_, mod, _, _, _) in self._micro_input_rules.items():
            for name in mod.read:
                _add_name(inputs, name)
                accepted_by_roi[roi].add(name)

        for roi, (_, mod, _, _) in self._micro_output_rules.items():
            for name in mod.write:
                _add_name(exposures, name)
                exposed_by_roi[roi].add(name)

        for source, target, mod, _ in self._couplings:
            for name in mod.read:
                _add_name(exposures, name)
            for name in mod.write:
                _add_name(inputs, name)
            _require_names(f"{self.roi(source).label} -> {self.roi(target).label} READ", mod.read, exposed_by_roi[source])
            _require_names(f"{self.roi(source).label} -> {self.roi(target).label} WRITE", mod.write, accepted_by_roi[target])

        for roi, values in self._initial_outputs.items():
            _require_names(f"{self.roi(roi).label} initial_output", values, exposed_by_roi[roi])
            for name in values:
                _add_name(exposures, name)

        for roi, values in self._dc_inputs.items():
            _require_names(f"{self.roi(roi).label} dc_input", values, accepted_by_roi[roi])
            for name in values:
                _add_name(inputs, name)

        return inputs, exposures, exposed_by_roi, accepted_by_roi

    def _build_native(self):
        inputs, exposures, _, _ = self._schema()
        native = _native._Network(self._connectivity, inputs, exposures, self._recorded_rois)

        for roi, values in self._initial_outputs.items():
            native_roi = native.roi(roi)
            for name, value in values.items():
                native.set_initial_exposure_value(native_roi, name, value)

        for roi, values in self._dc_inputs.items():
            native_roi = native.roi(roi)
            for name, value in values.items():
                native.set_dc_input_value(native_roi, name, value)

        for roi, (rule, state, params) in self._region_owners.items():
            native.use_region_rule(
                native.roi(roi),
                rule._native_for(inputs, exposures),
                state,
                params,
            )

        for index, micro in enumerate(self._micro_circuits):
            native.use_micro(f"micro_{index}", micro)

        for roi, (circuit_index, begins, ends) in self._micro_bindings.items():
            native.bind_micro_roi(circuit_index, native.roi(roi), begins, ends)

        for roi, (_, mod, state, params, random_streams) in self._micro_input_rules.items():
            port_values = self._micro_input_ports.get(roi)
            if port_values is None:
                raise KeyError(f"micro ROI {self.roi(roi).label!r} has no exposed input ports")
            missing_ports = [name for name in mod.ports if name not in port_values]
            extra_ports = sorted(set(port_values) - set(mod.ports))
            if missing_ports:
                raise KeyError(f"micro input mod is missing ports: {', '.join(missing_ports)}")
            if extra_ports:
                raise KeyError(f"micro input mod has unconsumed exposed ports: {', '.join(extra_ports)}")
            input_port_bases = [_input_port_base(port_values[name]) for name in mod.ports]
            random_rules, random_states = random_streams
            native.configure_micro_input_rule(
                native.roi(roi),
                mod._native_micro_input(),
                state,
                params,
                random_rules,
                random_states,
                input_port_bases,
                _offsets(inputs, mod.read, self.roi_count()),
            )

        for roi, (_, mod, state, params) in self._micro_output_rules.items():
            native.configure_micro_output_rule(
                native.roi(roi),
                mod._native_micro_output(),
                state,
                params,
                _offsets(exposures, mod.write, self.roi_count()),
            )

        for source, target, mod, params in self._couplings:
            native.couple(
                native.roi(source),
                native.roi(target),
                mod._native_coupling(),
                params,
                _offsets(exposures, mod.read, self.roi_count()),
                _offsets(inputs, mod.write, self.roi_count()),
            )
        return native


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
            network._build_native(),
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
        self._native = _native.MacroRuntime(network._build_native())
        self.dt_macro = float(dt_macro)

    def run(self, t_stop: float):
        return self._native.run(float(t_stop), self.dt_macro)


Connectivity = _native.Connectivity
