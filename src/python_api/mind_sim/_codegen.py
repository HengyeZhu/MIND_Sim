from __future__ import annotations

import hashlib
import math
import os
import re
import shlex
import subprocess
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import _native

_CODEGEN_ABI = "mind_mod_double_v3_region_field"
_COMPILE_FLAGS = ("-std=c++20", "-O3", "-fPIC", "-shared")
_COMPILER_VERSION_CACHE: dict[str, str] = {}
_MOD_RULE_CACHE: dict[tuple[str, int, int], "ModRule"] = {}
_OWNER_ARTIFACT_CACHE: dict[tuple[object, ...], tuple[str, str]] = {}


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


def _name_list(kind: str, values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    names = [values] if isinstance(values, str) else list(values)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{kind} names must be strings")
    return names


def _schema_names(kind: str, values: Mapping[str, object] | Sequence[str] | str | None) -> list[str]:
    return _name_list(kind, list(values) if isinstance(values, Mapping) else values)


def _require_identifier(name: str, kind: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{kind} must be a valid Python/C++ identifier")
    return name


def _identifier_list(kind: str, values: Sequence[str]) -> list[str]:
    return [_require_identifier(name, kind) for name in values]


def _cpp_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

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


def _offset_map(schema_names: Sequence[str], width: int) -> dict[str, int]:
    stride = int(width)
    return {name: index * stride for index, name in enumerate(schema_names)}


def _offsets(offset_by_name: Mapping[str, int], names: Sequence[str]) -> list[int]:
    return [offset_by_name[name] for name in names]


def _local_connectivity(values, node_count: int):
    node_count = int(node_count)
    if node_count <= 0:
        raise ValueError("neural field requires a positive node_count")
    if values is None:
        return _native.LocalConnectivity.from_arrays(node_count, [0] * (node_count + 1), [], [])
    if isinstance(values, _native.LocalConnectivity):
        if int(values.node_count) != node_count:
            raise ValueError("LocalConnectivity node_count must match node_map")
        return values
    if hasattr(values, "tocsr") or all(hasattr(values, name) for name in ("indptr", "indices", "data")):
        local = _native.LocalConnectivity.from_csr(values)
        if int(local.node_count) != node_count:
            raise ValueError("LocalConnectivity node_count must match node_map")
        return local
    raise TypeError("neural field local must be a mind_sim.LocalConnectivity")


def _node_values(value, node_count: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} values must be numeric")
    try:
        seq = list(value)
    except TypeError:
        return [float(value)] * node_count
    if len(seq) != node_count:
        raise ValueError(f"{name} node value count must match node_count")
    return [float(item) for item in seq]


def _field_state_values(
    names: Sequence[str],
    defaults: Mapping[str, float],
    overrides: Mapping[str, object] | None,
    node_count: int,
) -> list[float]:
    merged = dict(defaults)
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise KeyError(f"neural field state has unknown values: {', '.join(unknown)}")
    merged.update(overrides)
    missing = [name for name in names if name not in merged]
    if missing:
        raise KeyError(f"neural field state is missing values: {', '.join(missing)}")
    out: list[float] = []
    for name in names:
        out.extend(_node_values(merged[name], node_count, f"state {name}"))
    return out


_DERIVATIVE_RE = re.compile(r"^d([A-Za-z_][A-Za-z0-9_]*)/dt\s*=\s*(.+)$")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?!=)(.+)$")
_LOCAL_RE = re.compile(r"\blocal\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


def _owner_exposures(values: Sequence[str] | str | None) -> list[str]:
    names = _identifier_list("exposure name", _name_list("exposure", values))
    if not names:
        raise TypeError("owner exposures must be a non-empty string or sequence of strings")
    return names


def _owner_state_values(values: Mapping[str, object] | None) -> dict[str, object]:
    return {
        _require_identifier(str(name), "state name"): value
        for name, value in dict(values or {}).items()
    }


def _owner_param_values(values: Mapping[str, float] | None) -> dict[str, float]:
    params = _mapping_values("params", values)
    return {_require_identifier(name, "param name"): float(value) for name, value in params.items()}


def _reject_overlap(left_name: str, left: Sequence[str], right_name: str, right: Sequence[str]) -> None:
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise ValueError(f"{left_name} and {right_name} names must be distinct: {', '.join(overlap)}")


def _replace_local_calls(line: str) -> tuple[str, list[str]]:
    local_states: list[str] = []

    def replace(match: re.Match[str]) -> str:
        state = _require_identifier(match.group(1), "local state name")
        if state not in local_states:
            local_states.append(state)
        return f"local_{state}"

    return _LOCAL_RE.sub(replace, line), local_states


def _owner_statement_code(
    equations: str,
    *,
    known_names: set[str],
    allow_local: bool,
) -> tuple[str, list[str]]:
    if not isinstance(equations, str):
        raise TypeError("owner equations must be a string")
    lines: list[str] = []
    locals_: set[str] = set()
    local_states: list[str] = []
    for raw in str(equations).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if allow_local:
            line, found_local_states = _replace_local_calls(line)
            for state in found_local_states:
                if state not in local_states:
                    local_states.append(state)
        if line.endswith("{") or line.endswith("}") or line.startswith("//"):
            lines.append(line)
            continue
        line = line[:-1].strip() if line.endswith(";") else line
        derivative = _DERIVATIVE_RE.match(line)
        if derivative:
            state = _require_identifier(derivative.group(1), "state name")
            if state not in known_names:
                raise KeyError(f"derivative references undeclared state: {state}")
            lines.append(f"{state} += dt * ({derivative.group(2)});")
            continue
        assignment = _ASSIGN_RE.match(line)
        if assignment:
            lhs = _require_identifier(assignment.group(1), "assignment name")
            rhs = assignment.group(2)
            if lhs not in known_names and lhs not in locals_:
                locals_.add(lhs)
                lines.append(f"double {lhs} = {rhs};")
            else:
                lines.append(f"{lhs} = {rhs};")
            continue
        lines.append(line + ";")
    return "\n".join(lines), local_states


def _owner_prelude() -> str:
    return r"""
#include <algorithm>
#include <cmath>
#include <cstddef>

using std::abs;
using std::ceil;
using std::clamp;
using std::exp;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::isfinite;
using std::log;
using std::max;
using std::min;
using std::pow;
using std::round;
using std::size_t;

struct mind_rule_descriptor {
    int abi_version;
    int kind;
    const char* name;
    int read_count;
    const char* const* read_names;
    int write_count;
    const char* const* write_names;
    int emit_count;
    const char* const* emit_names;
    int param_count;
    const char* const* param_names;
    const double* param_defaults;
    int state_count;
    const char* const* state_names;
    const double* state_defaults;
    int random_count;
    const char* const* random_names;
    int local_state_count;
    const char* const* local_state_names;
};

struct mind_region_context {
    int owner_count;
    const int* roi_indices;
    int roi_count;
    int input_count;
    const double* input_soa;
    int exposure_count;
    double* exposure_soa;
    int state_count;
    double* state_soa;
    int param_count;
    const double* params_soa;
    const int* read_input_offsets;
    const int* write_exposure_offsets;
    double t;
    double dt;
};

struct mind_neural_field_context {
    int node_count;
    const int* node_to_roi;
    int roi_count;
    int input_count;
    const double* input_soa;
    int state_count;
    const double* previous_state_soa;
    double* state_soa;
    int param_count;
    const double* params;
    const int* local_indptr;
    const int* local_indices;
    const double* local_weights;
    const int* read_input_offsets;
    double t;
    double dt;
};
"""


def _cpp_name_array(symbol: str, names: Sequence[str]) -> str:
    if not names:
        return ""
    values = ", ".join(_cpp_string(name) for name in names)
    return f"static const char* {symbol}[] = {{{values}}};\n"


def _cpp_default_array(symbol: str, count: int) -> str:
    if count == 0:
        return ""
    return f"static const double {symbol}[] = {{{', '.join(['0.0'] * count)}}};\n"


def _array_pointer(symbol: str, names: Sequence[str]) -> str:
    return symbol if names else "nullptr"


def _default_pointer(symbol: str, count: int) -> str:
    return symbol if count else "nullptr"


def _owner_descriptor_source(
    *,
    kind_id: int,
    name: str,
    inputs: Sequence[str],
    exposures: Sequence[str],
    states: Sequence[str],
    params: Sequence[str],
    local_states: Sequence[str],
) -> str:
    return (
        _cpp_name_array("mind_read_names", inputs)
        + _cpp_name_array("mind_write_names", exposures)
        + _cpp_name_array("mind_param_names", params)
        + _cpp_name_array("mind_state_names", states)
        + _cpp_name_array("mind_local_state_names", local_states)
        + _cpp_default_array("mind_param_defaults", len(params))
        + _cpp_default_array("mind_state_defaults", len(states))
        + "static const mind_rule_descriptor mind_descriptor = {\n"
        + f"    3, {kind_id}, {_cpp_string(name)},\n"
        + f"    {len(inputs)}, {_array_pointer('mind_read_names', inputs)},\n"
        + f"    {len(exposures)}, {_array_pointer('mind_write_names', exposures)},\n"
        + "    0, nullptr,\n"
        + f"    {len(params)}, {_array_pointer('mind_param_names', params)}, "
        + f"{_default_pointer('mind_param_defaults', len(params))},\n"
        + f"    {len(states)}, {_array_pointer('mind_state_names', states)}, "
        + f"{_default_pointer('mind_state_defaults', len(states))},\n"
        + "    0, nullptr,\n"
        + f"    {len(local_states)}, {_array_pointer('mind_local_state_names', local_states)},\n"
        + "};\n"
        + 'extern "C" const mind_rule_descriptor* mind_rule_descriptor() { return &mind_descriptor; }\n'
    )


def _owner_region_source(
    *,
    name: str,
    equations: str,
    inputs: Sequence[str],
    exposures: Sequence[str],
    states: Sequence[str],
    params: Sequence[str],
) -> str:
    known = set(inputs) | set(states) | set(params) | {"t", "dt", "roi"}
    code, _ = _owner_statement_code(equations, known_names=known, allow_local=False)
    lines = [_owner_prelude()]
    lines.append(
        _owner_descriptor_source(
            kind_id=3,
            name=name,
            inputs=inputs,
            exposures=exposures,
            states=states,
            params=params,
            local_states=[],
        )
    )
    lines.append(
        r"""
extern "C" void mind_region_rule_apply(const mind_region_context* ctx) {
    const int owner_count = ctx->owner_count;
    const int* roi_indices = ctx->roi_indices;
    const double* input_soa = ctx->input_soa;
    double* exposure_soa = ctx->exposure_soa;
    double* state_soa = ctx->state_soa;
    const double* params_soa = ctx->params_soa;
    const int* read_offsets = ctx->read_input_offsets;
    const int* write_offsets = ctx->write_exposure_offsets;
    const double t = ctx->t;
    const double dt = ctx->dt;
    (void)t;
    for (int unit = 0; unit < owner_count; ++unit) {
        const int roi = roi_indices[unit];
"""
    )
    for index, field in enumerate(inputs):
        lines.append(f"        const double {field} = input_soa[read_offsets[{index}] + roi];\n")
    for index, field in enumerate(states):
        lines.append(f"        double& {field} = state_soa[({index} * owner_count) + unit];\n")
    for index, field in enumerate(params):
        lines.append(f"        const double {field} = params_soa[({index} * owner_count) + unit];\n")
    for line in code.splitlines():
        lines.append(f"        {line}\n")
    for index, exposure in enumerate(exposures):
        lines.append(f"        exposure_soa[write_offsets[{index}] + roi] = {exposure};\n")
    lines.append("    }\n}\n")
    return "".join(lines)


def _owner_neural_field_source(
    *,
    name: str,
    equations: str,
    inputs: Sequence[str],
    exposures: Sequence[str],
    states: Sequence[str],
    params: Sequence[str],
) -> tuple[str, list[str]]:
    known = set(inputs) | set(states) | set(params) | {"t", "dt", "node", "roi"}
    code, local_states = _owner_statement_code(equations, known_names=known, allow_local=True)
    for exposure in exposures:
        if exposure not in states:
            raise KeyError(f"neural field exposure must be a state name: {exposure}")
    lines = [_owner_prelude()]
    lines.append(
        _owner_descriptor_source(
            kind_id=4,
            name=name,
            inputs=inputs,
            exposures=exposures,
            states=states,
            params=params,
            local_states=local_states,
        )
    )
    lines.append(
        r"""
extern "C" void mind_neural_field_rule_apply(const mind_neural_field_context* ctx) {
    const int node_count = ctx->node_count;
    const int* node_to_roi = ctx->node_to_roi;
    const double* input_soa = ctx->input_soa;
    const double* previous_state_soa = ctx->previous_state_soa;
    double* state_soa = ctx->state_soa;
    const double* params = ctx->params;
    const int* local_indptr = ctx->local_indptr;
    const int* local_indices = ctx->local_indices;
    const double* local_weights = ctx->local_weights;
    const int* read_offsets = ctx->read_input_offsets;
    const double t = ctx->t;
    const double dt = ctx->dt;
    (void)t;
    for (int node = 0; node < node_count; ++node) {
        const int roi = node_to_roi[node];
"""
    )
    for index, field in enumerate(inputs):
        lines.append(f"        const double {field} = input_soa[read_offsets[{index}] + roi];\n")
    for index, field in enumerate(states):
        lines.append(f"        double& {field} = state_soa[({index} * node_count) + node];\n")
    state_index = {state: index for index, state in enumerate(states)}
    for state in local_states:
        lines.append(f"        double local_{state} = 0.0;\n")
        lines.append("        for (int edge = local_indptr[node]; edge < local_indptr[node + 1]; ++edge) {\n")
        lines.append(
            f"            local_{state} += local_weights[edge] * "
            f"previous_state_soa[({state_index[state]} * node_count) + local_indices[edge]];\n"
        )
        lines.append("        }\n")
    for index, field in enumerate(params):
        lines.append(f"        const double {field} = params[{index}];\n")
    for line in code.splitlines():
        lines.append(f"        {line}\n")
    lines.append("    }\n}\n")
    return "".join(lines), local_states


class OwnerRule:
    def __init__(
        self,
        *,
        kind: str,
        name: str,
        equations: str,
        inputs: Mapping[str, object] | Sequence[str] | str | None,
        exposures: Sequence[str] | str,
        state: Mapping[str, object] | None,
        params: Mapping[str, float] | None,
    ):
        self.kind = kind
        self.mod_name = _require_identifier(str(name), "owner name")
        self.name = _native._codegen_kernel_name(self.mod_name, f"{self.kind} owner")
        self.read = _identifier_list("input name", _schema_names("input", inputs))
        self.write = _owner_exposures(exposures)
        self.state = _owner_state_values(state)
        self.params = _owner_param_values(params)
        _reject_overlap("input", self.read, "exposure", self.write)
        _reject_overlap("input", self.read, "state", self.state_names)
        _reject_overlap("input", self.read, "param", self.param_names)
        _reject_overlap("state", self.state_names, "param", self.param_names)
        self.local_states: list[str] = []
        if kind == "region":
            source = _owner_region_source(
                name=self.name,
                equations=equations,
                inputs=self.read,
                exposures=self.write,
                states=self.state_names,
                params=self.param_names,
            )
        elif kind == "neural_field":
            source, self.local_states = _owner_neural_field_source(
                name=self.name,
                equations=equations,
                inputs=self.read,
                exposures=self.write,
                states=self.state_names,
                params=self.param_names,
            )
        else:
            raise ValueError(f"unknown owner kind: {kind}")
        cache_key = (
            kind,
            self.name,
            tuple(self.read),
            tuple(self.write),
            tuple(self.state_names),
            tuple(self.param_names),
            equations,
        )
        artifact = _OWNER_ARTIFACT_CACHE.get(cache_key)
        if artifact is None:
            artifact = _compile_artifact("owner_rule", source)
            _OWNER_ARTIFACT_CACHE[cache_key] = artifact
        self.library_path, self.cpp_path = artifact
        self._native_rule: object | None = None

    @property
    def state_names(self) -> list[str]:
        return list(self.state)

    @property
    def param_names(self) -> list[str]:
        return list(self.params)

    def _native_region(self):
        if self.kind != "region":
            raise TypeError(f"expected region owner, got {self.kind!r}")
        if self._native_rule is None:
            self._native_rule = _native._load_region_rule(self.library_path)
        return self._native_rule

    def _native_neural_field(self):
        if self.kind != "neural_field":
            raise TypeError(f"expected neural field owner, got {self.kind!r}")
        if self._native_rule is None:
            self._native_rule = _native._load_neural_field_rule(self.library_path)
        return self._native_rule

    def state_values(self) -> list[float]:
        return [float(self.state[name]) for name in self.state_names]

    def param_values(self) -> list[float]:
        return [float(self.params[name]) for name in self.param_names]

    def field_state_values(self, node_count: int) -> list[float]:
        return _field_state_values(self.state_names, {}, self.state, int(node_count))


class ModRule:
    def __init__(self, *, source: str, origin: str, path: Path | None = None):
        self.path = path
        self.origin = str(origin)
        cpp_source = _native._translate_mind_mod_to_cpp(source, self.origin)
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
        self.local_states = list(spec.get("local_state_names", []))
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
            raise TypeError(f"{self.origin} is a {self.kind!r} MindMod rule, expected {expected!r}")

    def _native_coupling(self):
        self._require_kind("coupling")
        if self._native_rule is None:
            self._native_rule = _native._load_compiled_coupling_rule(self.library_path)
        return self._native_rule

    def _native_region(self):
        self._require_kind("region")
        if self._native_rule is None:
            self._native_rule = _native._load_region_rule(self.library_path)
        return self._native_rule

    def _native_neural_field(self):
        self._require_kind("neural_field")
        if self._native_rule is None:
            self._native_rule = _native._load_neural_field_rule(self.library_path)
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

    def field_state_values(self, values: Mapping[str, object] | None, node_count: int) -> list[float]:
        self._require_kind("neural_field")
        return _field_state_values(self.state_names, self.state, values, int(node_count))

    def __repr__(self) -> str:
        return f"<mind_sim.ModRule kind={self.kind!r} name={self.mod_name!r} origin={self.origin!r}>"


def _load_mod_rule(path: str | Path) -> ModRule:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MOD_RULE_CACHE.get(key)
    if cached is None:
        cached = ModRule(
            source=resolved.read_text(encoding="utf-8"),
            origin=str(resolved),
            path=resolved,
        )
        _MOD_RULE_CACHE[key] = cached
    return cached


class ROI:
    def __init__(self, network: "Network", index: int, label: str):
        self._network = network
        self.index = int(index)
        self.label = str(label)
        self.name = self.label

    def initial_output(self, values: Mapping[str, float]) -> "ROI":
        target = self._network._initial_outputs.setdefault(self.index, {})
        for name, value in _mapping_values("initial output", values).items():
            target[name] = float(value)
        return self

    def dc_input(self, values: Mapping[str, float]) -> "ROI":
        target = self._network._dc_inputs.setdefault(self.index, {})
        for name, value in _mapping_values("dc input", values).items():
            target[name] = float(value)
        return self

    def record(self) -> "ROI":
        self._network._recorded_rois = [self.index]
        return self

    def connect(
        self,
        source,
        rule,
        *,
        params: Mapping[str, float] | None = None,
        state: Mapping[str, float] | None = None,
        random: Mapping[str, object] | None = None,
    ) -> "ROI":
        mod = self._network._resolve_mod_rule(rule)
        if not isinstance(source, ROI):
            raise TypeError("ROI.connect source must be a ROI handle")
        source_roi = self._network.roi(source)
        if mod.kind == "coupling":
            if random is not None:
                raise TypeError("coupling mod rules do not accept random providers")
            self._network._couplings.append(
                (source_roi.index, self.index, mod, mod.param_values(params))
            )
        elif mod.kind == "micro_input":
            self._network._micro_input_rules[self.index] = (
                source_roi.index,
                mod,
                mod.state_values(state),
                mod.param_values(params),
                mod.random_values(random),
            )
        elif mod.kind == "micro_output":
            if random is not None:
                raise TypeError("micro output mod rules do not accept random providers")
            self._network._micro_output_rules[self.index] = (
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
        equations: str,
        *,
        inputs: Mapping[str, object] | Sequence[str] | str | None = None,
        exposures: Sequence[str] | str,
        state: Mapping[str, object] | None = None,
        params: Mapping[str, float] | None = None,
        name: str | None = None,
    ) -> "ROI":
        owner = OwnerRule(
            kind="region",
            name=name or "region_owner",
            equations=equations,
            inputs=inputs,
            exposures=exposures,
            state=state,
            params=params,
        )
        self._network._region_owners[self.index] = (
            owner,
            owner.state_values(),
            owner.param_values(),
        )
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.ROI index={self.index} label={self.label!r}>"


class MicroCircuit:
    def __init__(self, micro, name: str = "micro"):
        self.micro = micro
        self.name = str(name)
        self.bindings: list[dict[str, object]] = []

    def bind_roi(
        self,
        roi,
        *,
        gid_ranges,
        ports: Mapping[str, object] | None = None,
    ) -> "MicroCircuit":
        begins, ends = _gid_ranges(gid_ranges)
        port_values = dict(ports or {})
        for name in port_values:
            if not isinstance(name, str):
                raise TypeError("micro ROI port names must be strings")
        self.bindings.append({
            "roi": roi,
            "begins": begins,
            "ends": ends,
            "ports": port_values,
        })
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.MicroCircuit name={self.name!r} bindings={len(self.bindings)}>"


class NeuralField:
    def __init__(
        self,
        name: str,
        equations: str | None = None,
        *,
        inputs: Mapping[str, object] | Sequence[str] | str | None = None,
        exposures: Sequence[str] | str | None = None,
        local=None,
        state: Mapping[str, object] | None = None,
        params: Mapping[str, float] | None = None,
    ):
        self.name = str(name)
        self.equations: str | None = None
        self.inputs: Mapping[str, object] | Sequence[str] | str | None = None
        self.exposures: Sequence[str] | str | None = None
        self.state: Mapping[str, object] | None = None
        self.params: Mapping[str, float] | None = None
        self.local_data = None
        if equations is not None:
            self.use(equations, inputs=inputs, exposures=exposures, state=state, params=params)
        if local is not None:
            self.local(local)

    def use(
        self,
        equations: str,
        *,
        inputs: Mapping[str, object] | Sequence[str] | str | None = None,
        exposures: Sequence[str] | str,
        state: Mapping[str, object] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "NeuralField":
        self.equations = equations
        self.inputs = inputs
        self.exposures = exposures
        self.state = state
        self.params = params
        return self

    def local(self, local) -> "NeuralField":
        self.local_data = local
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.NeuralField name={self.name!r}>"


class Network:
    def __init__(
        self,
        *,
        connectivity=None,
        labels=None,
        weights=None,
        delays=None,
        inputs: Sequence[str] | str | None = None,
        exposures: Sequence[str] | str | None = None,
    ):
        if connectivity is not None:
            self._connectivity = connectivity
        else:
            if labels is None or weights is None or delays is None:
                raise TypeError("Network requires connectivity or labels/weights/delays")
            self._connectivity = _native.Connectivity(list(labels), weights, delays)
        native_rois = list(self._connectivity.rois())
        self._labels = [roi.label for roi in native_rois]
        self._declared_inputs = _name_list("input", inputs)
        self._declared_exposures = _name_list("exposure", exposures)
        self._recorded_rois = list(range(self._connectivity.roi_count()))
        self._micro_input_ports: dict[int, dict[str, object]] = {}
        self._initial_outputs: dict[int, dict[str, float]] = {}
        self._dc_inputs: dict[int, dict[str, float]] = {}
        self._region_owners: dict[int, tuple[OwnerRule, list[float], list[float]]] = {}
        self._field_owners: list[dict[str, object]] = []
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

    def _resolve_mod_rule(self, rule) -> ModRule:
        if not isinstance(rule, (str, Path)):
            raise TypeError("MindMod rule must be a .mod path")
        path = Path(rule)
        if path.exists() or path.suffix == ".mod":
            return _load_mod_rule(path)
        raise KeyError(f"unknown MindMod rule path: {rule}")

    def use_micro(self, micro) -> "Network":
        circuit = micro if isinstance(micro, MicroCircuit) else MicroCircuit(micro)
        if self._micro_circuits:
            if circuit.micro is self._micro_circuits[0]:
                return self
            raise RuntimeError(
                "Network supports one CoreNEURON micro circuit; bind multiple ROI "
                "ranges to the same MicroCircuit owner instead of calling use_micro again"
            )
        index = 0
        self._micro_circuits.append(circuit.micro)
        for binding in circuit.bindings:
            roi_obj = self.roi(binding["roi"])
            self._micro_bindings[roi_obj.index] = (
                index,
                list(binding["begins"]),
                list(binding["ends"]),
            )
            self._micro_input_ports[roi_obj.index] = dict(binding["ports"])
        return self

    def use_neural_field(
        self,
        field: NeuralField,
        *,
        node_map,
    ) -> "Network":
        if not isinstance(field, NeuralField):
            raise TypeError("Network.use_neural_field expects a mind_sim.NeuralField")
        if field.equations is None or field.exposures is None:
            raise ValueError("NeuralField must declare equations and exposures with field.use(...)")
        owner_rule = OwnerRule(
            kind="neural_field",
            name=field.name,
            equations=field.equations,
            inputs=field.inputs,
            exposures=field.exposures,
            state=field.state,
            params=field.params,
        )
        if not isinstance(node_map, _native.NodeToRoiMap):
            raise TypeError("neural field node_map must be a mind_sim.NodeToRoiMap")
        mapping = node_map
        node_count = int(mapping.node_count)
        local = _local_connectivity(field.local_data, node_count)
        owner = {
            "name": field.name,
            "rule": owner_rule,
            "node_map": mapping,
            "local": local,
            "state": owner_rule.field_state_values(node_count),
            "params": owner_rule.param_values(),
            "reducers": [(exposure, exposure) for exposure in owner_rule.write],
        }
        self._field_owners.append(owner)
        return self

    def _schema(self) -> tuple[list[str], list[str], dict[int, set[str]], dict[int, set[str]]]:
        inputs: list[str] = list(self._declared_inputs)
        exposures: list[str] = list(self._declared_exposures)
        accepted_by_roi: dict[int, set[str]] = {roi.index: set() for roi in self._rois}
        exposed_by_roi: dict[int, set[str]] = {roi.index: set() for roi in self._rois}

        for roi, (rule, _, _) in self._region_owners.items():
            for name in rule.read:
                _add_name(inputs, name)
                accepted_by_roi[roi].add(name)
            for name in rule.write:
                _add_name(exposures, name)
                exposed_by_roi[roi].add(name)

        for owner in self._field_owners:
            rule = owner["rule"]
            owned_rois = tuple(dict.fromkeys(owner["node_map"].node_to_roi))
            for roi in owned_rois:
                for name in rule.read:
                    _add_name(inputs, name)
                    accepted_by_roi[roi].add(name)
            reducers = list(owner["reducers"])
            if not reducers:
                raise RuntimeError(
                    f"neural field {owner['name']!r} must declare at least one exposure"
                )
            for _, exposure in reducers:
                _add_name(exposures, exposure)
                for roi in owned_rois:
                    exposed_by_roi[roi].add(exposure)

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
            source_label = self._labels[source]
            target_label = self._labels[target]
            _require_names(f"{source_label} -> {target_label} READ", mod.read, exposed_by_roi[source])
            _require_names(f"{source_label} -> {target_label} WRITE", mod.write, accepted_by_roi[target])

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
        input_offsets = _offset_map(inputs, self.roi_count())
        exposure_offsets = _offset_map(exposures, self.roi_count())

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
                rule._native_region(),
                state,
                params,
                _offsets(input_offsets, rule.read),
                _offsets(exposure_offsets, rule.write),
            )

        for owner in self._field_owners:
            rule = owner["rule"]
            reducers = list(owner["reducers"])
            local = owner["local"]
            state_index_by_name = {name: index for index, name in enumerate(rule.state_names)}
            exposure_index_by_name = {name: index for index, name in enumerate(exposures)}
            native.use_neural_field(
                str(owner["name"]),
                rule._native_neural_field(),
                owner["node_map"],
                local,
                list(owner["state"]),
                list(owner["params"]),
                _offsets(input_offsets, rule.read),
                [state_index_by_name[state] for state, _ in reducers],
                [exposure_index_by_name[exposure] for _, exposure in reducers],
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
                _offsets(input_offsets, mod.read),
            )

        for roi, (_, mod, state, params) in self._micro_output_rules.items():
            native.configure_micro_output_rule(
                native.roi(roi),
                mod._native_micro_output(),
                state,
                params,
                _offsets(exposure_offsets, mod.write),
            )

        for source, target, mod, params in self._couplings:
            native.couple(
                native.roi(source),
                native.roi(target),
                mod._native_coupling(),
                params,
                _offsets(exposure_offsets, mod.read),
                _offsets(input_offsets, mod.write),
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
LocalConnectivity = _native.LocalConnectivity
NodeToRoiMap = _native.NodeToRoiMap
