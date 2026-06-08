#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def _dump_path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def _write_tsv_header_if_needed(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")


def _format_tsv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _append_tsv_row_if_requested(env_name: str, header: str, values) -> None:
    dump_path = _dump_path_from_env(env_name)
    if dump_path is None:
        return
    _write_tsv_header_if_needed(dump_path, header)
    with dump_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(_format_tsv_value(value) for value in values) + "\n")


PYR_COUNT = 800
BAS_COUNT = 200
OLM_COUNT = 200
CA3_REAL_CELL_COUNT = PYR_COUNT + BAS_COUNT + OLM_COUNT
SAMPLE_EPS_MS = 1.0e-12
MACRO_DT_MS = 0.1
MICRO_DT_MS = 0.025
CORENEURON_PSOLVE_STOP_GUARD_MS = 1.0e-9
NEURON_TIME_ALIGN_EPS_MS = 1.0e-7
PYR_CURRENT_NA = 0.1
OLM_CURRENT_NA = -25e-3
VOLTAGE_TRACE_SPECS = (
    {"label": "PYR[0].soma", "pop": "PYR", "gid": 0, "section": "soma", "loc": 0.5},
    {"label": "BAS[0].soma", "pop": "BAS", "gid": PYR_COUNT, "section": "soma", "loc": 0.5},
    {"label": "OLM[0].soma", "pop": "OLM", "gid": PYR_COUNT + BAS_COUNT, "section": "soma", "loc": 0.5},
)
_MICRO2MACRO_REPLAY_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_MACRO2MICRO_REPLAY_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def add_tvb_multiscale_paths(tvb_root: Path, tvb_multiscale_root: Path) -> None:
    for path in reversed(
        [
            tvb_root / "tvb_library",
            tvb_root / "tvb_contrib",
            tvb_root / "tvb_storage",
            tvb_root / "tvb_framework",
            tvb_multiscale_root,
        ]
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def canonical_macro_steps_from_times_ms(times_ms, dt_ms: float = MACRO_DT_MS) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    steps = np.full(times.shape, -1, dtype=int)
    finite = np.isfinite(times)
    if np.any(finite):
        steps[finite] = np.ceil((times[finite] - SAMPLE_EPS_MS) / float(dt_ms)).astype(int)
    return steps


def micro2macro_replay_path() -> Path | None:
    value = (
        os.environ.get("TVB_NETPYNE_MICRO2MACRO_REPLAY")
        or os.environ.get("TVB_NETPYNE_MICRO2MACRO_REPLAY_DUMP")
    )
    if not value:
        return None
    return Path(value)


def macro2micro_replay_path() -> Path | None:
    value = (
        os.environ.get("TVB_NETPYNE_MACRO2MICRO_REPLAY")
        or os.environ.get("TVB_NETPYNE_MACRO2MICRO_REPLAY_DUMP")
    )
    if not value:
        return None
    return Path(value)


def load_micro2macro_replay(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    stat = path.stat()
    cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MICRO2MACRO_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    valid_mind_transforms = {
        "ca3_pyr_spikes_to_vep",
        "ca3_bas_spikes_to_vep",
        "ca3_olm_spikes_to_vep",
    }
    times = []
    senders = []
    steps = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"micro2macro replay file has no header: {path}")
        for row in reader:
            transform = row.get("transform")
            if transform and transform not in valid_mind_transforms:
                continue
            if "sid" in row and row.get("sid", "") != "":
                sender = int(row["sid"])
            elif "sender" in row and row.get("sender", "") != "":
                sender = int(row["sender"])
            else:
                raise RuntimeError(f"micro2macro replay row has neither sid nor sender: {row}")

            if "spike_time_ms" in row and row.get("spike_time_ms", "") != "":
                spike_time = float(row["spike_time_ms"])
            elif "raw_time_ms" in row and row.get("raw_time_ms", "") != "":
                spike_time = float(row["raw_time_ms"])
            elif "event_time_ms" in row and row.get("event_time_ms", "") != "":
                spike_time = float(row["event_time_ms"])
            else:
                raise RuntimeError(f"micro2macro replay row has no spike time column: {row}")
            times.append(spike_time)
            senders.append(sender)
            if "window_stop_ms" in row and row.get("window_stop_ms", "") != "":
                steps.append(int(round(float(row["window_stop_ms"]) / MACRO_DT_MS)))
            elif "step" in row and row.get("step", "") != "":
                steps.append(int(row["step"]))
            else:
                steps.append(int(canonical_macro_steps_from_times_ms([spike_time], MACRO_DT_MS)[0]))

    replay_times = np.asarray(times, dtype=float)
    replay_senders = np.asarray(senders, dtype=int)
    replay_steps = np.asarray(steps, dtype=int)
    result = (replay_times, replay_senders, replay_steps)
    _MICRO2MACRO_REPLAY_CACHE.clear()
    _MICRO2MACRO_REPLAY_CACHE[cache_key] = result
    return result


def load_macro2micro_replay(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    stat = path.stat()
    cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MACRO2MICRO_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    times = []
    source_ids = []
    sample_steps = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"macro2micro replay file has no header: {path}")
        for row in reader:
            if "event_time_ms" in row and row.get("event_time_ms", "") != "":
                event_time = float(row["event_time_ms"])
            elif "spike_time_ms" in row and row.get("spike_time_ms", "") != "":
                event_time = float(row["spike_time_ms"])
            elif "time_ms" in row and row.get("time_ms", "") != "":
                event_time = float(row["time_ms"])
            else:
                raise RuntimeError(f"macro2micro replay row has no event time column: {row}")

            if "source_id" in row and row.get("source_id", "") != "":
                source_id = int(row["source_id"])
            elif "source" in row and row.get("source", "") != "":
                source_id = int(row["source"])
            elif "gid" in row and row.get("gid", "") != "":
                source_id = int(row["gid"])
            else:
                raise RuntimeError(f"macro2micro replay row has no source id column: {row}")

            if "sample_step" in row and row.get("sample_step", "") != "":
                sample_step = int(row["sample_step"])
            elif "step" in row and row.get("step", "") != "":
                sample_step = int(row["step"])
            else:
                sample_step = int(math.floor((event_time + SAMPLE_EPS_MS) / MACRO_DT_MS))

            times.append(event_time)
            source_ids.append(source_id)
            sample_steps.append(sample_step)

    replay_times = np.asarray(times, dtype=float)
    replay_sources = np.asarray(source_ids, dtype=int)
    replay_steps = np.asarray(sample_steps, dtype=int)
    if replay_times.size:
        order = np.lexsort((replay_sources, replay_times, replay_steps))
        replay_times = replay_times[order]
        replay_sources = replay_sources[order]
        replay_steps = replay_steps[order]
    result = (replay_times, replay_sources, replay_steps)
    _MACRO2MICRO_REPLAY_CACHE.clear()
    _MACRO2MICRO_REPLAY_CACHE[cache_key] = result
    return result


def align_neuron_time_to_nominal(nominal_time_ms: float, *, context: str) -> float:
    from neuron import h

    nominal_time_ms = float(nominal_time_ms)
    actual_time_ms = float(h.t)
    if not np.isfinite(actual_time_ms):
        raise RuntimeError(f"NEURON time is not finite while aligning {context}: {actual_time_ms}")
    if abs(actual_time_ms - nominal_time_ms) > NEURON_TIME_ALIGN_EPS_MS:
        raise RuntimeError(
            f"Refusing to align NEURON time for {context}: "
            f"h.t={actual_time_ms:.17g} ms, nominal={nominal_time_ms:.17g} ms"
        )
    h.t = nominal_time_ms
    return actual_time_ms


def should_attach_pc_netpyne_spike_recorder() -> bool:
    source = os.environ.get("TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE", "").strip().lower()
    return source == "pc" or os.environ.get("TVB_NETPYNE_ATTACH_PC_SPIKE_RECORDER", "0") == "1"


def prepare_coreneuron_mod_library(workdir: Path) -> Path:
    from neuron import coreneuron, h, load_mechanisms

    source_dir = Path(__file__).resolve().parent / "mod"
    build_dir = workdir / "coreneuron_mod"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    sources = [*source_dir.glob("*.mod"), *source_dir.glob("*.inc")]
    for source in sorted(sources):
        if source.name in seen_names:
            raise RuntimeError(f"duplicate MOD file name in CoreNEURON reference library: {source.name}")
        seen_names.add(source.name)
        shutil.copy2(source, build_dir / source.name)

    subprocess.run(["nrnivmodl", "-coreneuron", "."], cwd=build_dir, check=True)
    coreneuron_library = build_dir / "x86_64" / "libcorenrnmech.so"
    if not coreneuron_library.exists():
        raise RuntimeError(f"CoreNEURON mechanism library was not created: {coreneuron_library}")
    os.environ["CORENEURONLIB"] = str(coreneuron_library.resolve())
    load_mechanisms(str(build_dir))
    coreneuron.verbose = 0
    coreneuron.model_stats = False
    if not hasattr(h, "VecStim"):
        raise RuntimeError("CoreNEURON reference MOD library did not provide VecStim")
    return build_dir


def install_tvb_netpyne_nominal_runtime_hooks() -> None:
    from netpyne import sim
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    if getattr(NetpyneModule, "_mind_nominal_runtime_hooks", False):
        return

    original_prepare_simulation = NetpyneModule.prepareSimulation
    original_run = NetpyneModule.run
    original_get_spikes = NetpyneModule.getSpikes

    def prepare_simulation_with_nominal_time(self, duration):
        original_prepare_interval = sim.run.prepareSimWithIntervalFunc

        def prepare_interval_with_direct_voltage_recorder(*args, **kwargs):
            if not bool(getattr(sim.cfg, "coreneuron", False)):
                attach_direct_netpyne_voltage_recorder(self)
                attach_direct_netpyne_spike_recorder(self)
            result = original_prepare_interval(*args, **kwargs)
            if not bool(getattr(sim.cfg, "coreneuron", False)) and should_attach_pc_netpyne_spike_recorder():
                attach_pc_netpyne_spike_recorder(self)
            return result

        sim.run.prepareSimWithIntervalFunc = prepare_interval_with_direct_voltage_recorder
        try:
            result = original_prepare_simulation(self, duration)
            self._mind_absolute_run_step = int(round(float(self.time) / MACRO_DT_MS))
            self._mind_nominal_time = float(self._mind_absolute_run_step) * MACRO_DT_MS
            return result
        finally:
            sim.run.prepareSimWithIntervalFunc = original_prepare_interval

    def run_with_absolute_macro_stop(self, length):
        if not getattr(self, "_readyToRun", False):
            return original_run(self, length)

        length = float(length)
        current_step = int(getattr(self, "_mind_absolute_run_step", round(float(self.time) / MACRO_DT_MS)))
        step_count = int(round(length / MACRO_DT_MS))
        target_step = current_step + max(0, step_count)
        current_time = float(current_step) * MACRO_DT_MS
        target_time = min(float(target_step) * MACRO_DT_MS, float(sim.cfg.duration))
        align_neuron_time_to_nominal(current_time, context="TVB-NetPyNE run window start")

        self.stimulate(max(0.0, target_time - float(self.time)))

        if self.nextIntervalFuncCall:
            while self.nextIntervalFuncCall < min(target_time, sim.cfg.duration):
                interval_stop = float(self.nextIntervalFuncCall)
                sim.pc.psolve(interval_stop)
                align_neuron_time_to_nominal(interval_stop, context="TVB-NetPyNE interval callback")
                self.intervalFunc(self.time)
                self.nextIntervalFuncCall = self.time + self.interval

        if target_time > float(self.time) and self.time < sim.cfg.duration:
            sim.pc.psolve(target_time)
            align_neuron_time_to_nominal(target_time, context="TVB-NetPyNE run window stop")
            if os.environ.get("TVB_NETPYNE_FLUSH_CURRENT_TIME_EVENTS", "0") == "1":
                sim.pc.psolve(target_time)
                align_neuron_time_to_nominal(target_time, context="TVB-NetPyNE current-time event flush")

            if self.nextIntervalFuncCall:
                correction = self.time - target_time
                self.nextIntervalFuncCall += correction

        self._mind_absolute_run_step = target_step
        self._mind_nominal_time = target_time

    def get_spikes_from_live_single_host_recording(self, generatedBy=None, startingFrom=None):
        if sim.pc.nhost() != 1:
            return original_get_spikes(self, generatedBy=generatedBy, startingFrom=startingFrom)

        sim_data = getattr(sim, "simData", None)
        if sim_data is None or "spkt" not in sim_data or "spkid" not in sim_data:
            return [], []

        spktimes = np.asarray(sim_data["spkt"], dtype=float)
        spkgids = np.asarray(sim_data["spkid"], dtype=int)
        if startingFrom is not None:
            inds = np.nonzero(spktimes > float(startingFrom))
            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        if generatedBy is not None:
            inds = np.isin(spkgids, generatedBy)
            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        return spktimes, spkgids

    NetpyneModule.prepareSimulation = prepare_simulation_with_nominal_time
    NetpyneModule.run = run_with_absolute_macro_stop
    NetpyneModule.getSpikes = get_spikes_from_live_single_host_recording
    NetpyneModule._mind_nominal_runtime_hooks = True


def install_macro2micro_replay_netcon_input() -> str:
    from neuron import h
    from netpyne import sim
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    if getattr(NetpyneModule, "_mind_macro2micro_replay_netcon_input", False):
        return "macro2micro replay via NetCon.event"

    original_stimulate = NetpyneModule.stimulate

    def input_netcons_by_pre_gid():
        cache = getattr(sim.net, "_mind_input_netcons_by_pre_gid", None)
        if cache is not None:
            return cache
        cache = {}
        for post_cell in list(getattr(sim.net, "cells", []) or []):
            for conn in list(getattr(post_cell, "conns", []) or []):
                pre_gid = conn.get("preGid") if hasattr(conn, "get") else None
                if pre_gid is None:
                    continue
                try:
                    pre_gid = int(pre_gid)
                except (TypeError, ValueError):
                    continue
                netcon = conn.get("hObj") if hasattr(conn, "get") else None
                if netcon is None:
                    netcon = conn.get("hNetCon") if hasattr(conn, "get") else None
                if netcon is None or not hasattr(netcon, "event"):
                    continue
                try:
                    delay = float(conn.get("delay", netcon.delay))
                except Exception:
                    delay = 0.0
                cache.setdefault(pre_gid, []).append((netcon, delay))
        setattr(sim.net, "_mind_input_netcons_by_pre_gid", cache)
        return cache

    def clear_spike_generator_buffers(self) -> list[list[int]]:
        own_neurons_by_device = []
        for device in list(getattr(self, "spikeGenerators", []) or []):
            own_neurons = list(device.own_neurons)
            own_neurons_by_device.append(own_neurons)
            device.spikesPerNeuron = {}
        return own_neurons_by_device

    def stimulate_with_macro2micro_replay(self, length):
        replay_path = macro2micro_replay_path()
        if replay_path is None:
            return original_stimulate(self, length)

        own_neurons_by_device = clear_spike_generator_buffers(self)
        if getattr(self, "_mind_macro2micro_replay_scheduled", False):
            return None

        if sim.pc.nhost() != 1:
            raise RuntimeError("macro2micro NetCon replay currently expects a single-host NetPyNE run")
        if not own_neurons_by_device:
            raise RuntimeError("macro2micro replay found no NetPyNE spike generator devices")

        replay_times, replay_sources, replay_steps = load_macro2micro_replay(replay_path)
        direct_netcons = input_netcons_by_pre_gid()
        event_dump_path = _dump_path_from_env("TVB_NETPYNE_MACRO2MICRO_EVENT_DUMP")
        event_handle = None
        if event_dump_path is not None:
            _write_tsv_header_if_needed(
                event_dump_path,
                "interval_index\tsample_step\tsample_start_ms\tsample_stop_ms\tevent_time_ms\tsource_id\trate_hz\tmean\tnum_neurons",
            )
            event_handle = event_dump_path.open("a", encoding="utf-8")
        scheduled = 0
        current_time = float(h.t)
        try:
            for own_neurons in own_neurons_by_device:
                for event_time, source_id, sample_step in zip(replay_times, replay_sources, replay_steps):
                    source_id = int(source_id)
                    if source_id < 0 or source_id >= len(own_neurons):
                        continue
                    source_gid = int(own_neurons[source_id])
                    for netcon, delay in direct_netcons.get(source_gid, []):
                        delivery_time = float(event_time) + float(delay)
                        if delivery_time <= current_time + NEURON_TIME_ALIGN_EPS_MS:
                            raise RuntimeError(
                                "macro2micro replay attempted to schedule an already elapsed event: "
                                f"source_id={source_id}, event_time={float(event_time):.17g}, "
                                f"delay={float(delay):.17g}, h.t={current_time:.17g}"
                            )
                        netcon.event(delivery_time)
                        scheduled += 1
                    if event_handle is not None:
                        sample_start = float(sample_step) * MACRO_DT_MS
                        sample_stop = min(sample_start + MACRO_DT_MS, float(sim.cfg.duration))
                        event_handle.write(
                            f"-1\t{int(sample_step)}\t{sample_start:.17g}\t{sample_stop:.17g}\t"
                            f"{float(event_time):.17g}\t{source_id}\t\t\t{len(own_neurons)}\n"
                        )
        finally:
            if event_handle is not None:
                event_handle.close()
        self._mind_macro2micro_replay_scheduled = True
        self._mind_macro2micro_replay_scheduled_count = scheduled
        return None

    NetpyneModule.stimulate = stimulate_with_macro2micro_replay
    NetpyneModule._mind_macro2micro_replay_netcon_input = True
    return "macro2micro replay via NetCon.event"


def install_coreneuron_vecstim_tvb_netpyne_input() -> str:
    from neuron import h
    from netpyne import sim
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    if getattr(NetpyneModule, "_mind_core_vecstim_input", False):
        return "VecStim"

    original_import_model = NetpyneModule.importModel
    original_prepare_simulation = NetpyneModule.prepareSimulation
    original_run = NetpyneModule.run
    original_get_spikes = NetpyneModule.getSpikes

    def import_model_with_vecstim(self, netParams, simConfig, dt, config):
        original_import_model(self, netParams, simConfig, dt, config)
        netParams.cellParams["art_NetStim"] = {
            "cellModel": "VecStim",
            "params": {"spkTimes": []},
        }

    def prepare_simulation_without_auto_plots(self, duration):
        auto_created_pops = self._autoCreatedPops
        self._autoCreatedPops = []
        original_prepare_interval = sim.run.prepareSimWithIntervalFunc

        def prepare_interval_with_direct_voltage_recorder(*args, **kwargs):
            if not bool(getattr(sim.cfg, "coreneuron", False)):
                attach_direct_netpyne_voltage_recorder(self)
                attach_direct_netpyne_spike_recorder(self)
            result = original_prepare_interval(*args, **kwargs)
            if not bool(getattr(sim.cfg, "coreneuron", False)) and should_attach_pc_netpyne_spike_recorder():
                attach_pc_netpyne_spike_recorder(self)
            return result

        sim.run.prepareSimWithIntervalFunc = prepare_interval_with_direct_voltage_recorder
        try:
            result = original_prepare_simulation(self, duration)
            sim.cfg.analysis = {}
            self._mind_absolute_run_step = int(round(float(self.time) / MACRO_DT_MS))
            self._mind_nominal_time = float(self._mind_absolute_run_step) * MACRO_DT_MS
            return result
        finally:
            sim.run.prepareSimWithIntervalFunc = original_prepare_interval
            self._autoCreatedPops = auto_created_pops

    def run_with_absolute_macro_stop(self, length):
        if not getattr(self, "_readyToRun", False):
            return original_run(self, length)

        from netpyne import sim

        length = float(length)
        current_step = int(getattr(self, "_mind_absolute_run_step", round(float(self.time) / MACRO_DT_MS)))
        step_count = int(round(length / MACRO_DT_MS))
        target_step = current_step + max(0, step_count)
        current_time = float(current_step) * MACRO_DT_MS
        target_time = min(float(target_step) * MACRO_DT_MS, float(sim.cfg.duration))
        align_neuron_time_to_nominal(current_time, context="TVB-NetPyNE CoreNEURON run window start")

        self.stimulate(max(0.0, target_time - float(self.time)))

        if self.nextIntervalFuncCall:
            while self.nextIntervalFuncCall < min(target_time, sim.cfg.duration):
                interval_stop = float(self.nextIntervalFuncCall)
                sim.pc.psolve(interval_stop)
                align_neuron_time_to_nominal(interval_stop, context="TVB-NetPyNE CoreNEURON interval callback")
                self.intervalFunc(self.time)
                self.nextIntervalFuncCall = self.time + self.interval

        if target_time > float(self.time):
            if self.time < sim.cfg.duration:
                sim.pc.psolve(target_time)
                align_neuron_time_to_nominal(target_time, context="TVB-NetPyNE CoreNEURON run window stop")
                if os.environ.get("TVB_NETPYNE_FLUSH_CURRENT_TIME_EVENTS", "0") == "1":
                    sim.pc.psolve(target_time)
                    align_neuron_time_to_nominal(target_time, context="TVB-NetPyNE CoreNEURON current-time event flush")

                if self.nextIntervalFuncCall:
                    correction = self.time - target_time
                    self.nextIntervalFuncCall += correction

        self._mind_absolute_run_step = target_step
        self._mind_nominal_time = target_time

    def get_spikes_from_live_single_host_recording(self, generatedBy=None, startingFrom=None):
        if sim.pc.nhost() != 1:
            return original_get_spikes(self, generatedBy=generatedBy, startingFrom=startingFrom)

        sim_data = getattr(sim, "simData", None)
        if sim_data is None or "spkt" not in sim_data or "spkid" not in sim_data:
            return [], []

        spktimes = np.asarray(sim_data["spkt"], dtype=float)
        spkgids = np.asarray(sim_data["spkid"], dtype=int)
        if startingFrom is not None:
            inds = np.nonzero(spktimes > float(startingFrom))
            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        if generatedBy is not None:
            inds = np.isin(spkgids, generatedBy)
            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        return spktimes, spkgids

    def stimulate_with_vecstim(self, length):
        def input_netcons_by_pre_gid():
            cache = getattr(sim.net, "_mind_input_netcons_by_pre_gid", None)
            if cache is not None:
                return cache
            cache = {}
            for post_cell in list(getattr(sim.net, "cells", []) or []):
                for conn in list(getattr(post_cell, "conns", []) or []):
                    pre_gid = conn.get("preGid") if hasattr(conn, "get") else None
                    if pre_gid is None:
                        continue
                    try:
                        pre_gid = int(pre_gid)
                    except (TypeError, ValueError):
                        continue
                    netcon = conn.get("hObj") if hasattr(conn, "get") else None
                    if netcon is None:
                        netcon = conn.get("hNetCon") if hasattr(conn, "get") else None
                    if netcon is None or not hasattr(netcon, "event"):
                        continue
                    try:
                        delay = float(conn.get("delay", netcon.delay))
                    except Exception:
                        delay = 0.0
                    cache.setdefault(pre_gid, []).append((netcon, delay))
            setattr(sim.net, "_mind_input_netcons_by_pre_gid", cache)
            return cache

        direct_netcons = input_netcons_by_pre_gid()
        for device in self.spikeGenerators:
            own_neurons = list(device.own_neurons)
            spikes_per_neuron = dict(device.spikesPerNeuron)
            device.spikesPerNeuron = {}

            if sim.pc.nhost() > 1:
                spikes_per_neuron = sim.pc.py_broadcast(spikes_per_neuron, 0)
                own_neurons = [gid for gid in own_neurons if gid in sim.net.gid2lid]

            for local_index, gid in enumerate(own_neurons):
                spike_times = spikes_per_neuron.get(gid)
                if spike_times is None:
                    spike_times = spikes_per_neuron.get(local_index, [])
                for spike_time in spike_times:
                    for netcon, delay in direct_netcons.get(int(gid), []):
                        netcon.event(float(spike_time) + float(delay))

    NetpyneModule.importModel = import_model_with_vecstim
    NetpyneModule.prepareSimulation = prepare_simulation_without_auto_plots
    NetpyneModule.run = run_with_absolute_macro_stop
    NetpyneModule.getSpikes = get_spikes_from_live_single_host_recording
    NetpyneModule.stimulate = stimulate_with_vecstim
    NetpyneModule._compileOrLoadMod = lambda self: None
    NetpyneModule._mind_core_vecstim_input = True
    return "NetCon.event"


def netpyne_cell_sec_hobj(cell, section: str):
    secs = getattr(cell, "secs", {}) or {}
    sec = secs.get(section) if hasattr(secs, "get") else None
    if isinstance(sec, dict):
        return sec.get("hObj")
    return getattr(sec, "hObj", None)


def netpyne_cell_soma_hobj(cell):
    return netpyne_cell_sec_hobj(cell, "soma")


def attach_direct_netpyne_voltage_recorder(owner, *, gid: int = 0, pop: str = "PYR"):
    from neuron import h
    from netpyne import sim as netpyne_sim

    existing = getattr(owner, "_mind_direct_voltage_recorder", None)
    if (
        isinstance(existing, dict)
        and existing.get("time_vector") is not None
        and (existing.get("voltage_vector") is not None or existing.get("trace_records"))
    ):
        return existing

    net = getattr(netpyne_sim, "net", None)
    cells = list(getattr(net, "cells", []) or []) if net is not None else []
    gid2lid = getattr(net, "gid2lid", {}) if net is not None else {}

    def cell_gid_and_pop(cell) -> tuple[int | None, str]:
        tags = getattr(cell, "tags", {}) or {}
        raw_gid = getattr(cell, "gid", tags.get("gid", None))
        try:
            cell_gid = int(raw_gid)
        except (TypeError, ValueError):
            cell_gid = None
        return cell_gid, str(tags.get("pop", ""))

    def find_cell_for_spec(spec: dict):
        candidates = []
        spec_gid = int(spec["gid"])
        if spec_gid in gid2lid:
            try:
                candidates.append(cells[int(gid2lid[spec_gid])])
            except (IndexError, TypeError, ValueError):
                pass
        candidates.extend(cells)

        seen = set()
        unique_candidates = []
        for cell in candidates:
            identity = id(cell)
            if identity in seen:
                continue
            seen.add(identity)
            unique_candidates.append(cell)

        def score_cell(cell) -> tuple[int, int, int]:
            cell_gid, cell_pop = cell_gid_and_pop(cell)
            return (
                0 if cell_gid == spec_gid else 1,
                0 if cell_pop == str(spec["pop"]) else 1,
                int(cell_gid) if cell_gid is not None else 10**9,
            )

        for cell in sorted(unique_candidates, key=score_cell):
            cell_gid, cell_pop = cell_gid_and_pop(cell)
            if cell_gid == spec_gid or cell_pop == str(spec["pop"]):
                return cell
        return None

    time_vector = h.Vector()
    time_vector.record(h._ref_t)
    trace_records = []
    errors = []
    for spec in VOLTAGE_TRACE_SPECS:
        cell = find_cell_for_spec(spec)
        if cell is None:
            errors.append(f"could not find NetPyNE {spec['pop']} gid {spec['gid']} before finitialize")
            continue
        cell_gid, cell_pop = cell_gid_and_pop(cell)
        section_hobj = netpyne_cell_sec_hobj(cell, str(spec["section"]))
        if section_hobj is None:
            errors.append(f"could not find NetPyNE {spec['label']} section before finitialize")
            continue
        voltage_vector = h.Vector()
        voltage_vector.record(section_hobj(float(spec["loc"]))._ref_v)
        trace_records.append(
            {
                "label": str(spec["label"]),
                "gid": int(cell_gid) if cell_gid is not None else None,
                "pop": cell_pop,
                "section": str(spec["section"]),
                "loc": float(spec["loc"]),
                "voltage_vector": voltage_vector,
            }
        )

    pyr_cell = find_cell_for_spec(VOLTAGE_TRACE_SPECS[0])
    adend3_hobj = netpyne_cell_sec_hobj(pyr_cell, "Adend3") if pyr_cell is not None else None
    adend3_voltage_vector = h.Vector() if adend3_hobj is not None else None
    if adend3_voltage_vector is not None:
        adend3_voltage_vector.record(adend3_hobj(0.5)._ref_v)

    first_record = trace_records[0] if trace_records else {}
    recorder = {
        "time_vector": time_vector if trace_records else None,
        "voltage_vector": first_record.get("voltage_vector"),
        "trace_records": trace_records,
        "trace_labels": [record["label"] for record in trace_records],
        "adend3_voltage_vector": adend3_voltage_vector,
        "gid": first_record.get("gid"),
        "pop": first_record.get("pop", pop),
        "section": first_record.get("section", "soma"),
        "loc": first_record.get("loc", 0.5),
        "extra_sections": ["Adend3"] if adend3_hobj is not None else [],
        "record_step": "solver_dt",
        "install_time_ms": float(h.t),
    }
    if errors:
        recorder["trace_errors"] = errors
    setattr(owner, "_mind_direct_voltage_recorder", recorder)
    setattr(netpyne_sim, "_mind_direct_voltage_recorder", recorder)
    return recorder


def attach_direct_netpyne_spike_recorder(owner, *, threshold_mv: float = 0.0):
    from neuron import h
    from netpyne import sim as netpyne_sim

    existing = getattr(owner, "_mind_direct_spike_recorder", None)
    if isinstance(existing, dict) and existing.get("records"):
        return existing

    net = getattr(netpyne_sim, "net", None)
    cells = list(getattr(net, "cells", []) or []) if net is not None else []
    records = []
    for cell in cells:
        tags = getattr(cell, "tags", {}) or {}
        pop = str(tags.get("pop", ""))
        if pop not in {"PYR", "BAS", "OLM"}:
            continue
        raw_gid = getattr(cell, "gid", tags.get("gid", None))
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        soma_hobj = netpyne_cell_soma_hobj(cell)
        if soma_hobj is None:
            continue
        netcon = h.NetCon(soma_hobj(0.5)._ref_v, None, sec=soma_hobj)
        netcon.threshold = float(threshold_mv)
        time_vector = h.Vector()
        netcon.record(time_vector)
        records.append(
            {
                "gid": gid,
                "pop": pop,
                "netcon": netcon,
                "time_vector": time_vector,
            }
        )

    recorder = {
        "records": records,
        "threshold_mv": float(threshold_mv),
        "source": "direct soma NetCon.record",
    }
    setattr(owner, "_mind_direct_spike_recorder", recorder)
    setattr(netpyne_sim, "_mind_direct_spike_recorder", recorder)
    return recorder


def read_direct_netpyne_spikes(owner, generated_by=None) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from netpyne import sim as netpyne_sim
    except Exception:
        netpyne_sim = None

    recorder = getattr(owner, "_mind_direct_spike_recorder", None)
    if not isinstance(recorder, dict) and netpyne_sim is not None:
        recorder = getattr(netpyne_sim, "_mind_direct_spike_recorder", None)
    if not isinstance(recorder, dict):
        return None

    records = recorder.get("records")
    if not records:
        return None

    generated_set = None
    if generated_by is not None:
        generated_set = {int(gid) for gid in np.asarray(generated_by, dtype=int).ravel()}

    times = []
    gids = []
    for record in records:
        gid = int(record["gid"])
        if generated_set is not None and gid not in generated_set:
            continue
        vector = record.get("time_vector")
        if vector is None:
            continue
        spike_times = np.asarray(vector, dtype=float)
        if spike_times.size == 0:
            continue
        times.append(spike_times)
        gids.append(np.full(spike_times.shape, gid, dtype=int))

    if not times:
        return np.asarray([], dtype=float), np.asarray([], dtype=int)

    spktimes = np.concatenate(times)
    spkgids = np.concatenate(gids)
    order = np.argsort(spktimes, kind="mergesort")
    return spktimes[order], spkgids[order]


def attach_pc_netpyne_spike_recorder(owner):
    from neuron import h
    from netpyne import sim as netpyne_sim

    existing = getattr(owner, "_mind_pc_spike_recorder", None)
    if isinstance(existing, dict) and existing.get("time_vector") is not None:
        return existing

    time_vector = h.Vector()
    gid_vector = h.Vector()
    netpyne_sim.pc.spike_record(-1, time_vector, gid_vector)
    recorder = {
        "time_vector": time_vector,
        "gid_vector": gid_vector,
        "source": "ParallelContext.spike_record(-1)",
    }
    setattr(owner, "_mind_pc_spike_recorder", recorder)
    setattr(netpyne_sim, "_mind_pc_spike_recorder", recorder)
    return recorder


def read_pc_netpyne_spikes(owner, generated_by=None) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from netpyne import sim as netpyne_sim
    except Exception:
        netpyne_sim = None

    recorder = getattr(owner, "_mind_pc_spike_recorder", None)
    if not isinstance(recorder, dict) and netpyne_sim is not None:
        recorder = getattr(netpyne_sim, "_mind_pc_spike_recorder", None)
    if not isinstance(recorder, dict):
        return None

    spktimes = vector_to_numpy(recorder.get("time_vector"))
    spkgids = vector_to_numpy(recorder.get("gid_vector")).astype(int)
    if spktimes.size != spkgids.size:
        raise RuntimeError(
            "TVB-NetPyNE PC spike recorder returned mismatched time/gid vector sizes: "
            f"{spktimes.size} vs {spkgids.size}"
        )
    if generated_by is not None and spktimes.size:
        generated_set = np.asarray(generated_by, dtype=int).ravel()
        keep = np.isin(spkgids, generated_set)
        spktimes = spktimes[keep]
        spkgids = spkgids[keep]
    if spktimes.size:
        order = np.argsort(spktimes, kind="mergesort")
        spktimes = spktimes[order]
        spkgids = spkgids[order]
    return spktimes, spkgids


def dump_direct_netpyne_spikes_if_requested(orchestrator, metadata: dict) -> None:
    dump_path = _dump_path_from_env("TVB_NETPYNE_DIRECT_SPIKE_DUMP")
    if dump_path is None:
        return

    try:
        from netpyne import sim as netpyne_sim
    except Exception as exc:  # pragma: no cover - diagnostic path
        metadata["direct_spike_dump_error"] = str(exc)
        return

    candidates = [netpyne_sim]
    try:
        spiking = orchestrator.spikeNet_app.spiking_cosimulator
        candidates.extend([getattr(spiking, "netpyne_instance", None), spiking])
    except Exception:
        pass

    recorder = None
    for owner in candidates:
        if owner is None:
            continue
        recorder = getattr(owner, "_mind_direct_spike_recorder", None)
        if isinstance(recorder, dict) and recorder.get("records"):
            break
    if not isinstance(recorder, dict) or not recorder.get("records"):
        metadata["direct_spike_dump_error"] = "direct spike recorder is not available"
        return

    _write_tsv_header_if_needed(dump_path, "gid\tpop\traw_time_ms\tstep")
    count = 0
    with dump_path.open("a", encoding="utf-8") as handle:
        for record in recorder.get("records", []):
            vector = record.get("time_vector")
            if vector is None:
                continue
            times = vector_to_numpy(vector)
            gid = int(record["gid"])
            pop = str(record.get("pop", ""))
            steps = canonical_macro_steps_from_times_ms(times, MACRO_DT_MS)
            for raw_time, step in zip(times, steps):
                handle.write(f"{gid}\t{pop}\t{float(raw_time):.17g}\t{int(step)}\n")
                count += 1
    metadata["direct_spike_dump"] = str(dump_path)
    metadata["direct_spike_dump_count"] = int(count)


def make_initial_conditions(labels: list[str], micro_index: int, history_steps: int = 1) -> np.ndarray:
    rng = np.random.default_rng(1234)
    micro_label = labels[int(micro_index)]
    propagation_labels = {
        "Left-CA1",
        "Right-CA1",
        "Left-CA3",
        "Right-CA3",
        "Left-subiculum",
        "Right-subiculum",
        "Left-entorhinal",
        "Right-entorhinal",
    }
    x0 = []
    for label in labels:
        if label == micro_label:
            x0.append(0.0)
        elif label in propagation_labels:
            x0.append(-1.9)
        else:
            x0.append(-2.4)
    x0 = np.asarray(x0, dtype=float)
    initial = np.zeros((1, 2, len(labels), 1), dtype=float)
    initial[0, 0, :, 0] = x0 + 0.02 * rng.standard_normal(len(labels))
    initial[0, 0, int(micro_index), 0] = 0.0
    initial[0, 1, :, 0] = 0.0
    return np.repeat(initial, max(1, int(history_steps)), axis=0)


def tvb_history_steps(weights: np.ndarray, delays_ms: np.ndarray, dt_ms: float) -> int:
    delays_ms = np.asarray(delays_ms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0.0):
        return 1
    max_delay_ms = float(np.max(delays_ms[weights > 0.0]))
    return int(np.ceil(max_delay_ms / float(dt_ms))) + 1


def make_tvb_connectivity(labels: list[str], weights: np.ndarray, delays_ms: np.ndarray, speed_mm_per_ms: float):
    from tvb.datatypes.connectivity import Connectivity

    label_array = np.asarray(labels, dtype="<U128")
    weights = np.asarray(weights, dtype=float)
    delays_ms = np.asarray(delays_ms, dtype=float)
    centres = np.zeros((len(labels), 3), dtype=float)
    centres[:, 0] = np.arange(len(labels), dtype=float)

    conn = Connectivity()
    conn.weights = weights.copy()
    conn.tract_lengths = np.where(weights > 0.0, delays_ms * float(speed_mm_per_ms), 0.0)
    conn.speed = np.asarray([float(speed_mm_per_ms)], dtype=float)
    conn.region_labels = label_array
    conn.centres = centres
    conn.orientations = np.zeros((len(labels), 3), dtype=float)
    conn.areas = np.ones(len(labels), dtype=float)
    conn.cortical = np.asarray([label.startswith("ctx-") for label in labels], dtype=bool)
    conn.hemispheres = np.asarray([label.startswith("Right") or label.startswith("ctx-rh") for label in labels], dtype=bool)
    conn.configure()
    return conn


def read_live_netpyne_voltage(orchestrator, metadata: dict) -> float | None:
    cell_candidates = []
    try:
        from netpyne import sim as netpyne_sim

        net = getattr(netpyne_sim, "net", None)
        if net is not None:
            cell_candidates.extend(getattr(net, "cells", []) or [])
    except Exception as exc:  # pragma: no cover - diagnostic path
        metadata["voltage_live_read_import_error"] = str(exc)

    try:
        spiking = orchestrator.spikeNet_app.spiking_cosimulator
        netpyne_instance = getattr(spiking, "netpyne_instance", None)
        for owner in (netpyne_instance, spiking):
            net = getattr(owner, "net", None)
            if net is not None:
                cell_candidates.extend(getattr(net, "cells", []) or [])
    except Exception as exc:  # pragma: no cover - diagnostic path
        metadata["voltage_live_read_orchestrator_error"] = str(exc)

    seen = set()
    unique_cells = []
    for cell in cell_candidates:
        identity = id(cell)
        if identity in seen:
            continue
        seen.add(identity)
        unique_cells.append(cell)

    def score_cell(cell) -> tuple[int, int]:
        tags = getattr(cell, "tags", {}) or {}
        gid = getattr(cell, "gid", tags.get("gid", None))
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            gid_int = 10**9
        pop = str(tags.get("pop", ""))
        return (0 if gid_int == 0 else 1, 0 if pop == "PYR" else 1)

    for cell in sorted(unique_cells, key=score_cell):
        tags = getattr(cell, "tags", {}) or {}
        if getattr(cell, "gid", tags.get("gid", None)) != 0 and tags.get("pop") != "PYR":
            continue
        secs = getattr(cell, "secs", {}) or {}
        soma = secs.get("soma") if hasattr(secs, "get") else None
        h_section = None
        if isinstance(soma, dict):
            h_section = soma.get("hObj")
        else:
            h_section = getattr(soma, "hObj", None)
        if h_section is None:
            continue
        try:
            return float(h_section(0.5).v)
        except Exception as exc:  # pragma: no cover - diagnostic path
            metadata["voltage_live_read_segment_error"] = str(exc)
    metadata["voltage_live_read_error"] = "recorded NetPyNE cell soma segment is not available"
    return None


def append_final_voltage_sample_if_available(
    orchestrator,
    metadata: dict,
    time_ms: np.ndarray,
    voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if time_ms.size == 0 or voltage.size == 0 or time_ms.size != voltage.size:
        return time_ms, voltage
    duration_ms = float(metadata.get("duration_ms", np.nan))
    dt_ms = float(metadata.get("dt_micro_ms", 0.025))
    if not np.isfinite(duration_ms) or not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return time_ms, voltage
    if time_ms[-1] >= duration_ms - (0.5 * dt_ms):
        return time_ms, voltage
    missing_one_endpoint = math.isclose(
        float(time_ms[-1] + dt_ms),
        duration_ms,
        rel_tol=0.0,
        abs_tol=max(1.0e-9, dt_ms * 1.0e-9),
    )
    if not missing_one_endpoint:
        return time_ms, voltage

    final_voltage = read_live_netpyne_voltage(orchestrator, metadata)
    if final_voltage is None:
        metadata["voltage_final_sample_append"] = "skipped: live NEURON value unavailable"
        return time_ms, voltage

    metadata["voltage_final_sample_append"] = "live NEURON soma v at simulation stop"
    metadata["voltage_final_sample_time_ms"] = duration_ms
    metadata["voltage_final_sample_value"] = float(final_voltage)
    return (
        np.concatenate([time_ms, np.asarray([duration_ms], dtype=float)]),
        np.concatenate([voltage, np.asarray([final_voltage], dtype=float)]),
    )


def direct_voltage_recorder_candidates(orchestrator):
    candidates = []
    try:
        spiking = orchestrator.spikeNet_app.spiking_cosimulator
        netpyne_instance = getattr(spiking, "netpyne_instance", None)
        candidates.extend([netpyne_instance, spiking])
    except Exception:
        pass
    try:
        from netpyne import sim as netpyne_sim

        candidates.append(netpyne_sim)
    except Exception:
        pass

    seen = set()
    for owner in candidates:
        if owner is None:
            continue
        recorder = getattr(owner, "_mind_direct_voltage_recorder", None)
        if not isinstance(recorder, dict):
            continue
        identity = id(recorder)
        if identity in seen:
            continue
        seen.add(identity)
        yield recorder


def vector_to_numpy(vector) -> np.ndarray:
    if vector is None:
        return np.asarray([], dtype=float)
    if hasattr(vector, "to_python"):
        return np.asarray(vector.to_python(), dtype=float)
    return np.asarray(list(vector), dtype=float)


def extract_direct_recorded_voltage(orchestrator, metadata: dict) -> tuple[np.ndarray, np.ndarray] | None:
    for recorder in direct_voltage_recorder_candidates(orchestrator):
        error = recorder.get("error")
        if error:
            metadata["voltage_direct_recorder_error"] = str(error)
            continue
        time_ms = vector_to_numpy(recorder.get("time_vector"))
        voltage = vector_to_numpy(recorder.get("voltage_vector"))
        if time_ms.size == 0 or voltage.size == 0:
            metadata["voltage_direct_recorder_error"] = {
                "reason": "empty direct recorder vectors",
                "time_samples": int(time_ms.size),
                "voltage_samples": int(voltage.size),
            }
            continue
        if time_ms.size != voltage.size:
            sample_count = min(time_ms.size, voltage.size)
            metadata["voltage_direct_trimmed_samples"] = {
                "time": int(time_ms.size),
                "voltage": int(voltage.size),
            }
            time_ms = time_ms[:sample_count]
            voltage = voltage[:sample_count]
        duration_ms = float(metadata.get("duration_ms", np.nan))
        if np.isfinite(duration_ms):
            keep = (time_ms >= 0.0) & (time_ms <= duration_ms + 1.0e-9)
            if not np.all(keep):
                metadata["voltage_direct_duration_trimmed_samples"] = int(np.size(keep) - np.count_nonzero(keep))
                time_ms = time_ms[keep]
                voltage = voltage[keep]
        metadata["voltage_source"] = "direct NetPyNE h.Vector.record"
        metadata["voltage_direct_recorder"] = {
            "gid": recorder.get("gid"),
            "pop": recorder.get("pop"),
            "section": recorder.get("section"),
            "loc": recorder.get("loc"),
            "record_step": recorder.get("record_step"),
            "install_time_ms": recorder.get("install_time_ms"),
            "sample_count": int(time_ms.size),
        }
        return time_ms, voltage
    return None


def extract_direct_recorded_named_voltage(
    orchestrator,
    metadata: dict,
    *,
    vector_key: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    for recorder in direct_voltage_recorder_candidates(orchestrator):
        error = recorder.get("error")
        if error:
            continue
        time_ms = vector_to_numpy(recorder.get("time_vector"))
        voltage = vector_to_numpy(recorder.get(vector_key))
        if time_ms.size == 0 or voltage.size == 0:
            continue
        if time_ms.size != voltage.size:
            sample_count = min(time_ms.size, voltage.size)
            metadata[f"{label}_voltage_direct_trimmed_samples"] = {
                "time": int(time_ms.size),
                "voltage": int(voltage.size),
            }
            time_ms = time_ms[:sample_count]
            voltage = voltage[:sample_count]
        duration_ms = float(metadata.get("duration_ms", np.nan))
        if np.isfinite(duration_ms):
            keep = (time_ms >= 0.0) & (time_ms <= duration_ms + 1.0e-9)
            if not np.all(keep):
                metadata[f"{label}_voltage_direct_duration_trimmed_samples"] = (
                    int(np.size(keep) - np.count_nonzero(keep))
                )
                time_ms = time_ms[keep]
                voltage = voltage[keep]
        metadata[f"{label}_voltage_direct_recorder"] = {
            "gid": recorder.get("gid"),
            "pop": recorder.get("pop"),
            "section": label,
            "loc": 0.5,
            "record_step": recorder.get("record_step"),
            "install_time_ms": recorder.get("install_time_ms"),
            "sample_count": int(time_ms.size),
        }
        return time_ms, voltage
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def extract_direct_recorded_voltage_traces(
    orchestrator,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for recorder in direct_voltage_recorder_candidates(orchestrator):
        error = recorder.get("error")
        if error:
            continue
        time_ms = vector_to_numpy(recorder.get("time_vector"))
        records = list(recorder.get("trace_records") or [])
        if time_ms.size == 0 or not records:
            continue

        labels = []
        traces = []
        sample_count = int(time_ms.size)
        for record in records:
            voltage = vector_to_numpy(record.get("voltage_vector"))
            if voltage.size == 0:
                continue
            labels.append(str(record.get("label", f"{record.get('pop')}.{record.get('section')}")))
            traces.append(voltage)
            sample_count = min(sample_count, int(voltage.size))
        if not traces or sample_count <= 0:
            continue

        if any(trace.size != time_ms.size for trace in traces):
            metadata["voltage_traces_direct_trimmed_samples"] = {
                "time": int(time_ms.size),
                "traces": [int(trace.size) for trace in traces],
            }
        time_ms = time_ms[:sample_count]
        voltage_traces = np.column_stack([trace[:sample_count] for trace in traces])

        duration_ms = float(metadata.get("duration_ms", np.nan))
        if np.isfinite(duration_ms):
            keep = (time_ms >= 0.0) & (time_ms <= duration_ms + 1.0e-9)
            if not np.all(keep):
                metadata["voltage_traces_direct_duration_trimmed_samples"] = (
                    int(np.size(keep) - np.count_nonzero(keep))
                )
                time_ms = time_ms[keep]
                voltage_traces = voltage_traces[keep]

        metadata["voltage_traces_source"] = "direct NetPyNE h.Vector.record"
        metadata["voltage_trace_labels"] = labels
        metadata["voltage_traces_direct_recorder"] = [
            {
                "label": str(record.get("label")),
                "gid": record.get("gid"),
                "pop": record.get("pop"),
                "section": record.get("section"),
                "loc": record.get("loc"),
            }
            for record in records
        ]
        metadata["voltage_traces_sample_count"] = int(time_ms.size)
        return np.asarray(labels, dtype=object), time_ms, voltage_traces

    return (
        np.asarray([], dtype=object),
        np.asarray([], dtype=float),
        np.empty((0, 0), dtype=float),
    )


def extract_recorded_voltage(orchestrator, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    direct = extract_direct_recorded_voltage(orchestrator, metadata)
    if direct is not None:
        return direct

    spiking = orchestrator.spikeNet_app.spiking_cosimulator
    netpyne_instance = getattr(spiking, "netpyne_instance", None)
    all_sim_data = getattr(netpyne_instance, "allSimData", None)
    if all_sim_data is None:
        all_sim_data = getattr(spiking, "allSimData", None)
    if all_sim_data is None:
        try:
            from netpyne import sim as netpyne_sim

            all_sim_data = getattr(netpyne_sim, "allSimData", None)
            if all_sim_data is None:
                all_sim_data = getattr(netpyne_sim, "simData", None)
        except Exception as exc:  # pragma: no cover - diagnostic path
            metadata["voltage_global_read_error"] = str(exc)
    if all_sim_data is None:
        metadata["voltage_read_error"] = "NetPyNE allSimData is not available"
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    trace = all_sim_data.get("voltage") if hasattr(all_sim_data, "get") else None
    if trace is None:
        metadata["voltage_read_error"] = "NetPyNE voltage trace is not available"
        if hasattr(all_sim_data, "keys"):
            metadata["voltage_available_keys"] = [str(key) for key in all_sim_data.keys()]
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if isinstance(trace, dict):
        if not trace:
            metadata["voltage_read_error"] = "NetPyNE voltage trace is empty"
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        voltage = np.asarray(next(iter(trace.values())), dtype=float)
    else:
        voltage = np.asarray(trace, dtype=float)

    if hasattr(all_sim_data, "get") and all_sim_data.get("t") is not None:
        time_ms = np.asarray(all_sim_data.get("t"), dtype=float)
    else:
        time_ms = np.arange(voltage.size, dtype=float) * 0.025
    if time_ms.size != voltage.size:
        sample_count = min(time_ms.size, voltage.size)
        metadata["voltage_trimmed_samples"] = {"time": int(time_ms.size), "voltage": int(voltage.size)}
        time_ms = time_ms[:sample_count]
        voltage = voltage[:sample_count]
    return append_final_voltage_sample_if_available(orchestrator, metadata, time_ms, voltage)


def dump_netpyne_connections_from_sim(path: Path) -> None:
    from netpyne import sim as netpyne_sim

    net = getattr(netpyne_sim, "net", None)
    cells = list(getattr(net, "cells", []) or []) if net is not None else []
    _write_tsv_header_if_needed(
        path,
        "cell_order\tconn_order\tpost_gid\tpost_pop\tpre_gid\tpre_pop\tpre_label\t"
        "sec\tloc\tsyn_mech\tweight\tdelay\tthreshold",
    )
    with path.open("a", encoding="utf-8") as handle:
        for cell_order, cell in enumerate(cells):
            post_gid = getattr(cell, "gid", "")
            tags = getattr(cell, "tags", {}) or {}
            post_pop = tags.get("pop", "")
            for conn_order, conn in enumerate(getattr(cell, "conns", []) or []):
                row = [
                    cell_order,
                    conn_order,
                    post_gid,
                    post_pop,
                    conn.get("preGid", conn.get("pre_gid", "")),
                    conn.get("prePop", conn.get("pre_pop", "")),
                    conn.get("preLabel", conn.get("pre_label", "")),
                    conn.get("sec", ""),
                    conn.get("loc", ""),
                    conn.get("synMech", ""),
                    conn.get("weight", ""),
                    conn.get("delay", ""),
                    conn.get("threshold", ""),
                ]
                handle.write("\t".join(_format_tsv_value(value) for value in row) + "\n")


def pt3d_geom(
    points: list[tuple[float, float, float, float]],
    *,
    nseg: int,
    Ra: float,
    cm: float,
) -> dict:
    return {
        "pt3d": [(float(x), float(y), float(z), float(diam)) for x, y, z, diam in points],
        "nseg": int(nseg),
        "Ra": float(Ra),
        "cm": float(cm),
    }


def pyrcell_rule(pyr_current_na: float) -> dict:
    common_pas = {"e": -70.0, "g": 0.0000357}
    return {
        "conds": {"cellType": "PYR"},
        "secs": {
            "soma": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 0.0, 20.0), (0.0, 0.0, 20.0, 20.0)],
                    nseg=1,
                    Ra=150.0,
                    cm=1.0,
                ),
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {},
                    "kacurrent": {},
                    "kdrcurrent": {},
                    "hcurrent": {},
                },
                "pointps": {"inj": {"mod": "IClamp", "loc": 0.5, "del": 0.2, "dur": 1.0e9, "amp": float(pyr_current_na)}},
            },
            "Bdend": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, -200.0, 2.0)],
                    nseg=1,
                    Ra=150.0,
                    cm=1.0,
                ),
                "topol": {"parentSec": "soma", "parentX": 0.0, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 1.0},
                    "kacurrent": {},
                    "kdrcurrent": {},
                    "hcurrent": {},
                },
            },
            "Adend1": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 20.0, 2.0), (0.0, 0.0, 170.0, 2.0)],
                    nseg=1,
                    Ra=150.0,
                    cm=1.0,
                ),
                "topol": {"parentSec": "soma", "parentX": 0.5, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.072},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -82.0, "g": 0.0002},
                },
            },
            "Adend2": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 170.0, 2.0), (0.0, 0.0, 320.0, 2.0)],
                    nseg=1,
                    Ra=150.0,
                    cm=1.0,
                ),
                "topol": {"parentSec": "Adend1", "parentX": 1.0, "childX": 0.0},
                "mechs": {
                    "pas": dict(common_pas),
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.0, "gd": 0.120},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -90.0, "g": 0.0004},
                },
            },
            "Adend3": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 320.0, 2.0), (0.0, 0.0, 470.0, 2.0)],
                    nseg=1,
                    Ra=150.0,
                    cm=2.0,
                ),
                "topol": {"parentSec": "Adend2", "parentX": 1.0, "childX": 0.0},
                "mechs": {
                    "pas": {"e": -70.0, "g": 0.0000714},
                    "nacurrent": {"ki": 0.5},
                    "kacurrent": {"g": 0.0, "gd": 0.200},
                    "kdrcurrent": {},
                    "hcurrent": {"v50": -90.0, "g": 0.0007},
                },
            },
        },
    }


def interneuron_rule(cell_type: str, olm: bool, olm_current_na: float) -> dict:
    total_area_um2 = 10000.0
    diam = float(total_area_um2**0.5)
    length = float(diam / np.pi)
    mechs = {
        "pas": {"e": -65.0, "g": 0.1e-3},
        "Nafbwb": {},
        "Kdrbwb": {},
    }
    if olm:
        mechs.update({"Iholmw": {}, "Caolmw": {}, "ICaolmw": {}, "KCaolmw": {}})
    pointps = {}
    if olm:
        pointps["inj"] = {"mod": "IClamp", "loc": 0.5, "del": 0.2, "dur": 1.0e9, "amp": float(olm_current_na)}
    return {
        "conds": {"cellType": cell_type},
        "secs": {
            "soma": {
                "geom": pt3d_geom(
                    [(0.0, 0.0, 0.0, diam), (0.0, 0.0, length, diam)],
                    nseg=1,
                    Ra=35.4,
                    cm=1.0,
                ),
                "mechs": mechs,
                "pointps": pointps,
            }
        },
    }


SYNAPSE_LOCS = {
    "somaGABAf": ("soma", 0.5, "gaba_fast"),
    "somaAMPAf": ("soma", 0.5, "ampa"),
    "BdendAMPA": ("Bdend", 1.0, "ampa"),
    "BdendNMDA": ("Bdend", 1.0, "nmda"),
    "Adend2GABAs": ("Adend2", 0.5, "gaba_slow"),
    "Adend3GABAf": ("Adend3", 0.5, "gaba_fast"),
    "Adend3AMPAf": ("Adend3", 0.5, "ampa"),
    "Adend3NMDA": ("Adend3", 0.5, "nmda"),
    "somaGABAss": ("soma", 0.5, "gaba_septal"),
    "somaNMDA": ("soma", 0.5, "nmda"),
}


def synapse_params(kind: str) -> dict:
    if kind == "ampa":
        return {"mod": "MyExp2SynBB", "tau1": 0.05, "tau2": 5.3, "e": 0.0}
    if kind == "gaba_fast":
        return {"mod": "MyExp2SynBB", "tau1": 0.07, "tau2": 9.1, "e": -80.0}
    if kind == "gaba_slow":
        return {"mod": "MyExp2SynBB", "tau1": 0.2, "tau2": 20.0, "e": -80.0}
    if kind == "gaba_septal":
        return {"mod": "MyExp2SynBB", "tau1": 20.0, "tau2": 40.0, "e": -80.0}
    if kind == "nmda":
        return {
            "mod": "MyExp2SynNMDABB",
            "tau1": 0.05,
            "tau2": 5.3,
            "tau1NMDA": 15.0,
            "tau2NMDA": 150.0,
            "r": 1.0,
            "e": 0.0,
        }
    raise ValueError(f"unknown synapse kind {kind}")


def install_ca3_stimulus_mapping() -> None:
    from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

    current = NetpyneModule.connectStimuli
    if getattr(current, "_mind_ca3_stimulus_mapping", False):
        return
    original_connect_stimuli = current

    def connect_stimuli(self, sourcePop, targetPop, weight, delay, receptorType, prob=None):
        if targetPop == "PYR" and receptorType in SYNAPSE_LOCS:
            source_cells = int(self.netParams.popParams[sourcePop]["numCells"])
            target_cells = int(self.netParams.popParams[targetPop]["numCells"])
            sec, loc, _kind = SYNAPSE_LOCS[receptorType]
            params = {
                "preConds": {"pop": sourcePop},
                "postConds": {"pop": targetPop},
                "weight": weight,
                "delay": delay,
                "synMech": receptorType,
                "sec": sec,
                "loc": loc,
            }
            if source_cells == target_cells:
                params["connList"] = [[int(i), int(i)] for i in range(source_cells)]
            elif prob:
                params["probability"] = prob
            elif source_cells <= target_cells:
                params["divergence"] = 1.0
            else:
                params["convergence"] = 1.0
            self.netParams.connParams[sourcePop + "->" + targetPop] = params
            return
        return original_connect_stimuli(self, sourcePop, targetPop, weight, delay, receptorType, prob)

    connect_stimuli._mind_ca3_stimulus_mapping = True
    NetpyneModule.connectStimuli = connect_stimuli


def make_netpyne_params(
    *,
    connections: bool,
    wseed: int,
    duration_ms: float,
    pyr_current_na: float,
    olm_current_na: float,
    record_voltage: bool,
    coreneuron_enabled: bool = True,
):
    from netpyne import specs

    net_params = specs.NetParams()
    net_params.defaultThreshold = 0.0
    net_params.scale = 1.0
    net_params.sizeX = 1.0
    net_params.sizeY = 1.0
    net_params.sizeZ = 1.0
    net_params.cellParams["PYR"] = pyrcell_rule(pyr_current_na)
    net_params.cellParams["BAS"] = interneuron_rule("BAS", olm=False, olm_current_na=olm_current_na)
    net_params.cellParams["OLM"] = interneuron_rule("OLM", olm=True, olm_current_na=olm_current_na)
    net_params.popParams["PYR"] = {"cellType": "PYR", "numCells": PYR_COUNT}
    net_params.popParams["BAS"] = {"cellType": "BAS", "numCells": BAS_COUNT}
    net_params.popParams["OLM"] = {"cellType": "OLM", "numCells": OLM_COUNT}
    for syn_name, (_sec, _loc, kind) in SYNAPSE_LOCS.items():
        net_params.synMechParams[syn_name] = synapse_params(kind)
    if connections:
        random.seed(int(wseed))

        net_params.connParams["PYR->BAS:somaNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(pyr_local), int(bas_local)]
                for bas_local in range(BAS_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 100)
            ],
            "weight": 1.15 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaNMDA",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->OLM:somaNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(pyr_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 10)
            ],
            "weight": 0.7e-3,
            "delay": 2.0,
            "synMech": "somaNMDA",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->PYR:BdendNMDA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(pyr_local_pre), int(pyr_local_post)]
                for pyr_local_post in range(PYR_COUNT)
                for pyr_local_pre in random.sample(range(PYR_COUNT), 25)
                if pyr_local_pre != pyr_local_post
            ],
            "weight": 0.004e-3,
            "delay": 2.0,
            "synMech": "BdendNMDA",
            "sec": "Bdend",
            "loc": 1.0,
        }

        net_params.connParams["PYR->BAS:somaAMPAf"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(pyr_local), int(bas_local)]
                for bas_local in range(BAS_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 100)
            ],
            "weight": 0.3 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaAMPAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->OLM:somaAMPAf"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(pyr_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for pyr_local in random.sample(range(PYR_COUNT), 10)
            ],
            "weight": 0.3 * 1.2e-3,
            "delay": 2.0,
            "synMech": "somaAMPAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["PYR->PYR:BdendAMPA"] = {
            "preConds": {"pop": "PYR"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(pyr_local_pre), int(pyr_local_post)]
                for pyr_local_post in range(PYR_COUNT)
                for pyr_local_pre in random.sample(range(PYR_COUNT), 25)
                if pyr_local_pre != pyr_local_post
            ],
            "weight": 0.5 * 0.04e-3,
            "delay": 2.0,
            "synMech": "BdendAMPA",
            "sec": "Bdend",
            "loc": 1.0,
        }

        net_params.connParams["BAS->BAS:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "BAS"},
            "connList": [
                [int(bas_local_pre), int(bas_local_post)]
                for bas_local_post in range(BAS_COUNT)
                for bas_local_pre in random.sample(range(BAS_COUNT), 60)
                if bas_local_pre != bas_local_post
            ],
            "weight": 3.0 * 1.5e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["BAS->PYR:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(bas_local), int(pyr_local)]
                for pyr_local in range(PYR_COUNT)
                for bas_local in random.sample(range(BAS_COUNT), 50)
            ],
            "weight": 4.0 * 0.18e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["BAS->OLM:somaGABAf"] = {
            "preConds": {"pop": "BAS"},
            "postConds": {"pop": "OLM"},
            "connList": [
                [int(bas_local), int(olm_local)]
                for olm_local in range(OLM_COUNT)
                for bas_local in random.sample(range(BAS_COUNT), 17)
            ],
            "weight": 0.05 * 4.0 * 0.18e-3,
            "delay": 2.0,
            "synMech": "somaGABAf",
            "sec": "soma",
            "loc": 0.5,
        }

        net_params.connParams["OLM->PYR:Adend2GABAs"] = {
            "preConds": {"pop": "OLM"},
            "postConds": {"pop": "PYR"},
            "connList": [
                [int(olm_local), int(pyr_local)]
                for pyr_local in range(PYR_COUNT)
                for olm_local in random.sample(range(OLM_COUNT), 10)
            ],
            "weight": 0.08 * 4.0 * 3.0 * 6.0e-3,
            "delay": 2.0,
            "synMech": "Adend2GABAs",
            "sec": "Adend2",
            "loc": 0.5,
        }

    sim_config = specs.SimConfig()
    sim_config.duration = float(duration_ms)
    sim_config.dt = 0.025
    sim_config.hParams["v_init"] = -65.0
    sim_config.createNEURONObj = True
    sim_config.createPyStruct = True
    sim_config.addSynMechs = True
    sim_config.pt3dRelativeToCellLocation = False
    if record_voltage:
        sim_config.recordCells = [("PYR", 0), ("BAS", 0), ("OLM", 0)]
        sim_config.recordTraces = {"voltage": {"sec": "soma", "loc": 0.5, "var": "v"}}
        sim_config.recordTime = True
    else:
        sim_config.recordCells = []
        sim_config.recordTraces = {}
        sim_config.recordTime = False
    sim_config.recordCellsSpikes = -1
    sim_config.recordStep = 0.025
    sim_config.use_local_dt = False
    sim_config.coreneuron = bool(coreneuron_enabled)
    sim_config.recordStim = False
    sim_config.saveCellSecs = False
    sim_config.saveCellConns = False
    sim_config.gatherOnlySimData = True
    sim_config.timing = False
    sim_config.verbose = False
    sim_config.progressBar = 0
    sim_config.printRunTime = False
    sim_config.printPopAvgRates = False
    sim_config.analysis = {}
    return net_params, sim_config


class CA3NetpyneBuilderMixin:
    connections = True
    wseed = 4321
    duration_ms = 20.0
    pyr_current_na = 0.1
    olm_current_na = -25e-3
    record_voltage = True
    coreneuron_enabled = True
    modeldb_dir = Path(__file__).resolve().parents[1] / "mind_sim" / "mod"
    population_order = 1

    def configure(self):
        from neuron import load_mechanisms

        load_mechanisms(str(self.modeldb_dir))
        net_params, sim_config = make_netpyne_params(
            connections=bool(self.connections),
            wseed=int(self.wseed),
            duration_ms=float(self.duration_ms),
            pyr_current_na=float(self.pyr_current_na),
            olm_current_na=float(self.olm_current_na),
            record_voltage=bool(self.record_voltage),
            coreneuron_enabled=bool(self.coreneuron_enabled),
        )
        super().configure(net_params, sim_config, autoCreateSpikingNodes=True)

    def proxy_node_synaptic_model_funcs(self):
        return {"PYR": lambda _source_node, _target_node: "Adend3AMPAf"}

    def set_defaults(self):
        self.populations = [
            {"label": "PYR", "model": "PYR", "nodes": self.spiking_nodes_inds, "params": {"global_label": "PYR"}, "scale": PYR_COUNT},
            {"label": "BAS", "model": "BAS", "nodes": self.spiking_nodes_inds, "params": {"global_label": "BAS"}, "scale": BAS_COUNT},
            {"label": "OLM", "model": "OLM", "nodes": self.spiking_nodes_inds, "params": {"global_label": "OLM"}, "scale": OLM_COUNT},
        ]
        self.populations_connections = []
        self.nodes_connections = []
        self.output_devices = [self.set_spike_recorder()]
        self.input_devices = []

    def set_spike_recorder(self):
        connections = OrderedDict()
        connections["CA3"] = ("PYR", "BAS", "OLM")
        return {
            "model": "spike_recorder_all",
            "params": {},
            "connections": connections,
            "nodes": self.spiking_nodes_inds,
        }

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super().build()


class CA3TVBNetpyneInterfaceBuilderMixin:
    ca3_spikes_to_epileptor_x = None
    ca3_coupling_input_to_spike_rate = None

    def default_output_config(self):
        drive_weight = float(getattr(self, "_ca3_drive_weight", 0.02e-3))
        drive_delay_ms = float(getattr(self, "_ca3_drive_delay_ms", 0.2))
        interface = self._get_output_interfaces()
        interface["model"] = "RATE"
        interface["transformer_model"] = self.ca3_coupling_input_to_spike_rate
        # With coupling_mode="TVB", this selects TVB's computed incoming coupling
        # for the x1 coupling variable, i.e. the CA3 ca3_input analogue, not raw x1.
        interface["voi"] = "x1"
        interface["populations"] = "PYR"
        interface["spiking_proxy_inds"] = self.proxy_inds
        interface["proxy_inds"] = self.proxy_inds
        interface["coupling_mode"] = "TVB"
        interface["cvoi"] = np.asarray([0], dtype=int)
        interface["transformer_params"] = {
            "base_hz": np.asarray([1.0], dtype=float),
            "gain_hz": np.asarray([45.0], dtype=float),
            "max_rate_hz": np.asarray([120.0], dtype=float),
            "threshold": np.asarray([-0.35], dtype=float),
            "slope": np.asarray([4.0], dtype=float),
        }
        interface["proxy_params"] = {"number_of_neurons": PYR_COUNT}
        interface["weights"] = drive_weight
        interface["delays"] = drive_delay_ms
        interface["receptor_type"] = lambda _source_node, _target_node: "Adend3AMPAf"

    def default_input_config(self):
        drive_delay_ms = float(getattr(self, "_ca3_drive_delay_ms", 0.2))
        interface = self._get_input_interfaces()
        interface["model"] = "SPIKES"
        interface["proxy"] = "SPIKES"
        interface["voi"] = "x1"
        interface["populations"] = ("PYR", "BAS", "OLM")
        interface["spiking_proxy_inds"] = self.proxy_inds
        interface["proxy_inds"] = self.proxy_inds
        interface["transformer_model"] = self.ca3_spikes_to_epileptor_x
        interface["transformer_params"] = {
            "tau_pyr_ms": np.asarray([50.0], dtype=float),
            "tau_bas_ms": np.asarray([20.0], dtype=float),
            "tau_olm_ms": np.asarray([80.0], dtype=float),
            "x_baseline": np.asarray([-1.8], dtype=float),
            "pyr_gain": np.asarray([2.0], dtype=float),
            "bas_gain": np.asarray([-0.7], dtype=float),
            "olm_gain": np.asarray([-0.4], dtype=float),
            "exchange_window_ms": np.asarray([float(getattr(self, "_ca3_exchange_window_ms", 0.5))], dtype=float),
        }
        interface["proxy_params"] = {}
        interface["delays"] = drive_delay_ms


def build_tvb_model(labels: list[str], micro_index: int, macro_i_ext: float, kvf: float, ks: float):
    from tvb.simulator.models.epileptor import Epileptor2D

    micro_label = labels[int(micro_index)]
    propagation_labels = {
        "Left-CA1",
        "Right-CA1",
        "Left-CA3",
        "Right-CA3",
        "Left-subiculum",
        "Right-subiculum",
        "Left-entorhinal",
        "Right-entorhinal",
    }
    x0_values = []
    for label in labels:
        if label == micro_label:
            x0_values.append(-1.6)
        elif label in propagation_labels:
            x0_values.append(-1.9)
        else:
            x0_values.append(-2.4)
    x0 = np.asarray(x0_values, dtype=float)
    return Epileptor2D(
        a=np.asarray([1.0], dtype=float),
        b=np.asarray([3.0], dtype=float),
        c=np.asarray([1.0], dtype=float),
        d=np.asarray([5.0], dtype=float),
        r=np.asarray([0.00035], dtype=float),
        x0=x0,
        Iext=np.asarray([float(macro_i_ext)], dtype=float),
        slope=np.asarray([0.0], dtype=float),
        Kvf=np.asarray([float(kvf)], dtype=float),
        Ks=np.asarray([float(ks)], dtype=float),
        tt=np.asarray([1.0], dtype=float),
        modification=np.asarray([False], dtype=bool),
        variables_of_interest=("x1", "z"),
    )


def configure_tvb_multiscale(
    *,
    labels: list[str],
    weights: np.ndarray,
    delays: np.ndarray,
    ca3_index: int,
    duration_ms: float,
    macro_i_ext: float,
    drive_weight: float,
    pyr_current_na: float,
    olm_current_na: float,
    conduction_speed_mm_per_ms: float,
    output_base: Path,
    exchange_window_ms: float,
    mod_library_dir: Path,
    record_voltage: bool,
    coreneuron_enabled: bool,
):
    from tvb.basic.profile import TvbProfile

    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    from tvb.simulator import coupling, integrators, monitors
    from tvb.basic.neotraits.api import Attr, NArray
    from tvb_multiscale.core.interfaces.base.transformers.models.base import LinearRate
    from tvb_multiscale.core.interfaces.base.transformers.models.elephant import ElephantSpikesHistogramRate
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorNetpyneBuilder
    from tvb_multiscale.tvb_netpyne.config import Config, initialize_logger
    from tvb_multiscale.tvb_netpyne.interfaces.models.default import DefaultTVBNetpyneInterfaceBuilder
    from tvb_multiscale.tvb_netpyne.interfaces.io import NetpyneSpikeRecorderSet
    from tvb_multiscale.tvb_netpyne.netpyne_models.devices import (
        NetpyneOutputDeviceDict,
        NetpyneOutputSpikeDeviceDict,
        NetpyneSpikeRecorder,
    )
    from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
    from tvb_multiscale.tvb_netpyne.orchestrators import TVBNetpyneSerialOrchestrator

    install_ca3_stimulus_mapping()

    class NetpyneAllSpikeRecorder(NetpyneSpikeRecorder):
        def __init__(self, *args, **kwargs):
            self._population_labels = []
            self.latestRecordTime = 0.0
            self.latestRecordStep = 0
            super().__init__(*args, **kwargs)

        @property
        def population_label(self):
            return "+".join(self._population_labels)

        @population_label.setter
        def population_label(self, value):
            if value not in self._population_labels:
                self._population_labels = [*self._population_labels, value]

        @property
        def neurons(self):
            return np.asarray(
                [gid for label in self._population_labels for gid in self.netpyne_instance.cellGidsForPop(label)],
                dtype=int,
            )

        def get_new_events(self, variables=None, **filter_kwargs):
            try:
                from neuron import h
            except Exception:  # pragma: no cover - diagnostic path
                h = None
            window_start_step = int(getattr(self, "latestRecordStep", 0))
            window_stop = float(getattr(self.netpyne_instance, "_mind_nominal_time", self.netpyne_instance.time))
            window_stop_step = int(
                getattr(
                    self.netpyne_instance,
                    "_mind_absolute_run_step",
                    round(window_stop / MACRO_DT_MS),
                )
            )
            window_start = float(window_start_step) * MACRO_DT_MS
            window_stop = float(window_stop_step) * MACRO_DT_MS
            api_spikes = self.netpyne_instance.getSpikes(
                generatedBy=self.neurons,
                startingFrom=None,
            )
            requested_source = os.environ.get("TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE", "").strip().lower()
            if not requested_source:
                if micro2macro_replay_path() is not None:
                    requested_source = "replay"
                else:
                    requested_source = (
                        "netpyne"
                        if os.environ.get("TVB_NETPYNE_DISABLE_DIRECT_MICRO2MACRO", "0") == "1"
                        else "direct"
                    )
            pc_spikes = (
                read_pc_netpyne_spikes(self.netpyne_instance, generated_by=self.neurons)
                if requested_source == "pc"
                else None
            )
            direct_spikes = (
                read_direct_netpyne_spikes(self.netpyne_instance, generated_by=self.neurons)
                if requested_source == "direct"
                else None
            )
            if requested_source == "pc":
                if pc_spikes is None:
                    raise RuntimeError("TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE=pc but no PC spike recorder is attached")
                spktimes, spkgids = pc_spikes
                source_label = "pc"
            elif requested_source == "direct":
                if direct_spikes is None:
                    raise RuntimeError(
                        "TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE=direct but no direct spike recorder is attached"
                    )
                spktimes, spkgids = direct_spikes
                source_label = "direct"
            elif requested_source == "netpyne":
                spktimes, spkgids = api_spikes
                source_label = "netpyne"
            elif requested_source == "replay":
                replay_path = micro2macro_replay_path()
                if replay_path is None:
                    raise RuntimeError(
                        "TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE=replay requires "
                        "TVB_NETPYNE_MICRO2MACRO_REPLAY"
                    )
                replay_times, replay_senders, replay_steps = load_micro2macro_replay(replay_path)
                if replay_times.size:
                    in_replay_window = (replay_steps > window_start_step) & (replay_steps <= window_stop_step)
                    generated_set = np.asarray(self.neurons, dtype=int).ravel()
                    in_replay_window &= np.isin(replay_senders, generated_set)
                    spktimes = replay_times[in_replay_window]
                    spkgids = replay_senders[in_replay_window]
                else:
                    spktimes = replay_times
                    spkgids = replay_senders
                source_label = "replay"
            else:
                raise RuntimeError(
                    "TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE must be one of: direct, pc, netpyne, replay; "
                    f"got {requested_source!r}"
                )
            spktimes = np.asarray(spktimes, dtype=float)
            spkgids = np.asarray(spkgids, dtype=int)
            source_count = int(spktimes.size)
            step_min = ""
            step_max = ""
            target_step_count = 0
            if spktimes.size:
                spike_steps = canonical_macro_steps_from_times_ms(spktimes, MACRO_DT_MS)
                step_min = int(np.min(spike_steps))
                step_max = int(np.max(spike_steps))
                target_step_count = int(np.count_nonzero(spike_steps == window_stop_step))
                _append_tsv_row_if_requested(
                    "TVB_NETPYNE_RECORDER_READ_DUMP",
                    "window_start_step\twindow_stop_step\twindow_start_ms\twindow_stop_ms\t"
                    "h_t_ms\tnetpyne_time_ms\tnominal_time_ms\tsource\tsource_count\t"
                    "source_step_min\tsource_step_max\ttarget_step_count\tselected_count",
                    [
                        window_start_step,
                        window_stop_step,
                        window_start,
                        window_stop,
                        float(h.t) if h is not None else "",
                        float(self.netpyne_instance.time),
                        float(getattr(self.netpyne_instance, "_mind_nominal_time", self.netpyne_instance.time)),
                        source_label,
                        source_count,
                        step_min,
                        step_max,
                        target_step_count,
                        int(np.count_nonzero((spike_steps > window_start_step) & (spike_steps <= window_stop_step))),
                    ],
                )
                in_window = (spike_steps > window_start_step) & (spike_steps <= window_stop_step)
                spktimes = spktimes[in_window]
                spkgids = spkgids[in_window]
            else:
                _append_tsv_row_if_requested(
                    "TVB_NETPYNE_RECORDER_READ_DUMP",
                    "window_start_step\twindow_stop_step\twindow_start_ms\twindow_stop_ms\t"
                    "h_t_ms\tnetpyne_time_ms\tnominal_time_ms\tsource\tsource_count\t"
                    "source_step_min\tsource_step_max\ttarget_step_count\tselected_count",
                    [
                        window_start_step,
                        window_stop_step,
                        window_start,
                        window_stop,
                        float(h.t) if h is not None else "",
                        float(self.netpyne_instance.time),
                        float(getattr(self.netpyne_instance, "_mind_nominal_time", self.netpyne_instance.time)),
                        source_label,
                        source_count,
                        step_min,
                        step_max,
                        target_step_count,
                        0,
                    ],
                )

            num_spikes = len(spktimes)
            if num_spikes > 0:
                period = window_stop - window_start
                if period > 0.0:
                    rate = 1000 * num_spikes / len(self.neurons) / period
                    print(
                        f"Netpyne:: recorded {len(spktimes)} spikes from {self.population_label}. "
                            f"Approx. rate: {rate}. Timeframe {window_start} + {period}"
                    )
            self.latestRecordStep = window_stop_step
            self.latestRecordTime = window_stop
            return {"senders": spkgids, "times": spktimes}

    class CA3SpikesToEpileptorX(ElephantSpikesHistogramRate):
        input_buffer = Attr(field_type=list, required=True, default=[])
        tau_pyr_ms = NArray(label="PYR activity decay tau", required=True, default=np.asarray([50.0], dtype=float))
        tau_bas_ms = NArray(label="BAS activity decay tau", required=True, default=np.asarray([20.0], dtype=float))
        tau_olm_ms = NArray(label="OLM activity decay tau", required=True, default=np.asarray([80.0], dtype=float))
        x_baseline = NArray(label="Epileptor x baseline", required=True, default=np.asarray([-1.8], dtype=float))
        pyr_gain = NArray(label="PYR Epileptor x gain", required=True, default=np.asarray([2.0], dtype=float))
        bas_gain = NArray(label="BAS Epileptor x gain", required=True, default=np.asarray([-0.7], dtype=float))
        olm_gain = NArray(label="OLM Epileptor x gain", required=True, default=np.asarray([-0.4], dtype=float))
        exchange_window_ms = NArray(label="Exchange window", required=True, default=np.asarray([0.5], dtype=float))
        activity = NArray(label="Activity state", required=False, default=np.asarray([], dtype=float))

        def _compute(self, input_buffer, *args, **kwargs):
            proxy_count = len(input_buffer)
            if self.activity.size != proxy_count * 3:
                self.activity = np.zeros(proxy_count * 3, dtype=float)
            activity_state = self.activity.reshape((proxy_count, 3))
            tau_pyr_ms = self._assert_size("tau_pyr_ms")
            tau_bas_ms = self._assert_size("tau_bas_ms")
            tau_olm_ms = self._assert_size("tau_olm_ms")
            x_baseline = self._assert_size("x_baseline")
            pyr_gain = self._assert_size("pyr_gain")
            bas_gain = self._assert_size("bas_gain")
            olm_gain = self._assert_size("olm_gain")
            self._assert_size("exchange_window_ms")
            step_dt_ms = float(self.dt)
            input_first_step = int(self.input_time[0])
            input_last_step = int(self.input_time[-1])
            parsed_buffers = []
            for proxy_index, proxy_buffer in enumerate(input_buffer):
                if isinstance(proxy_buffer, dict):
                    spikes = np.asarray(proxy_buffer.get("times", []), dtype=float)
                    senders = np.asarray(proxy_buffer.get("senders", []), dtype=int)
                else:
                    spikes = np.asarray(proxy_buffer, dtype=float)
                    senders = np.full(spikes.shape, -1, dtype=int)
                output_spike_steps = np.asarray([], dtype=int)
                if spikes.size:
                    output_spike_steps = canonical_macro_steps_from_times_ms(spikes, step_dt_ms)
                parsed_buffers.append((spikes, senders, output_spike_steps))
            steps = np.arange(input_first_step, input_last_step + 1, dtype=int)
            output = np.zeros((proxy_count, steps.size), dtype=float)
            input_dump_path = _dump_path_from_env("TVB_NETPYNE_MICRO2MACRO_INPUT_DUMP")
            if input_dump_path is not None:
                _write_tsv_header_if_needed(
                    input_dump_path,
                    "input_first_step\tinput_last_step\tproxy_index\tbuffer_type\tcount\t"
                    "time_min_ms\ttime_max_ms\tsender_min\tsender_max\tkeys",
                )
                with input_dump_path.open("a", encoding="utf-8") as input_dump_handle:
                    for proxy_index, proxy_buffer in enumerate(input_buffer):
                        if isinstance(proxy_buffer, dict):
                            spikes_dbg = np.asarray(proxy_buffer.get("times", []), dtype=float)
                            senders_dbg = np.asarray(proxy_buffer.get("senders", []), dtype=int)
                            keys_dbg = ",".join(str(key) for key in sorted(proxy_buffer.keys()))
                            buffer_type = "dict"
                        else:
                            spikes_dbg = np.asarray(proxy_buffer, dtype=float)
                            senders_dbg = np.asarray([], dtype=int)
                            keys_dbg = ""
                            buffer_type = type(proxy_buffer).__name__
                        if spikes_dbg.size:
                            time_min = f"{float(np.min(spikes_dbg)):.17g}"
                            time_max = f"{float(np.max(spikes_dbg)):.17g}"
                        else:
                            time_min = ""
                            time_max = ""
                        if senders_dbg.size:
                            sender_min = str(int(np.min(senders_dbg)))
                            sender_max = str(int(np.max(senders_dbg)))
                        else:
                            sender_min = ""
                            sender_max = ""
                        input_dump_handle.write(
                            f"{int(self.input_time[0])}\t{int(self.input_time[-1])}\t{int(proxy_index)}\t"
                            f"{buffer_type}\t{int(spikes_dbg.size)}\t{time_min}\t{time_max}\t"
                            f"{sender_min}\t{sender_max}\t{keys_dbg}\n"
                        )
            spike_dump_path = _dump_path_from_env("TVB_NETPYNE_MICRO2MACRO_SPIKE_DUMP")
            if spike_dump_path is not None:
                _write_tsv_header_if_needed(
                    spike_dump_path,
                    "step\tt0_ms\tt1_ms\tproxy_index\traw_time_ms\tevent_time_ms\tsender\tlocal_sender\tpopulation\tclamped",
                )
            spike_dump_handle = spike_dump_path.open("a", encoding="utf-8") if spike_dump_path is not None else None
            try:
                for proxy_index, (spikes, senders, output_spike_steps) in enumerate(parsed_buffers):
                    pyr_activity, bas_activity, olm_activity = activity_state[proxy_index]
                    for step_index, step in enumerate(steps):
                        t1 = float(step) * step_dt_ms
                        t0 = float(step - 1) * step_dt_ms
                        current_time = t0
                        window_spikes = output_spike_steps == int(step)
                        window_times = spikes[window_spikes]
                        if senders.size == spikes.size and np.any(senders >= 0):
                            window_senders = senders[window_spikes]
                        else:
                            window_senders = np.full(window_times.shape, 0, dtype=int)
                        if window_times.size:
                            order = np.argsort(window_times, kind="mergesort")
                            for spike_time, sender in zip(window_times[order], window_senders[order]):
                                event_time = min(max(float(spike_time), current_time), t1)
                                decay_ms = max(0.0, event_time - current_time)
                                pyr_activity *= np.exp(-decay_ms / float(tau_pyr_ms[proxy_index]))
                                bas_activity *= np.exp(-decay_ms / float(tau_bas_ms[proxy_index]))
                                olm_activity *= np.exp(-decay_ms / float(tau_olm_ms[proxy_index]))
                                current_time = event_time
                                local_sender = self._ca3_local_sender(int(sender))
                                if 0 <= local_sender < PYR_COUNT:
                                    population = "PYR"
                                    pyr_activity += 1.0 / float(PYR_COUNT)
                                elif PYR_COUNT <= local_sender < PYR_COUNT + BAS_COUNT:
                                    population = "BAS"
                                    bas_activity += 1.0 / float(BAS_COUNT)
                                elif PYR_COUNT + BAS_COUNT <= local_sender < CA3_REAL_CELL_COUNT:
                                    population = "OLM"
                                    olm_activity += 1.0 / float(OLM_COUNT)
                                else:
                                    population = "unknown"
                                if spike_dump_handle is not None:
                                    spike_dump_handle.write(
                                        f"{int(step)}\t{t0:.17g}\t{t1:.17g}\t{int(proxy_index)}\t"
                                        f"{float(spike_time):.17g}\t{event_time:.17g}\t{int(sender)}\t"
                                        f"{int(local_sender)}\t{population}\t{int(event_time != float(spike_time))}\n"
                                    )
                        decay_ms = max(0.0, t1 - current_time)
                        pyr_activity *= np.exp(-decay_ms / float(tau_pyr_ms[proxy_index]))
                        bas_activity *= np.exp(-decay_ms / float(tau_bas_ms[proxy_index]))
                        olm_activity *= np.exp(-decay_ms / float(tau_olm_ms[proxy_index]))
                        output[proxy_index, step_index] = (
                            float(x_baseline[proxy_index])
                            + float(pyr_gain[proxy_index]) * pyr_activity
                            + float(bas_gain[proxy_index]) * bas_activity
                            + float(olm_gain[proxy_index]) * olm_activity
                        )
                    activity_state[proxy_index] = [pyr_activity, bas_activity, olm_activity]
            finally:
                if spike_dump_handle is not None:
                    spike_dump_handle.close()
            return output

        @staticmethod
        def _ca3_local_sender(sender: int) -> int:
            sender = int(sender)
            if sender >= CA3_REAL_CELL_COUNT:
                sender -= PYR_COUNT
            return sender

    class CA3CouplingInputToSpikeRate(LinearRate):
        base_hz = NArray(label="baseline event rate", required=True, default=np.asarray([1.0], dtype=float))
        gain_hz = NArray(label="sigmoid gain", required=True, default=np.asarray([45.0], dtype=float))
        max_rate_hz = NArray(label="maximum event rate", required=True, default=np.asarray([120.0], dtype=float))
        threshold = NArray(label="input threshold", required=True, default=np.asarray([-0.35], dtype=float))
        slope = NArray(label="input slope", required=True, default=np.asarray([4.0], dtype=float))

        def _compute(self, input_buffer):
            base_hz = self._assert_size("base_hz")
            gain_hz = self._assert_size("gain_hz")
            max_rate_hz = self._assert_size("max_rate_hz")
            threshold = self._assert_size("threshold")
            slope = self._assert_size("slope")
            output = []
            for proxy_index, proxy_buffer in enumerate(input_buffer):
                ca3_input = np.asarray(proxy_buffer, dtype=float)
                exponent = np.clip(
                    -float(slope[proxy_index]) * (ca3_input - float(threshold[proxy_index])),
                    -60.0,
                    60.0,
                )
                rate = float(base_hz[proxy_index]) + float(gain_hz[proxy_index]) / (1.0 + np.exp(exponent))
                output.append(np.clip(rate, 0.0, float(max_rate_hz[proxy_index])))
            return output

    NetpyneOutputSpikeDeviceDict["spike_recorder_all"] = NetpyneAllSpikeRecorder
    NetpyneOutputDeviceDict["spike_recorder_all"] = NetpyneAllSpikeRecorder
    NetpyneSpikeRecorderSet.model = "spike_recorder_all"
    NetpyneSpikeRecorderSet._spikeNet_output_device_type = NetpyneAllSpikeRecorder

    class CA3NetpyneBuilder(CA3NetpyneBuilderMixin, NetpyneNetworkBuilder):
        pass

    class CA3TVBNetpyneInterfaceBuilder(CA3TVBNetpyneInterfaceBuilderMixin, DefaultTVBNetpyneInterfaceBuilder):
        pass

    config = Config(output_base=str(output_base), initialize_logger=True, verbosity=0)
    config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder_all"] = {}
    config.SIMULATION_LENGTH = float(duration_ms)
    config.TVB_TO_SPIKING_DT_RATIO = 4
    config.MIN_SPIKING_DT = 0.025
    config.DEFAULT_SPIKING_MIN_DELAY = 0.025

    conn = make_tvb_connectivity(labels, weights, delays, conduction_speed_mm_per_ms)
    cosim_builder = CoSimulatorNetpyneBuilder()
    cosim_builder.connectivity = conn
    cosim_builder.model = build_tvb_model(labels, ca3_index, macro_i_ext=macro_i_ext, kvf=0.35, ks=0.0)
    cosim_builder.coupling = coupling.Linear(a=np.asarray([1.0], dtype=float))
    cosim_builder.integrator = integrators.EulerDeterministic(dt=0.1)
    cosim_builder.noise_strength = np.asarray([0.0], dtype=float)
    cosim_builder.monitors = (monitors.Raw(period=0.1),)
    cosim_builder.monitor_period = 0.1
    cosim_builder.simulation_length = float(duration_ms)
    cosim_builder.initial_conditions = make_initial_conditions(labels, ca3_index, tvb_history_steps(weights, delays, 0.1))
    cosim_builder.scale_connectivity_weights = None
    cosim_builder.scale_connectivity_weights_by_percentile = None
    cosim_builder.remove_self_connections = True
    cosim_builder.delays_flag = True
    cosim_builder.min_tract_length = 0.1 * conduction_speed_mm_per_ms

    ca3_builder = CA3NetpyneBuilder()
    ca3_builder.config = config
    ca3_builder.spiking_nodes_inds = np.asarray([ca3_index], dtype=int)
    ca3_builder.duration_ms = float(duration_ms) + 1.0
    ca3_builder.pyr_current_na = float(pyr_current_na)
    ca3_builder.olm_current_na = float(olm_current_na)
    ca3_builder.record_voltage = bool(record_voltage)
    ca3_builder.coreneuron_enabled = bool(coreneuron_enabled)
    ca3_builder.modeldb_dir = Path(mod_library_dir)
    ca3_builder.population_order = 1

    interface_builder = CA3TVBNetpyneInterfaceBuilder()
    interface_builder.config = config
    interface_builder.proxy_inds = np.asarray([ca3_index], dtype=int)
    interface_builder.exclusive_nodes = True
    interface_builder.model = "RATE"
    interface_builder.default_coupling_mode = "TVB"
    interface_builder._ca3_drive_weight = float(drive_weight)
    interface_builder._ca3_drive_delay_ms = 0.2
    interface_builder._ca3_exchange_window_ms = float(exchange_window_ms)
    interface_builder.ca3_coupling_input_to_spike_rate = CA3CouplingInputToSpikeRate
    interface_builder.ca3_spikes_to_epileptor_x = CA3SpikesToEpileptorX
    interface_builder.output_interfaces = []
    interface_builder.input_interfaces = []

    logger = initialize_logger(__name__, config=config)
    orchestrator = TVBNetpyneSerialOrchestrator(config=config, logger=logger)
    orchestrator.tvb_app.cosimulator_builder = cosim_builder
    orchestrator.spikeNet_app.spikeNet_builder = ca3_builder
    orchestrator.tvb_app.interfaces_builder = interface_builder
    return orchestrator


def save_outputs(
    output: Path,
    orchestrator,
    labels: list[str],
    weights: np.ndarray,
    delays: np.ndarray,
    metadata: dict,
    ca3_index: int,
    exchange_window_ms: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tvb_results = orchestrator.tvb_app.results
    tvb_time = np.asarray([], dtype=float)
    tvb_data = np.asarray([], dtype=float)
    if tvb_results:
        first = tvb_results[0]
        if first is not None:
            tvb_time = np.asarray(first[0], dtype=float)
            tvb_data = np.asarray(first[1], dtype=float)
    voltage_labels, voltage_trace_time, voltage_traces = extract_direct_recorded_voltage_traces(
        orchestrator,
        metadata,
    )
    if voltage_traces.size:
        voltage_time = voltage_trace_time
        voltage = voltage_traces[:, 0].copy()
    else:
        voltage_time, voltage = extract_recorded_voltage(orchestrator, metadata)
        voltage_labels = np.asarray(["PYR[0].soma"], dtype=object) if voltage.size else np.asarray([], dtype=object)
        voltage_trace_time = voltage_time
        voltage_traces = voltage[:, None] if voltage.size else np.empty((0, 0), dtype=float)
    adend3_voltage_time, adend3_voltage = extract_direct_recorded_named_voltage(
        orchestrator,
        metadata,
        vector_key="adend3_voltage_vector",
        label="Adend3",
    )
    dump_direct_netpyne_spikes_if_requested(orchestrator, metadata)

    macro_x = np.asarray([], dtype=float)
    macro_z = np.asarray([], dtype=float)
    if tvb_data.ndim == 4 and tvb_data.shape[1] >= 2:
        macro_x = tvb_data[:, 0, :, 0].copy()
        macro_z = tvb_data[:, 1, :, 0].copy()
    elif tvb_data.ndim == 3 and tvb_data.shape[1] >= 2:
        macro_x = tvb_data[:, 0, :].copy()
        macro_z = tvb_data[:, 1, :].copy()
    if tvb_time.size:
        duration_ms = float(metadata["duration_ms"])
        keep = (tvb_time >= 0.0) & (tvb_time <= duration_ms + 1e-9)
        tvb_time = tvb_time[keep]
        if tvb_data.size:
            tvb_data = tvb_data[keep]
        if macro_x.size:
            macro_x = macro_x[keep]
            macro_z = macro_z[keep]
    macro_exposures = (
        np.stack((macro_x, macro_z), axis=2)
        if macro_x.size and macro_z.size
        else np.empty((0, len(labels), 2), dtype=float)
    )

    np.savez_compressed(
        output,
        labels=np.asarray(labels, dtype=object),
        weights=weights,
        delays=delays,
        exposure_names=np.asarray(["x", "z"], dtype=object),
        time_ms=tvb_time,
        tvb_time_ms=tvb_time,
        tvb_raw=tvb_data,
        macro_exposures=macro_exposures,
        macro_x=macro_x,
        macro_z=macro_z,
        voltage_time=voltage_time,
        voltage=voltage,
        voltage_trace_time=voltage_trace_time,
        voltage_labels=voltage_labels,
        voltage_traces=voltage_traces,
        adend3_voltage_time=adend3_voltage_time,
        adend3_voltage=adend3_voltage,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def force_tvb_coupling_scale(orchestrator, scale: float) -> float:
    cosimulator = orchestrator.tvb_app.cosimulator
    value = np.asarray([float(scale)], dtype=float)
    cosimulator.coupling.a = value
    cosimulator.coupling.configure()
    for monitor in getattr(cosimulator, "cosim_monitors", ()):
        monitor_coupling = getattr(monitor, "coupling", None)
        if monitor_coupling is not None and hasattr(monitor_coupling, "a"):
            monitor_coupling.a = value.copy()
            monitor_coupling.configure()
    try:
        orchestrator.tvb_app.dumb_tvb_simulator_serialized()
    except Exception:
        pass
    return float(cosimulator.coupling.a[0])


def force_tvb_synchronization_time(orchestrator, synchronization_time_ms: float, dt_macro_ms: float) -> tuple[float, int]:
    cosimulator = orchestrator.tvb_app.cosimulator
    synchronization_steps_float = float(synchronization_time_ms) / float(dt_macro_ms)
    synchronization_steps = int(round(synchronization_steps_float))
    if synchronization_steps < 1 or not math.isclose(
        synchronization_steps_float,
        synchronization_steps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("TVB synchronization time must be an integer multiple of dt_macro")

    cosimulator.synchronization_time = float(synchronization_time_ms)
    cosimulator.synchronization_n_step = int(synchronization_steps)
    if hasattr(cosimulator, "_configure_synchronization_time"):
        cosimulator._configure_synchronization_time()

    actual_time = float(getattr(cosimulator, "synchronization_time"))
    actual_steps = int(getattr(cosimulator, "synchronization_n_step"))
    if actual_steps != synchronization_steps or not math.isclose(
        actual_time,
        float(synchronization_time_ms),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "TVB synchronization time was not fixed to the requested exchange window: "
            f"requested {synchronization_time_ms} ms/{synchronization_steps} steps, "
            f"got {actual_time} ms/{actual_steps} steps"
        )
    return actual_time, actual_steps


def install_neuron_random_poisson_generator(
    *,
    seed: int,
    roi_index: int,
    dt_macro_ms: float,
    dt_micro_ms: float,
    duration_ms: float,
) -> str:
    from neuron import h
    from tvb_multiscale.tvb_netpyne.netpyne import utils as netpyne_utils
    from tvb_multiscale.tvb_netpyne.netpyne_models import devices as netpyne_devices

    seed_low = int(seed) & 0xFFFFFFFF
    seed_high = (int(seed) >> 32) & 0xFFFFFFFF
    stream_cache: dict[int, list] = {}

    def streams_for(num_neurons: int):
        if num_neurons not in stream_cache:
            streams = []
            for neuron_index in range(int(num_neurons)):
                stream = h.Random()
                stream.Random123(int(seed_low), int(neuron_index), int(seed_high ^ int(roi_index)))
                streams.append(stream)
            stream_cache[int(num_neurons)] = streams
        return stream_cache[int(num_neurons)]

    def uniform01(stream) -> float:
        return float(stream.uniform(0.0, 1.0))

    def poisson_count(stream, mean: float) -> int:
        if mean <= 0.0:
            return 0
        limit = math.exp(-float(mean))
        product = 1.0
        count = -1
        while product > limit:
            count += 1
            product *= uniform01(stream)
        return int(count)

    replay_path = macro2micro_replay_path()
    if replay_path is not None:
        if os.environ.get("TVB_NETPYNE_MACRO2MICRO_DIRECT_REPLAY", "0") == "1":
            def generate_empty_spikes_for_population(num_neurons, rates, times):
                return {}

            netpyne_utils.generateSpikesForPopulation = generate_empty_spikes_for_population
            netpyne_devices.generateSpikesForPopulation = generate_empty_spikes_for_population
            return f"direct NetCon.event macro2micro replay from {replay_path}"

        replay_times, replay_sources, replay_steps = load_macro2micro_replay(replay_path)

        def generate_replayed_spikes_for_population(num_neurons, rates, times):
            del rates
            times = np.asarray(times, dtype=float).reshape(-1)
            requested_steps = []
            for sample_time in times:
                sample_step = int(round((float(sample_time) - float(dt_micro_ms)) / float(dt_macro_ms)))
                if sample_step >= 0:
                    requested_steps.append(sample_step)
            if requested_steps:
                requested_steps_array = np.asarray(requested_steps, dtype=int)
                selected = np.isin(replay_steps, requested_steps_array)
            else:
                selected = np.zeros(replay_steps.shape, dtype=bool)
            selected &= replay_sources >= 0
            selected &= replay_sources < int(num_neurons)

            spikes_per_neuron: dict[int, list[float]] = {}
            event_dump_path = _dump_path_from_env("TVB_NETPYNE_MACRO2MICRO_EVENT_DUMP")
            event_handle = None
            if event_dump_path is not None:
                _write_tsv_header_if_needed(
                    event_dump_path,
                    "interval_index\tsample_step\tsample_start_ms\tsample_stop_ms\tevent_time_ms\tsource_id\trate_hz\tmean\tnum_neurons",
                )
                event_handle = event_dump_path.open("a", encoding="utf-8")
            try:
                for event_time, source_id, sample_step in zip(
                    replay_times[selected],
                    replay_sources[selected],
                    replay_steps[selected],
                ):
                    source_id = int(source_id)
                    event_time = float(event_time)
                    sample_step = int(sample_step)
                    spikes_per_neuron.setdefault(source_id, []).append(event_time)
                    if event_handle is not None:
                        sample_start = float(sample_step) * float(dt_macro_ms)
                        sample_stop = min(sample_start + float(dt_macro_ms), float(duration_ms))
                        event_handle.write(
                            f"-1\t{sample_step}\t{sample_start:.17g}\t"
                            f"{sample_stop:.17g}\t{event_time:.17g}\t{source_id}\t"
                            f"\t\t{int(num_neurons)}\n"
                        )
            finally:
                if event_handle is not None:
                    event_handle.close()
            for neuron_spikes in spikes_per_neuron.values():
                neuron_spikes.sort()
            return spikes_per_neuron

        netpyne_utils.generateSpikesForPopulation = generate_replayed_spikes_for_population
        netpyne_devices.generateSpikesForPopulation = generate_replayed_spikes_for_population
        return f"replay macro2micro events from {replay_path}"

    def generate_spikes_for_population(num_neurons, rates, times):
        rates = np.asarray(rates, dtype=float).reshape(-1)
        times = np.asarray(times, dtype=float).reshape(-1)
        interval_count = min(int(rates.size), int(times.size))
        spikes_per_neuron: dict[int, list[float]] = {}
        streams = streams_for(int(num_neurons))
        event_dump_path = _dump_path_from_env("TVB_NETPYNE_MACRO2MICRO_EVENT_DUMP")
        trace_dump_path = _dump_path_from_env("TVB_NETPYNE_MACRO2MICRO_TRACE_DUMP")
        if event_dump_path is not None:
            _write_tsv_header_if_needed(
                event_dump_path,
                "interval_index\tsample_step\tsample_start_ms\tsample_stop_ms\tevent_time_ms\tsource_id\trate_hz\tmean\tnum_neurons",
            )
        if trace_dump_path is not None:
            _write_tsv_header_if_needed(
                trace_dump_path,
                "interval_index\tsample_step\tsample_start_ms\tsample_stop_ms\trate_hz\tmean\tnum_neurons",
            )

        event_handle = event_dump_path.open("a", encoding="utf-8") if event_dump_path is not None else None
        trace_handle = trace_dump_path.open("a", encoding="utf-8") if trace_dump_path is not None else None
        try:
            for interval_index in range(interval_count):
                sample_time = float(times[interval_index])
                exchange_start_step = int(round((sample_time - float(dt_micro_ms)) / float(dt_macro_ms)))
                if exchange_start_step < 0:
                    continue
                interval_start = float(exchange_start_step) * float(dt_macro_ms)
                if interval_start >= float(duration_ms) - SAMPLE_EPS_MS:
                    continue
                interval_stop = min(interval_start + float(dt_macro_ms), float(duration_ms))
                window_ms = interval_stop - interval_start
                rate_hz = max(0.0, float(rates[interval_index]))
                if window_ms <= 0.0 or rate_hz <= 0.0:
                    continue
                mean = rate_hz * window_ms / 1000.0
                if trace_handle is not None:
                    trace_handle.write(
                        f"{interval_index}\t{exchange_start_step}\t{interval_start:.17g}\t"
                        f"{interval_stop:.17g}\t{rate_hz:.17g}\t{mean:.17g}\t{int(num_neurons)}\n"
                    )
                for neuron_index, stream in enumerate(streams):
                    stream.seq(max(0, exchange_start_step))
                    count = poisson_count(stream, mean)
                    if count <= 0:
                        continue
                    neuron_spikes = spikes_per_neuron.setdefault(int(neuron_index), [])
                    for _ in range(count):
                        event_time = interval_start + window_ms * uniform01(stream)
                        neuron_spikes.append(event_time)
                        if event_handle is not None:
                            event_handle.write(
                                f"{interval_index}\t{exchange_start_step}\t{interval_start:.17g}\t"
                                f"{interval_stop:.17g}\t{event_time:.17g}\t{int(neuron_index)}\t"
                                f"{rate_hz:.17g}\t{mean:.17g}\t{int(num_neurons)}\n"
                            )
        finally:
            if event_handle is not None:
                event_handle.close()
            if trace_handle is not None:
                trace_handle.close()
        for neuron_spikes in spikes_per_neuron.values():
            neuron_spikes.sort()
        return spikes_per_neuron

    netpyne_utils.generateSpikesForPopulation = generate_spikes_for_population
    netpyne_devices.generateSpikesForPopulation = generate_spikes_for_population
    return "NEURON h.Random Random123(seed_low, source_id, seed_high^roi) + Knuth Poisson + uniform-in-window event placement"


def install_netpyne_threads(num_threads: int) -> int:
    from neuron import h
    from netpyne import sim as netpyne_sim
    from netpyne.sim import setup as netpyne_setup

    original_create_parallel_context = getattr(
        netpyne_setup.createParallelContext,
        "_mind_original_create_parallel_context",
        netpyne_setup.createParallelContext,
    )

    def create_parallel_context_with_threads():
        original_create_parallel_context()
        netpyne_sim.pc.nthread(int(num_threads))

    create_parallel_context_with_threads._mind_original_create_parallel_context = (
        original_create_parallel_context
    )
    netpyne_setup.createParallelContext = create_parallel_context_with_threads
    netpyne_sim.createParallelContext = create_parallel_context_with_threads
    pc = getattr(netpyne_sim, "pc", None)
    if pc is None:
        pc = h.ParallelContext()
        netpyne_sim.pc = pc
    pc.nthread(int(num_threads))
    return int(pc.nthread())


def main() -> None:
    parser = argparse.ArgumentParser(description="True TVB-multiscale Epileptor2D macro plus NetPyNE CA3 micro reference.")
    parser.add_argument(
        "--connectivity-csv",
        type=Path,
        required=True,
        help="Labelled matrix CSV with weights and delays_ms sections.",
    )
    parser.add_argument("--duration-ms", "--t-ms", dest="duration_ms", type=float, default=20.0)
    parser.add_argument("--macro-i-ext", "--i-ext", dest="macro_i_ext", type=float, default=3.1)
    parser.add_argument("--drive-weight", "--drive-w", dest="drive_weight", type=float, default=0.02e-3)
    parser.add_argument("--conduction-speed-mm-per-ms", type=float, default=5.0)
    parser.add_argument("--netpyne-threads", type=int, default=4)
    parser.add_argument(
        "--micro-backend",
        choices=("coreneuron", "neuron"),
        default="coreneuron",
        help=(
            "NetPyNE execution backend. coreneuron keeps the fast reference path; "
            "neuron runs the same NetPyNE model through NEURON, which supports "
            "interval voltage trace recording."
        ),
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_netpyne_ca3_cosim.npz"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("ca3_epilepsy_cosim/outputs/tvb_netpyne_ca3_workdir"),
    )
    args = parser.parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")

    for path in (
        str(Path(__file__).resolve().parents[3] / "src" / "python_api"),
        str(Path(__file__).resolve().parents[3] / "build"),
    ):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import mind_sim as ms

    add_tvb_multiscale_paths(Path.home() / "tvb-root", Path.home() / "tvb-multiscale")
    args.workdir.mkdir(parents=True, exist_ok=True)
    coreneuron_enabled = args.micro_backend == "coreneuron"
    install_tvb_netpyne_nominal_runtime_hooks()
    if coreneuron_enabled:
        mod_library_dir = prepare_coreneuron_mod_library(args.workdir)
        coreneuron_mod_library = mod_library_dir
        input_event_source = install_coreneuron_vecstim_tvb_netpyne_input()
    else:
        mod_library_dir = Path(__file__).resolve().parents[1] / "mind_sim" / "mod"
        coreneuron_mod_library = None
        if macro2micro_replay_path() is not None:
            os.environ["TVB_NETPYNE_MACRO2MICRO_DIRECT_REPLAY"] = "1"
            input_event_source = install_macro2micro_replay_netcon_input()
        else:
            input_event_source = "DynamicVecStim"
    dt_macro_ms = 0.1
    exchange_window_ms = 0.5
    rois = ms.macro.load_rois(args.connectivity_csv)
    labels = list(rois.labels)
    weights = np.asarray(rois.weights, dtype=float)
    delays = np.asarray(rois.delays, dtype=float)
    ca3_label = "Left-CA3"
    if ca3_label not in labels:
        raise SystemExit("connectivity labels do not contain ROI 'Left-CA3'")
    ca3_index = labels.index(ca3_label)
    min_positive_delay = float(np.min(delays[delays > 0.0])) if np.any(delays > 0.0) else 0.0
    if min_positive_delay <= 0.0:
        raise SystemExit("connectivity delays must contain at least one positive delay")
    exchange_steps_float = exchange_window_ms / dt_macro_ms
    exchange_steps = int(round(exchange_steps_float))
    if exchange_steps < 1 or not math.isclose(exchange_steps_float, exchange_steps, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("exchange window must be an integer multiple of dt_macro")
    if exchange_window_ms > min_positive_delay + 1e-9:
        raise SystemExit("exchange window must not exceed the minimum positive connectivity delay")

    pre_start = time.perf_counter()
    netpyne_num_threads = install_netpyne_threads(int(args.netpyne_threads))
    record_voltage = int(netpyne_num_threads) == 1
    macro2micro_random_stream = install_neuron_random_poisson_generator(
        seed=1,
        roi_index=ca3_index,
        dt_macro_ms=dt_macro_ms,
        dt_micro_ms=0.025,
        duration_ms=float(args.duration_ms),
    )
    orchestrator = configure_tvb_multiscale(
        labels=labels,
        weights=weights,
        delays=delays,
        ca3_index=ca3_index,
        duration_ms=float(args.duration_ms),
        macro_i_ext=float(args.macro_i_ext),
        drive_weight=float(args.drive_weight),
        pyr_current_na=PYR_CURRENT_NA,
        olm_current_na=OLM_CURRENT_NA,
        conduction_speed_mm_per_ms=float(args.conduction_speed_mm_per_ms),
        output_base=args.workdir,
        exchange_window_ms=exchange_window_ms,
        mod_library_dir=mod_library_dir,
        record_voltage=record_voltage,
        coreneuron_enabled=coreneuron_enabled,
    )
    orchestrator.start()
    orchestrator.configure()
    orchestrator.build()
    connection_dump_path = _dump_path_from_env("TVB_NETPYNE_CONNECTION_DUMP")
    if connection_dump_path is not None:
        dump_netpyne_connections_from_sim(connection_dump_path)
    actual_coupling_a = force_tvb_coupling_scale(orchestrator, 1.0)
    requested_synchronization_time, requested_synchronization_steps = force_tvb_synchronization_time(
        orchestrator,
        exchange_window_ms,
        dt_macro_ms,
    )
    pre_run_s = time.perf_counter() - pre_start

    run_start = time.perf_counter()
    orchestrator.simulate(float(args.duration_ms))
    run_s = time.perf_counter() - run_start
    actual_synchronization_time = float(getattr(orchestrator.tvb_app.cosimulator, "synchronization_time", np.nan))
    actual_synchronization_steps = int(getattr(orchestrator.tvb_app.cosimulator, "synchronization_n_step", -1))

    metadata = {
        "source": "TVB-multiscale TVB macro API plus NetPyNE/NEURON ModelDB 186768 CA3",
        "macro_backend": "tvb_multiscale CoSimulatorNetpyneBuilder + TVB built-in Epileptor2D",
        "micro_backend": (
            "tvb_multiscale NetPyNE + CoreNEURON"
            if coreneuron_enabled
            else "tvb_multiscale NetPyNE + NEURON"
        ),
        "interface": "TVBNetpyneSerialOrchestrator, TVB computed CA3 incoming coupling input to PYR poisson proxy, PYR+BAS+OLM spikes to Epileptor x1",
        "coreneuron_enabled": bool(coreneuron_enabled),
        "input_event_source": input_event_source,
        "coreneuron_mod_library": str(coreneuron_mod_library) if coreneuron_mod_library is not None else None,
        "netpyne_modeldb_mod_library": str(mod_library_dir),
        "macro2micro_random_stream": macro2micro_random_stream,
        "stimulus_mapping": "CA3 external proxy is driven by TVB computed incoming coupling input, then uses explicit source_i_to_PYR_i connList at Adend3(0.5)",
        "connectivity_csv": str(args.connectivity_csv),
        "duration_ms": float(args.duration_ms),
        "dt_macro_ms": float(dt_macro_ms),
        "dt_micro_ms": 0.025,
        "netpyne_num_threads": int(netpyne_num_threads),
        "initial_history_steps": int(tvb_history_steps(weights, delays, dt_macro_ms)),
        "min_positive_delay_ms": float(min_positive_delay),
        "exchange_window_ms": float(exchange_window_ms),
        "exchange_window_steps": int(exchange_steps),
        "tvb_requested_synchronization_time_ms": float(requested_synchronization_time),
        "tvb_requested_synchronization_steps": int(requested_synchronization_steps),
        "tvb_actual_synchronization_time_ms": float(actual_synchronization_time),
        "tvb_actual_synchronization_steps": int(actual_synchronization_steps),
        "ca3_label": ca3_label,
        "ca3_index": int(ca3_index),
        "macro_i_ext": float(args.macro_i_ext),
        "tvb_coupling_a": float(actual_coupling_a),
        "epileptor2d_kvf": 0.35,
        "epileptor2d_ks": 0.0,
        "epileptor2d_r": 0.00035,
        "epileptor2d_tt": 1.0,
        "epileptor2d_modification": False,
        "drive_weight": float(args.drive_weight),
        "drive_delay_ms": 0.2,
        "pyr_current_na": PYR_CURRENT_NA,
        "olm_current_na": OLM_CURRENT_NA,
        "conduction_speed_mm_per_ms": float(args.conduction_speed_mm_per_ms),
        "micro2macro_spike_source": (
            os.environ.get("TVB_NETPYNE_MICRO2MACRO_SPIKE_SOURCE", "").strip().lower()
            or ("replay" if micro2macro_replay_path() is not None else "direct")
        ),
        "micro2macro_replay": str(micro2macro_replay_path()) if micro2macro_replay_path() is not None else None,
        "voltage_recording": (
            "representative PYR/BAS/OLM soma voltages in voltage_traces; fixed output key voltage remains PYR[0].soma"
            if record_voltage
            else "disabled for multi-thread reference run"
        ),
        "voltage_trace_labels": [str(spec["label"]) for spec in VOLTAGE_TRACE_SPECS] if record_voltage else [],
        "spike_validation": "derive representative PYR/BAS/OLM soma spikes from recorded voltage threshold crossings; API spike arrays are not exported",
        "recorded_exposures": ["x", "z"],
        "pre_run_s": float(pre_run_s),
        "run_s": float(run_s),
    }
    try:
        save_outputs(
            args.output,
            orchestrator,
            labels,
            weights,
            delays,
            metadata,
            ca3_index,
            exchange_window_ms,
        )
    finally:
        try:
            orchestrator.clean_up()
        except TypeError:
            # NetPyNE 1.1.1 can fail in postRun timing bookkeeping after
            # interval runs; traces and spikes are already collected above.
            pass
        orchestrator.stop()

    print(f"output={args.output}")
    print("backend=tvb_multiscale_tvb_netpyne")
    print(f"micro_backend={args.micro_backend}")
    print(f"netpyne_num_threads={netpyne_num_threads}")
    print(f"pre_run_s={pre_run_s:.6f}")
    print(f"run_s={run_s:.6f}")


if __name__ == "__main__":
    main()
