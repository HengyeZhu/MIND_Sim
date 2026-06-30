from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from decimal import Decimal

from . import _native


def _values(values: Mapping[str, float] | None) -> dict[str, float]:
    return {str(name): float(value) for name, value in dict(values or {}).items()}


def _output_names(outputs) -> list[str]:
    if outputs is None:
        return []
    if isinstance(outputs, str):
        return [outputs]
    return [str(name) for name in outputs]


def _apply_macro_config(builder) -> None:
    _native._macro_config.apply(builder)


class ROI:
    def __init__(self, network: "Network", index: int, label: str):
        self._network = network
        self._index = int(index)
        self.label = str(label)

    def record(self, output: str, /) -> "ROI":
        self._network._builder.record(self._index, output)
        return self

    def use_macro(
        self,
        mechanism,
        *,
        initial_state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "ROI":
        self._network._builder.use_region(
            self._index,
            str(mechanism),
            _values(initial_state),
            _values(params),
        )
        return self

    def insert(self, source, mechanism, *, params: Mapping[str, float] | None = None) -> "ROI":
        source_index = self._network._roi_index(source)
        self._network._builder.macro2macro(
            source_index,
            self._index,
            str(mechanism),
            _values(params),
        )
        return self

    def macro2micro(
        self,
        mechanism,
        *,
        target,
        weight: float,
        delay: float,
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "ROI":
        # target is the concrete micro point process that receives macro events.
        self._network._builder.macro2micro(
            self._index,
            str(mechanism),
            target,
            float(weight),
            float(delay),
            _values(state),
            _values(params),
        )
        return self

    def micro2macro(
        self,
        mechanism,
        sid: int,
        *,
        state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "ROI":
        # sid is the explicit spike-source id registered by
        # micro.network().register_spike_source(...). It may equal cell.gid when
        # each cell has one spike source, but it is not inferred from the cell.
        self._network._builder.micro2macro(
            self._index,
            str(mechanism),
            int(sid),
            _values(state),
            _values(params),
        )
        return self

    def use_micro(self, micro, *, exposures: Iterable[str]) -> "ROI":
        self._network._builder.use_micro(self._index, micro, [str(name) for name in exposures])
        return self

    def initial_history(self, history, *, outputs, layout: str = "time_output") -> "ROI":
        self._network._builder.set_roi_initial_history(
            self._index,
            history,
            _output_names(outputs),
            str(layout),
        )
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.ROI name={self.label!r}>"


class Network:
    def __init__(
        self,
        *,
        connectivity=None,
        labels=None,
        weights=None,
        delays=None,
    ):
        if connectivity is None:
            connectivity = _native.Connectivity(list(labels), weights, delays)
        self._connectivity = connectivity
        self._builder = _native._NetworkBuilder(connectivity)
        _apply_macro_config(self._builder)
        self._rois = [ROI(self, roi.index, roi.label) for roi in self._builder.rois()]
        self._roi_by_label = {roi.label: roi for roi in self._rois}

    @property
    def labels(self) -> list[str]:
        return list(self._connectivity.labels)

    @property
    def weights(self) -> list[list[float]]:
        return self._connectivity.weights

    @property
    def delays(self) -> list[list[float]]:
        return self._connectivity.delays

    def roi(self, name: str) -> ROI:
        label = str(name)
        try:
            return self._roi_by_label[label]
        except KeyError as exc:
            raise KeyError(f"unknown ROI label: {label}") from exc

    def _roi_index(self, roi) -> int:
        if isinstance(roi, ROI):
            if roi._network is not self:
                raise ValueError("ROI belongs to a different Network")
            return roi._index
        return self.roi(str(roi))._index

    def __iter__(self) -> Iterator[ROI]:
        return iter(self._rois)

    def __len__(self) -> int:
        return len(self._rois)

    def roi_count(self) -> int:
        return int(self._builder.roi_count())

    def min_positive_delay(self) -> float:
        return float(self._builder.min_positive_delay())

    def _build_native(self):
        return self._builder.build()


class Simulator:
    def __init__(self, rois: Network, macro2micro_seed: int = 1):
        self._native = _native.Simulator(rois._build_native(), macro2micro_seed)

    def run(self, t_stop: float):
        return self._native.run(float(t_stop))


class MacroSimulator:
    def __init__(self, rois: Network):
        self._native = _native.MacroRuntime(rois._build_native())

    def run(self, t_stop: float | None = None, *, n_steps: int | None = None):
        if n_steps is not None:
            if t_stop is not None:
                raise TypeError("run accepts either t_stop or n_steps, not both")
            if not isinstance(n_steps, int) or isinstance(n_steps, bool):
                raise TypeError("n_steps must be an integer")
            t_stop = float(Decimal(str(n_steps)) * Decimal(str(self._native.dt())))
        if t_stop is None:
            raise TypeError("run requires t_stop or n_steps")
        return self._native.run(float(t_stop))


Connectivity = _native.Connectivity
