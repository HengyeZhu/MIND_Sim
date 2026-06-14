from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal

from . import _native


_macro_dt: float | None = None


def _set_macro_dt(value: float) -> None:
    global _macro_dt
    _macro_dt = value


def _values(values: Mapping[str, float] | None) -> dict[str, float]:
    return {str(name): float(value) for name, value in dict(values or {}).items()}


def _apply_macro_config(builder) -> None:
    _native._macro_config.apply(builder)


class ROI:
    def __init__(self, network: "Network", index: int, label: str):
        self._network = network
        self._index = int(index)
        self.label = str(label)
        self.name = self.label

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
        source_roi = self._network._builder.roi(str(source))
        self._network._builder.macro2macro(
            int(source_roi.index),
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

    def use_micro(self, *, exposures: Iterable[str]) -> "ROI":
        self._network._builder.use_micro(self._index, [str(name) for name in exposures])
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
        roi = self._builder.roi(str(name))
        return self._roi_from_native(roi)

    def _roi_from_native(self, roi) -> ROI:
        return self._rois[int(roi.index)]

    def rois(self) -> list[ROI]:
        return list(self._rois)

    def roi_count(self) -> int:
        return int(self._builder.roi_count())

    def min_positive_delay(self) -> float:
        return float(self._builder.min_positive_delay())

    def initial_history(self, history, *, outputs=None, layout: str = "time_output_roi") -> "Network":
        self._builder.set_initial_history(
            history,
            list(outputs) if outputs is not None else [],
            str(layout),
        )
        return self

    def _build_native(self):
        return self._builder.build()


class Simulator:
    def __init__(self, rois: Network):
        self._native = _native.Simulator(rois._build_native())

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
            if _macro_dt is None:
                raise RuntimeError("run(n_steps=...) requires ms.macro.dt(...)")
            t_stop = float(Decimal(str(n_steps)) * Decimal(str(_macro_dt)))
        if t_stop is None:
            raise TypeError("run requires t_stop or n_steps")
        return self._native.run(float(t_stop))


Connectivity = _native.Connectivity
