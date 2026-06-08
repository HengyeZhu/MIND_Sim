from __future__ import annotations

from collections.abc import Mapping

from . import _native


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

    def dc_input(self, values: Mapping[str, float]) -> "ROI":
        self._network._builder.set_dc_input(self._index, _values(values))
        return self

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

    def use_micro(self) -> "ROI":
        self._network._builder.use_micro(self._index)
        return self

    def __repr__(self) -> str:
        return f"<mind_sim.ROI name={self.label!r}>"


class NeuralField:
    def __init__(
        self,
        name: str,
        rule=None,
        *,
        local=None,
        initial_state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ):
        self.name = str(name)
        self.rule = None
        self.local_data = local
        self.initial_state = initial_state
        self.params = params
        if rule is not None:
            self.use(rule, initial_state=initial_state, params=params)

    def use(
        self,
        rule,
        *,
        initial_state: Mapping[str, float] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "NeuralField":
        self.rule = str(rule)
        self.initial_state = initial_state
        self.params = params
        return self

    def local(self, local) -> "NeuralField":
        self.local_data = local
        return self


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

    def use_neural_field(self, field: NeuralField, *, node_map) -> "Network":
        if field.local_data is None:
            self._builder.use_neural_field(
                field.name,
                str(field.rule),
                node_map,
                _values(field.initial_state),
                _values(field.params),
            )
        else:
            self._builder.use_neural_field(
                field.name,
                str(field.rule),
                node_map,
                field.local_data,
                _values(field.initial_state),
                _values(field.params),
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

    def run(self, t_stop: float):
        return self._native.run(float(t_stop))


Connectivity = _native.Connectivity
LocalConnectivity = _native.LocalConnectivity
NodeToRoiMap = _native.NodeToRoiMap
