from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .. import _native
from .._network import Network, _set_macro_dt


def load_rois(
    path: str | Path | None = None,
    *,
    labels: Sequence[str] | None = None,
    weights=None,
    delays=None,
) -> Network:
    if path is not None:
        return Network(connectivity=_native.Connectivity.from_csv(str(Path(path))))
    if labels is None or weights is None or delays is None:
        raise TypeError("load_rois requires either a path or labels, weights, and delays")
    return Network(labels=list(labels), weights=weights, delays=delays)


def load_mech(directory: str | Path) -> None:
    _native._macro_config.load_mech(str(directory))


def dt(value: float) -> None:
    dt_value = float(value)
    _native._macro_config.dt(dt_value)
    _set_macro_dt(dt_value)


def exchange_window(value: float) -> None:
    _native._macro_config.exchange_window(float(value))
