from __future__ import annotations

from pathlib import Path

from .. import _native
from .._network import Network


def load_rois(path: str | Path) -> Network:
    return Network(connectivity=_native.Connectivity.from_csv(str(Path(path))))


def load_mech(directory: str | Path) -> None:
    _native._macro_config.load_mech(str(directory))


def dt(value: float) -> None:
    _native._macro_config.dt(float(value))


def exchange_window(value: float) -> None:
    _native._macro_config.exchange_window(float(value))
