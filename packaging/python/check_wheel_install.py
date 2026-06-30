from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import mind_sim
import mind_sim._native as native


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


package_dir = Path(mind_sim.__file__).resolve().parent
data_dir = package_dir / ".data"
script_dir = Path(sys.executable).parent


def console_script(name: str) -> str | None:
    found = shutil.which(name)
    if found is not None:
        return found
    candidate = script_dir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None

require(hasattr(mind_sim, "load_mech"), "mind_sim.load_mech is missing")
require(not hasattr(mind_sim, "Sim"), "legacy mind_sim.Sim entry point is still exposed")
require(isinstance(mind_sim.micro.sim(), native.Sim), "mind_sim.micro.sim() is not using the installed native extension")
mind_simulator = console_script("mind-simulator")
mind_nrnivmodl = console_script("mind-nrnivmodl")
require(mind_simulator is not None, "mind-simulator console script was not installed")
require(mind_nrnivmodl is not None, "mind-nrnivmodl console script was not installed")

for binary in (data_dir / "bin" / "nmodl", data_dir / "bin" / "mind-nrnivmodl"):
    require(binary.is_file(), f"wheel binary is missing: {binary}")
    require(os.access(binary, os.X_OK), f"wheel binary is not executable: {binary}")

subprocess.run([mind_simulator], check=True)
