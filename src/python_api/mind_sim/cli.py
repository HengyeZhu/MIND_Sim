from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    print("MIND_Sim native package is installed. Use `import mind_sim` from Python.")
    return 0


def mind_nrnivmodl_main() -> int:
    exe = Path(__file__).resolve().parent / "bin" / "mind_nrnivmodl"
    if not exe.exists():
        raise SystemExit(f"mind_nrnivmodl executable was not installed: {exe}")
    return subprocess.call([str(exe), *sys.argv[1:]])
