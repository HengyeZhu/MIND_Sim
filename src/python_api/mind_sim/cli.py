from __future__ import annotations

import ctypes.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    print("MIND_Sim native package is installed. Use `import mind_sim` from Python.")
    return 0


def _existing_file(path: Path | None) -> str | None:
    if path is not None and path.is_file():
        return str(path)
    return None


def _find_libpython() -> str | None:
    try:
        from find_libpython import find_libpython
    except ImportError:
        find_libpython = None
    if find_libpython is not None:
        located = find_libpython()
        if located:
            found = _existing_file(Path(located))
            if found is not None and not found.endswith(".a"):
                return found

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    names = [
        sysconfig.get_config_var("LDLIBRARY"),
        sysconfig.get_config_var("INSTSONAME"),
        f"libpython{version}.so",
        f"libpython{version}.so.1.0",
        f"libpython{version}.dylib",
    ]
    roots = [
        sysconfig.get_config_var("LIBDIR"),
        sysconfig.get_config_var("LIBPL"),
        str(Path(sys.base_prefix) / "lib"),
        str(Path(sys.exec_prefix) / "lib"),
    ]
    for root in dict.fromkeys(path for path in roots if path):
        root_path = Path(root)
        for name in dict.fromkeys(name for name in names if name):
            found = _existing_file(root_path / name)
            if found is not None and not found.endswith(".a"):
                return found

    located = ctypes.util.find_library(f"python{version}")
    if located:
        located_path = Path(located)
        if located_path.is_file():
            return str(located_path)
        if not located_path.is_absolute() and located_path.name.startswith("libpython"):
            return located
    return None


def _prepend_path(env: dict[str, str], name: str, path: Path) -> None:
    value = str(path)
    current = env.get(name)
    env[name] = value if not current else f"{value}{os.pathsep}{current}"


def _prepend_python_paths(env: dict[str, str]) -> None:
    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key)
        if path:
            _prepend_path(env, "PYTHONPATH", Path(path))


def mind_nrnivmodl_main() -> int:
    package_dir = Path(__file__).resolve().parent
    data_dir = package_dir / ".data"
    exe = data_dir / "bin" / "mind-nrnivmodl"
    if not exe.exists():
        raise SystemExit(f"mind-nrnivmodl executable was not installed: {exe}")
    env = os.environ.copy()
    env.setdefault("NMODLHOME", str(data_dir))
    if "NMODL_PYLIB" not in env:
        libpython = _find_libpython()
        if libpython is not None:
            env["NMODL_PYLIB"] = libpython
            if os.path.isabs(libpython):
                _prepend_path(env, "LD_LIBRARY_PATH", Path(libpython).parent)
    _prepend_python_paths(env)
    _prepend_path(env, "LD_LIBRARY_PATH", data_dir / "lib")
    return subprocess.call([str(exe), *sys.argv[1:]], env=env)
