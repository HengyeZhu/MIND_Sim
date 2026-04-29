#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def nmodl_environment(nmodl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NMODLHOME"] = str(nmodl.resolve().parent.parent)
    return env


def collect_mods(paths: list[Path]) -> list[Path]:
    mods: list[Path] = []
    for path in paths:
        if path.is_dir():
            mods.extend(sorted(path.glob("*.mod")))
        else:
            mods.append(path)
    return sorted({mod.resolve() for mod in mods})


def write_modl_reg(path: Path, mod_names: list[str], function_name: str) -> None:
    lines = [
        "#include <cstdio>",
        "namespace coreneuron {",
        "extern int nrnmpi_myid;",
        "extern int nrn_nobanner_;",
    ]
    for name in mod_names:
        lines.append(f"extern void _{name}_reg();")
    lines.extend(
        [
            f"void {function_name}() {{",
            "    if (!nrn_nobanner_ && nrnmpi_myid < 1) {",
            '        std::fprintf(stderr, " Additional MIND_Sim mechanisms from MOD files\\n");',
        ]
    )
    for name in mod_names:
        lines.append(f'        std::fprintf(stderr, " {name}.mod");')
    lines.extend(['        std::fprintf(stderr, "\\n\\n");', "    }"])
    for name in mod_names:
        lines.append(f"    _{name}_reg();")
    lines.extend(["}", "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_object(
    cxx: str,
    source: Path,
    output: Path,
    includes: list[str],
    defines: list[str],
    cxx_flags: list[str],
) -> None:
    command = [cxx, "-std=c++17", "-O2", "-fPIC"]
    for flag in cxx_flags:
        command.extend(shlex.split(flag))
    command.extend(f"-I{include}" for include in includes)
    command.extend(f"-D{define}" for define in defines)
    command.extend(["-c", str(source), "-o", str(output)])
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mods", nargs="+", type=Path)
    parser.add_argument("--nmodl", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--modl-reg-name", default="modl_reg")
    parser.add_argument("--cxx-flag", action="append", default=[])
    parser.add_argument("--link-flag", action="append", default=[])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()

    mods = collect_mods(args.mods)
    if not mods:
        raise RuntimeError("no MOD files supplied")

    output = args.output.resolve()
    core_dir = output / "core"
    obj_dir = output / "obj"
    scratch_dir = output / "scratch"
    for directory in (core_dir, obj_dir, scratch_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    nmodl_env = nmodl_environment(Path(args.nmodl))
    run([args.nmodl,
         *map(str, mods),
         "-o",
         str(core_dir),
         "--scratch",
         str(scratch_dir),
         "host",
         "--c",
         "passes",
         "--inline"],
        env=nmodl_env)

    mod_names = [mod.stem for mod in mods]
    modl_reg_source = (
        "mind_modl_reg.cpp"
        if args.modl_reg_name == "modl_reg"
        else f"{args.modl_reg_name}.cpp"
    )
    write_modl_reg(core_dir / modl_reg_source, mod_names, args.modl_reg_name)

    if args.generate_only:
        return 0

    objects: list[Path] = []
    for source in sorted(core_dir.glob("*.cpp")):
        obj = obj_dir / f"{source.stem}.o"
        compile_object(args.cxx, source, obj, args.include, args.define, args.cxx_flag)
        objects.append(obj)

    library = output / "libmindcorenrnmech.so"
    link_flags: list[str] = []
    for flag in args.link_flag:
        link_flags.extend(shlex.split(flag))
    link_command = [args.cxx, "-shared", "-fPIC", "-o", str(library), *map(str, objects), *link_flags]
    if os.uname().sysname == "Linux":
        link_command.append("-Wl,--allow-shlib-undefined")
    run(link_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
