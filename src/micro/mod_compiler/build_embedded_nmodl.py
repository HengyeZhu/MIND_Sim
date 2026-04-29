#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from hashlib import sha1
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_module(name: str) -> None:
    __import__(name)


def write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def copy_if_changed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.read_bytes() == target.read_bytes():
        return
    shutil.copy2(source, target)


def embedded_inputs(source_dir: Path) -> list[Path]:
    roots = [
        source_dir / "src" / "nmodl",
        source_dir / "share" / "lib" / "python" / "neuron" / "nmodl",
        source_dir / "share" / "nmodl",
        source_dir / "external" / "CLI11" / "include",
        source_dir / "external" / "Random123" / "include",
        source_dir / "external" / "eigen" / "Eigen",
        source_dir / "external" / "fmt" / "include",
        source_dir / "external" / "json" / "include",
        source_dir / "external" / "pybind11" / "include",
        source_dir / "external" / "spdlog" / "include",
    ]
    files = [Path(__file__).resolve()]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def build_signature(args: argparse.Namespace) -> str:
    digest = sha1()
    for value in [
        str(args.cxx.resolve()),
        str(args.bison.resolve()),
        str(args.flex.resolve()),
        str(args.flex_include.resolve()) if args.flex_include else "",
        args.build_type,
        sys.version,
    ]:
        digest.update(value.encode())
        digest.update(b"\0")
    source_dir = args.source_dir.resolve()
    for path in embedded_inputs(source_dir):
        digest.update(str(path.relative_to(source_dir) if path.is_relative_to(source_dir) else path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def strip_header_body(path: Path) -> str:
    text = path.read_text()
    return re.sub(r".*#pragma once[ \t\r\n]*", "", text, flags=re.DOTALL)


def render_support_files(source_dir: Path, build_dir: Path) -> None:
    src_dir = source_dir / "src" / "nmodl"
    gen_dir = build_dir / "src" / "nmodl"

    config = (src_dir / "config" / "config.cpp.in").read_text()
    config = config.replace("@NMODL_GIT_REVISION@", "bundled")
    config = config.replace("@PROJECT_VERSION@", "9.0.1")
    config = config.replace("@CMAKE_SHARED_LIBRARY_SUFFIX@", ".so")
    config = config.replace("@CMAKE_INSTALL_PREFIX@", str(build_dir))
    config = config.replace("@NMODL_PROJECT_PLATLIB_BINARY_DIR@", str(build_dir))
    write_text_if_changed(gen_dir / "config" / "config.cpp", config)

    crout_hpp = strip_header_body(src_dir / "solver" / "crout" / "crout.hpp")
    newton_hpp = strip_header_body(src_dir / "solver" / "newton" / "newton.hpp")
    newton_hpp = newton_hpp.replace('#include <crout/crout.hpp>\n', "")
    solver = (src_dir / "solver" / "solver.hpp.inc").read_text()
    solver = solver.replace("@NMODL_CROUT_HPP@", crout_hpp)
    solver = solver.replace("@NMODL_NEWTON_HPP@", newton_hpp)
    write_text_if_changed(gen_dir / "solver" / "solver.hpp", solver)

    ode_py = (source_dir / "share" / "lib" / "python" / "neuron" / "nmodl" / "ode.py").read_text()
    ode = (src_dir / "pybind" / "ode_py.hpp.inc").read_text()
    ode = ode.replace("@NMODL_ODE_PY@", ode_py)
    write_text_if_changed(gen_dir / "pybind" / "ode_py.hpp", ode)

    copy_if_changed(
        source_dir / "share" / "nmodl" / "nrnunits.lib",
        build_dir / "share" / "nmodl" / "nrnunits.lib",
    )


def generate_ast(source_dir: Path, build_dir: Path, python: Path) -> None:
    src_dir = source_dir / "src" / "nmodl"
    run(
        [
            str(python),
            str(src_dir / "language" / "code_generator.py"),
            "--disable-pybind",
            "--base-dir",
            str(build_dir / "src" / "nmodl"),
        ],
        cwd=src_dir / "language",
    )


def generate_parser_and_lexer(source_dir: Path, build_dir: Path, bison: Path, flex: Path) -> None:
    src_dir = source_dir / "src" / "nmodl"
    gen_dir = build_dir / "src" / "nmodl"
    parser_dir = gen_dir / "parser"
    lexer_dir = gen_dir / "lexer"
    for path in [
        parser_dir / "c",
        parser_dir / "diffeq",
        parser_dir / "nmodl",
        parser_dir / "unit",
        lexer_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    run([str(bison), "-d", "-o", "nmodl/nmodl_parser.cpp", str(src_dir / "parser" / "nmodl.yy")], cwd=parser_dir)
    run([str(bison), "-d", "-o", "verbatim_parser.cpp", str(src_dir / "parser" / "verbatim.yy")], cwd=parser_dir)
    run([str(bison), "-d", "-o", "diffeq/diffeq_parser.cpp", str(src_dir / "parser" / "diffeq.yy")], cwd=parser_dir)
    run([str(bison), "-d", "-o", "c/c11_parser.cpp", str(src_dir / "parser" / "c11.yy")], cwd=parser_dir)
    run([str(bison), "-d", "-o", "unit/unit_parser.cpp", str(src_dir / "parser" / "unit.yy")], cwd=parser_dir)

    run([str(flex), str(src_dir / "lexer" / "nmodl.ll")], cwd=lexer_dir)
    run([str(flex), str(src_dir / "lexer" / "verbatim.l")], cwd=lexer_dir)
    run([str(flex), str(src_dir / "lexer" / "diffeq.ll")], cwd=lexer_dir)
    run([str(flex), str(src_dir / "lexer" / "c11.ll")], cwd=lexer_dir)
    run([str(flex), str(src_dir / "lexer" / "unit.ll")], cwd=lexer_dir)


def python_link_flags() -> list[str]:
    libdir = Path(sysconfig.get_config_var("LIBDIR") or sys.prefix)
    version = sysconfig.get_config_var("VERSION") or f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        libdir / f"libpython{version}.so",
        libdir / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so",
        libdir / f"libpython{sys.version_info.major}.so",
        libdir / (sysconfig.get_config_var("LDLIBRARY") or ""),
    ]
    flags: list[str] = []
    libpython = next((path for path in candidates if path.name and path.exists()), None)
    if libpython:
        flags.append(str(libpython))
    else:
        flags.append(f"-lpython{version}")
    flags.extend(["-Wl,-rpath," + str(libdir), "-ldl", "-lpthread", "-lm", "-lutil"])
    return flags


def source_files(source_dir: Path, build_dir: Path) -> list[Path]:
    src_dir = source_dir / "src" / "nmodl"
    gen_dir = build_dir / "src" / "nmodl"
    files = [
        src_dir / "main.cpp",
        src_dir / "lexer" / "token_mapping.cpp",
        src_dir / "lexer" / "nmodl_utils.cpp",
        src_dir / "lexer" / "modtoken.cpp",
        src_dir / "parser" / "nmodl_driver.cpp",
        src_dir / "parser" / "diffeq_driver.cpp",
        src_dir / "parser" / "diffeq_context.cpp",
        src_dir / "parser" / "c11_driver.cpp",
        src_dir / "parser" / "unit_driver.cpp",
        src_dir / "units" / "units.cpp",
        src_dir / "pybind" / "pyembed.cpp",
        src_dir / "pybind" / "wrapper.cpp",
        gen_dir / "config" / "config.cpp",
        gen_dir / "parser" / "nmodl" / "nmodl_parser.cpp",
        gen_dir / "parser" / "verbatim_parser.cpp",
        gen_dir / "parser" / "diffeq" / "diffeq_parser.cpp",
        gen_dir / "parser" / "c" / "c11_parser.cpp",
        gen_dir / "parser" / "unit" / "unit_parser.cpp",
        gen_dir / "lexer" / "nmodl_base_lexer.cpp",
        gen_dir / "lexer" / "verbatim_lexer.cpp",
        gen_dir / "lexer" / "diffeq_base_lexer.cpp",
        gen_dir / "lexer" / "c11_base_lexer.cpp",
        gen_dir / "lexer" / "unit_base_lexer.cpp",
    ]
    for directory in ["codegen", "printer", "symtab", "utils"]:
        files.extend(sorted((src_dir / directory).glob("*.cpp")))
    files.extend(path for path in sorted((src_dir / "visitors").glob("*.cpp")) if path.name != "main.cpp")
    files.extend(sorted((gen_dir / "ast").glob("*.cpp")))
    files.extend(sorted((gen_dir / "visitors").glob("*.cpp")))
    return files


def compile_nmodl(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    src_dir = source_dir / "src" / "nmodl"
    gen_dir = build_dir / "src" / "nmodl"
    obj_dir = build_dir / "obj"
    bin_path = build_dir / "bin" / "nmodl"

    shutil.rmtree(obj_dir, ignore_errors=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    opt_flag = "-O3" if args.build_type.lower() in {"release", "relwithdebinfo", "minsizerel"} else "-O0"
    include_dirs = [
        src_dir,
        gen_dir,
        gen_dir / "pybind",
        source_dir / "external" / "CLI11" / "include",
        source_dir / "external" / "Random123" / "include",
        source_dir / "external" / "eigen",
        source_dir / "external" / "fmt" / "include",
        source_dir / "external" / "json" / "include",
        source_dir / "external" / "pybind11" / "include",
        source_dir / "external" / "spdlog" / "include",
        Path(sysconfig.get_path("include")),
    ]
    if args.flex_include:
        include_dirs.append(args.flex_include.resolve())

    common_flags = [
        "-std=c++17",
        "-fPIC",
        opt_flag,
        "-DFMT_HEADER_ONLY",
        "-DSPDLOG_FMT_EXTERNAL",
        "-DNMODL_STATIC_PYWRAPPER=1",
    ]
    for include_dir in include_dirs:
        common_flags.extend(["-I", str(include_dir)])

    objects: list[Path] = []
    compile_commands: list[list[str]] = []
    for source in source_files(source_dir, build_dir):
        digest = sha1(str(source).encode()).hexdigest()[:12]
        obj = obj_dir / f"{source.stem}_{digest}.o"
        objects.append(obj)
        compile_commands.append([str(args.cxx), *common_flags, "-c", str(source), "-o", str(obj)])

    def compile_one(command: list[str]) -> None:
        run(command)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in concurrent.futures.as_completed(pool.submit(compile_one, command) for command in compile_commands):
            future.result()

    response_file = build_dir / "objects.rsp"
    response_file.write_text("\n".join(str(obj) for obj in objects))
    run([str(args.cxx), "@" + str(response_file), "-o", str(bin_path), *python_link_flags()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cxx", type=Path, required=True)
    parser.add_argument("--bison", type=Path)
    parser.add_argument("--flex", type=Path)
    parser.add_argument("--flex-include", type=Path)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bison is None:
        bison = shutil.which("bison")
        if bison is None:
            raise RuntimeError("bison is required to build embedded NMODL")
        args.bison = Path(bison)
    if args.flex is None:
        flex = shutil.which("flex")
        if flex is None:
            raise RuntimeError("flex is required to build embedded NMODL")
        args.flex = Path(flex)
    if not args.flex_include:
        for candidate in [Path(sys.prefix) / "include", args.flex.resolve().parent.parent / "include", Path("/usr/include")]:
            if (candidate / "FlexLexer.h").exists():
                args.flex_include = candidate
                break
    for module in ["jinja2", "sympy", "yaml"]:
        require_module(module)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    signature = build_signature(args)
    stamp = args.build_dir / ".embedded_nmodl.signature"
    output = args.build_dir / "bin" / "nmodl"
    if output.exists() and stamp.exists() and stamp.read_text() == signature:
        return
    generate_ast(args.source_dir, args.build_dir, args.python)
    render_support_files(args.source_dir, args.build_dir)
    generate_parser_and_lexer(args.source_dir, args.build_dir, args.bison, args.flex)
    compile_nmodl(args)
    stamp.write_text(signature)


if __name__ == "__main__":
    main()
