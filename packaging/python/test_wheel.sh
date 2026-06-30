#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $(basename "$0") PYTHON_EXE WHEEL_PATTERN [PYTEST_ARGS...]" >&2
  exit 2
fi

python_exe="$1"
wheel_pattern="$2"
shift 2

pytest_args=("$@")
if [[ "${#pytest_args[@]}" -eq 0 ]]; then
  pytest_args=(tests)
fi

shopt -s nullglob
wheels=( ${wheel_pattern} )
shopt -u nullglob

if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one wheel for pattern: ${wheel_pattern}" >&2
  printf 'Matched wheels:\n' >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 2
fi

venv_dir="${MIND_SIM_WHEEL_TEST_VENV:-.wheel-test-venv}"
rm -rf "${venv_dir}"
"${python_exe}" -m venv "${venv_dir}"

# shellcheck source=/dev/null
source "${venv_dir}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "${wheels[0]}"
python -m pip install pytest scipy

python - <<'PY'
from pathlib import Path
import shutil
import mind_sim
import mind_sim._native as native

assert hasattr(mind_sim, "load_mech")
assert not hasattr(mind_sim, "Sim")
assert isinstance(mind_sim.micro.sim(), native.Sim)
package_path = Path(mind_sim.__file__).resolve()
source_package = (Path.cwd() / "src" / "python_api" / "mind_sim").resolve()
assert not package_path.is_relative_to(source_package)
print(package_path)
print(Path(native.__file__).resolve())

for build_dir in Path("tests").glob("**/x86_64"):
    if build_dir.is_dir():
        shutil.rmtree(build_dir)
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

python -m pytest "${pytest_args[@]}"
