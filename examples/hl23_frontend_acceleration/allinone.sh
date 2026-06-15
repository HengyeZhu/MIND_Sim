#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TSTOP_MS="${TSTOP_MS:-50}"
THREADS="${THREADS:-4}"
MIND_ENV="${MIND_ENV:-mind_sim}"
CORENEURON_ENV="${CORENEURON_ENV:-test}"
CORENEURON_HOME="${CORENEURON_HOME:-${NRN_HOME:-${NEURON_HOME:-}}}"
if [[ -z "${CORENEURON_HOME}" && -d "${HOME}/nrn-install-nvhpc-cpu-fast" ]]; then
  CORENEURON_HOME="${HOME}/nrn-install-nvhpc-cpu-fast"
fi
export NEURON_HOME="${CORENEURON_HOME}"
NVHPC_LIB_DIR="${NVHPC_LIB_DIR:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/lib}"
NVHPC_BIN_DIR="${NVHPC_BIN_DIR:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outputs/allinone_${TSTOP_MS}ms}"

MIND_SCRIPT="${SCRIPT_DIR}/run_mind_sim_hl23.py"
CORENEURON_SCRIPT="${SCRIPT_DIR}/run_coreneuron_hl23.py"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_frontend.py"
MOD_DIR="${SCRIPT_DIR}/assets/mod"
LOGDIR="${OUTDIR}/logs"

MIND_OUTPUT="${MIND_OUTPUT:-${OUTDIR}/mind_sim_hl23_${TSTOP_MS}ms.npz}"
CORENEURON_OUTPUT="${CORENEURON_OUTPUT:-${OUTDIR}/coreneuron_hl23_${TSTOP_MS}ms.npz}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-${OUTDIR}/comparison.json}"
CORENEURON_MOD_BUILD="${OUTDIR}/coreneuron_mod"
CORENEURON_MECH_LIB="${CORENEURON_MOD_BUILD}/x86_64/libnrnmech.so"

mkdir -p "${OUTDIR}" "${LOGDIR}" "${CORENEURON_MOD_BUILD}"

if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_CMD="${HOME}/miniconda3/bin/conda"
else
  echo "error: conda is required on PATH or at ${HOME}/miniconda3/bin/conda" >&2
  exit 1
fi

CONDA_BASE="$("${CONDA_CMD}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

clean_mod_builds() {
  local dir="$1"
  rm -rf "${dir}/x86_64" "${dir}/i686" "${dir}/aarch64" "${dir}/arm64"
}

require_executable() {
  local executable="$1"
  if command -v "${executable}" >/dev/null 2>&1; then
    return
  fi
  echo "error: command not found in active env: ${executable}" >&2
  exit 1
}

require_mind_sim_install() {
  local prefix="${CONDA_PREFIX:-}"
  if [[ -z "${prefix}" ]]; then
    echo "error: no active conda environment" >&2
    exit 1
  fi
  require_executable mind_nrnivmodl
  python - <<'PY'
from pathlib import Path
import os
import mind_sim
import mind_sim._native as native

prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
for path in (Path(mind_sim.__file__).resolve(), Path(native.__file__).resolve()):
    if prefix not in path.parents:
        raise SystemExit(f"mind_sim is not imported from active env: {path}")
print(f"mind_sim={Path(mind_sim.__file__).resolve()}")
print(f"mind_sim_native={Path(native.__file__).resolve()}")
PY
}

activate_coreneuron_env() {
  conda activate "${CORENEURON_ENV}"
  if [[ -n "${CORENEURON_HOME}" ]]; then
    export PATH="${CORENEURON_HOME}/bin:${PATH}"
    export PYTHONPATH="${CORENEURON_HOME}/lib/python"
    export LD_LIBRARY_PATH="${CORENEURON_HOME}/lib:${NVHPC_LIB_DIR}:${CONDA_PREFIX}/lib:/usr/local/cuda/lib64"
  fi
}

set_coreneuron_compile_env() {
  unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
  export CC="${NVHPC_CC:-${NVHPC_BIN_DIR}/nvc}"
  export CXX="${NVHPC_CXX:-${NVHPC_BIN_DIR}/nvc++}"
}

require_coreneuron_install() {
  require_executable nrnivmodl
  python - <<'PY'
from pathlib import Path
import os
import neuron

neuron_file = Path(neuron.__file__).resolve()
coreneuron_home = os.environ.get("NEURON_HOME", "")
if coreneuron_home:
    home = Path(coreneuron_home).resolve()
    if home not in neuron_file.parents:
        raise SystemExit(f"neuron is not imported from CoreNEURON install prefix {home}: {neuron_file}")
print(f"neuron={neuron_file}")
PY
}

prepare_mind_mechanisms() {
  echo
  echo "== MIND_Sim mechanisms =="
  conda activate "${MIND_ENV}"
  require_mind_sim_install
  clean_mod_builds "${MOD_DIR}"
  mind_nrnivmodl "${MOD_DIR}" 2>&1 | tee "${LOGDIR}/mind_nrnivmodl.log"
  if [[ ! -f "${MOD_DIR}/x86_64/libcorenrnmech.so" ]]; then
    echo "error: MIND_Sim mechanism library was not generated" >&2
    exit 1
  fi
  ldd "${MOD_DIR}/x86_64/libcorenrnmech.so" | tee "${LOGDIR}/mind_nrnivmodl.ldd.log"
  conda deactivate
}

prepare_coreneuron_mechanisms() {
  echo
  echo "== CoreNEURON mechanisms =="
  activate_coreneuron_env
  set_coreneuron_compile_env
  require_coreneuron_install
  rm -rf "${CORENEURON_MOD_BUILD}"
  mkdir -p "${CORENEURON_MOD_BUILD}"
  cp -a "${MOD_DIR}/"*.mod "${CORENEURON_MOD_BUILD}/"
  clean_mod_builds "${CORENEURON_MOD_BUILD}"
  (
    cd "${CORENEURON_MOD_BUILD}"
    nrnivmodl -coreneuron .
  ) 2>&1 | tee "${LOGDIR}/coreneuron_nrnivmodl.log"
  if [[ ! -f "${CORENEURON_MECH_LIB}" ]]; then
    echo "error: CoreNEURON mechanism library was not generated: ${CORENEURON_MECH_LIB}" >&2
    exit 1
  fi
  ldd "${CORENEURON_MECH_LIB}" | tee "${LOGDIR}/coreneuron_nrnivmodl.ldd.log"
  conda deactivate
}

run_mind() {
  echo
  echo "== MIND_Sim frontend, CPU =="
  conda activate "${MIND_ENV}"
  require_mind_sim_install
  python "${MIND_SCRIPT}" \
    --duration-ms "${TSTOP_MS}" \
    --threads "${THREADS}" \
    --output "${MIND_OUTPUT}" 2>&1 | tee "${LOGDIR}/mind_sim.log"
  conda deactivate
}

run_coreneuron() {
  echo
  echo "== CPU CoreNEURON baseline =="
  activate_coreneuron_env
  require_coreneuron_install
  python "${CORENEURON_SCRIPT}" \
    --duration-ms "${TSTOP_MS}" \
    --threads "${THREADS}" \
    --mechanism-lib "${CORENEURON_MECH_LIB}" \
    --output "${CORENEURON_OUTPUT}" 2>&1 | tee "${LOGDIR}/coreneuron.log"
  conda deactivate
}

compare_outputs() {
  echo
  echo "== Compare outputs =="
  conda activate "${CORENEURON_ENV}"
  python "${COMPARE_SCRIPT}" \
    --mind "${MIND_OUTPUT}" \
    --coreneuron "${CORENEURON_OUTPUT}" \
    --output-json "${COMPARE_OUTPUT}" 2>&1 | tee "${LOGDIR}/compare.log"
  conda deactivate
}

cd "${REPO_ROOT}"

echo "HL23 frontend acceleration all-in-one workflow"
echo "tstop_ms=${TSTOP_MS}"
echo "threads=${THREADS}"
echo "mind_env=${MIND_ENV}"
echo "coreneuron_env=${CORENEURON_ENV}"
echo "coreneuron_home=${CORENEURON_HOME:-}"
echo "nvhpc_lib_dir=${NVHPC_LIB_DIR}"
echo "outdir=${OUTDIR}"

prepare_mind_mechanisms
prepare_coreneuron_mechanisms
run_mind
run_coreneuron
compare_outputs

echo
echo "Done."
echo "Outputs: ${OUTDIR}"
