#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TSTOP_MS="${TSTOP_MS:-50}"
THREADS="${THREADS:-4}"
MIND_ENV="${MIND_ENV:-mind_sim}"
CORENEURON_ENV="${CORENEURON_ENV:-test}"
CORENEURON_HOME="${CORENEURON_HOME:-${NRN_HOME:-${NEURON_HOME:-}}}"
export NEURON_HOME="${CORENEURON_HOME}"
NVHPC_LIB_DIR="${NVHPC_LIB_DIR:-}"
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

mkdir -p "${OUTDIR}" "${LOGDIR}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

clean_mod_builds() {
  local dir="$1"
  rm -rf "${dir}/x86_64" "${dir}/i686" "${dir}/aarch64" "${dir}/arm64"
}

prepend_ld_library_path() {
  local path="$1"
  if [[ -z "${path}" || ! -d "${path}" ]]; then
    return
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${path}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${path}"
  fi
}

activate_coreneuron_env() {
  conda activate "${CORENEURON_ENV}"
  if [[ -n "${CORENEURON_HOME}" ]]; then
    export PATH="${CORENEURON_HOME}/bin:${PATH}"
    export PYTHONPATH="${CORENEURON_HOME}/lib/python"
    prepend_ld_library_path "${CORENEURON_HOME}/lib"
  fi
  prepend_ld_library_path "${NVHPC_LIB_DIR}"
  prepend_ld_library_path "${CONDA_PREFIX}/lib"
}

set_coreneuron_compile_env() {
  unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
  export CC="${NVHPC_CC:-nvc}"
  export CXX="${NVHPC_CXX:-nvc++}"
}

prepare_mind_mechanisms() {
  echo
  echo "== MIND_Sim mechanisms =="
  conda activate "${MIND_ENV}"
  clean_mod_builds "${MOD_DIR}"
  mind-nrnivmodl "${MOD_DIR}" 2>&1 | tee "${LOGDIR}/mind-nrnivmodl.log"
  conda deactivate
}

prepare_coreneuron_mechanisms() {
  echo
  echo "== CoreNEURON mechanisms =="
  activate_coreneuron_env
  set_coreneuron_compile_env
  rm -rf "${CORENEURON_MOD_BUILD}"
  mkdir -p "${CORENEURON_MOD_BUILD}"
  cp -a "${MOD_DIR}/"*.mod "${CORENEURON_MOD_BUILD}/"
  clean_mod_builds "${CORENEURON_MOD_BUILD}"
  (
    cd "${CORENEURON_MOD_BUILD}"
    nrnivmodl -coreneuron .
  ) 2>&1 | tee "${LOGDIR}/coreneuron_nrnivmodl.log"
  conda deactivate
}

run_mind() {
  echo
  echo "== MIND_Sim frontend, CPU =="
  conda activate "${MIND_ENV}"
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
echo "nvhpc_lib_dir=${NVHPC_LIB_DIR:-<not set>}"
echo "outdir=${OUTDIR}"

prepare_mind_mechanisms
prepare_coreneuron_mechanisms
run_mind
run_coreneuron
compare_outputs

echo
echo "Done."
echo "Outputs: ${OUTDIR}"
