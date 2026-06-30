#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DURATION_MS="${DURATION_MS:-1000}"
MIND_ENV="${MIND_ENV:-mind_sim}"
TVB_ENV="${TVB_ENV:-test}"
CONNECTIVITY_CSV="${CONNECTIVITY_CSV:-${SCRIPT_DIR}/data/synthetic_hybrid_ca3_connectivity.csv}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outputs/allinone_${DURATION_MS}ms}"
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
REBUILD_MECHS="${REBUILD_MECHS:-1}"

MIND_SCRIPT="${SCRIPT_DIR}/mind_sim/run_vep_ca3_cosim.py"
TVB_NEURON_SCRIPT="${SCRIPT_DIR}/neuron_tvb/run_tvb_neuron_ca3_cosim.py"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_allinone.py"
LOGDIR="${OUTDIR}/logs"
PLOTDIR="${OUTDIR}/plots"

MIND_1="${MIND_1:-${OUTDIR}/mind_${DURATION_MS}ms_1thread.npz}"
MIND_4="${MIND_4:-${OUTDIR}/mind_${DURATION_MS}ms_4thread.npz}"
TVB_1="${TVB_1:-${OUTDIR}/tvb_neuron_${DURATION_MS}ms_1thread.npz}"
TVB_4="${TVB_4:-${OUTDIR}/tvb_neuron_${DURATION_MS}ms_4thread.npz}"

mkdir -p "${OUTDIR}" "${LOGDIR}" "${PLOTDIR}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

clean_mod_builds() {
  local mod_dir="$1"
  rm -rf "${mod_dir}/x86_64" \
         "${mod_dir}/i686" \
         "${mod_dir}/aarch64" \
         "${mod_dir}/arm64"
}

prepare_mind_mechanisms() {
  local mod_dir="${SCRIPT_DIR}/mind_sim/mod"

  echo
  echo "== MIND_Sim mechanisms =="
  conda activate "${MIND_ENV}"
  clean_mod_builds "${mod_dir}"
  mind-nrnivmodl "${mod_dir}" 2>&1 | tee "${LOGDIR}/mind-nrnivmodl.log"
  conda deactivate
}

prepare_tvb_neuron_mechanisms() {
  local mod_dir="${SCRIPT_DIR}/neuron_tvb/mod"

  echo
  echo "== TVB+NEURON mechanisms =="
  conda activate "${TVB_ENV}"
  clean_mod_builds "${mod_dir}"
  (
    cd "${mod_dir}"
    nrnivmodl .
  ) 2>&1 | tee "${LOGDIR}/tvb_neuron_nrnivmodl.log"
  conda deactivate
}

prepare_mechanisms() {
  if [[ "${REBUILD_MECHS}" != "1" ]]; then
    echo
    echo "== Mechanism rebuild skipped because REBUILD_MECHS=${REBUILD_MECHS} =="
    return
  fi
  prepare_mind_mechanisms
  prepare_tvb_neuron_mechanisms
}

run_mind() {
  local threads="$1"
  local output="$2"
  local log="${LOGDIR}/mind_${DURATION_MS}ms_${threads}thread.log"

  echo
  echo "== MIND_Sim async, ${threads} thread(s) =="
  conda activate "${MIND_ENV}"
  python "${MIND_SCRIPT}" \
    --connectivity-csv "${CONNECTIVITY_CSV}" \
    --duration-ms "${DURATION_MS}" \
    --micro-threads "${threads}" \
    --output "${output}" 2>&1 | tee "${log}"
  conda deactivate
}

run_tvb_neuron() {
  local threads="$1"
  local output="$2"
  local log="${LOGDIR}/tvb_neuron_${DURATION_MS}ms_${threads}thread.log"

  echo
  echo "== TVB+NEURON, ${threads} thread(s) =="
  conda activate "${TVB_ENV}"
  python "${TVB_NEURON_SCRIPT}" \
    --connectivity-csv "${CONNECTIVITY_CSV}" \
    --duration-ms "${DURATION_MS}" \
    --micro-threads "${threads}" \
    --output "${output}" \
    --workdir "${OUTDIR}/tvb_neuron_workdir_${threads}thread" 2>&1 | tee "${log}"
  conda deactivate
}

compare_and_plot() {
  echo
  echo "== Compare and plot =="
  conda activate "${TVB_ENV}"
  python "${COMPARE_SCRIPT}" \
    --outdir "${OUTDIR}" \
    --plotdir "${PLOTDIR}" \
    --duration-ms "${DURATION_MS}" \
    --mind-1 "${MIND_1}" \
    --mind-4 "${MIND_4}" \
    --tvb-neuron-1 "${TVB_1}" \
    --tvb-neuron-4 "${TVB_4}"
  conda deactivate
}

cd "${REPO_ROOT}"

echo "CA3 all-in-one workflow"
echo "duration_ms=${DURATION_MS}"
echo "connectivity_csv=${CONNECTIVITY_CSV}"
echo "outdir=${OUTDIR}"
echo "mind_env=${MIND_ENV}"
echo "tvb_env=${TVB_ENV}"
echo "rebuild_mechs=${REBUILD_MECHS}"

if [[ "${RUN_EXPERIMENTS}" == "1" ]]; then
  prepare_mechanisms
  run_mind 1 "${MIND_1}"
  run_mind 4 "${MIND_4}"
  run_tvb_neuron 1 "${TVB_1}"
  run_tvb_neuron 4 "${TVB_4}"
else
  echo
  echo "== Experiments skipped because RUN_EXPERIMENTS=${RUN_EXPERIMENTS} =="
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  compare_and_plot
else
  echo
  echo "== Compare/plot skipped because RUN_COMPARE=${RUN_COMPARE} =="
fi

echo
echo "Done."
echo "Outputs: ${OUTDIR}"
