#!/usr/bin/env bash
set -euo pipefail

platform="${1:-linux}"
python_selector="${2:-all}"

if [[ "$#" -gt 2 ]]; then
  echo "This helper only builds the default CPU wheels." >&2
  echo "Use a direct CMake configure/build/install for NVHPC or GPU builds." >&2
  exit 2
fi

if [[ "${platform}" != "linux" ]]; then
  echo "Only Linux x86_64 CPU wheels are built for this release." >&2
  exit 2
fi

case "${python_selector}" in
  all)
    export CIBW_BUILD="${CIBW_BUILD:-cp310-* cp311-* cp312-* cp313-*}"
    ;;
  3\*)
    export CIBW_BUILD="${CIBW_BUILD:-cp3*-*}"
    ;;
  cp*)
    export CIBW_BUILD="${CIBW_BUILD:-${python_selector}-*}"
    ;;
  [0-9][0-9]|[0-9][0-9][0-9])
    export CIBW_BUILD="${CIBW_BUILD:-cp${python_selector}-*}"
    ;;
  *)
    echo "Unknown Python selector: ${python_selector}" >&2
    exit 2
    ;;
esac

export CIBW_ARCHS_LINUX="${CIBW_ARCHS_LINUX:-x86_64}"
export CIBW_MANYLINUX_X86_64_IMAGE="${CIBW_MANYLINUX_X86_64_IMAGE:-${MIND_SIM_MANYLINUX_IMAGE:-mindsim_wheel:manylinux_2_28_x86_64}}"
export MIND_SIM_ENABLE_GPU=OFF
export MIND_SIM_BINARY_DIST_BUILD=ON

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${MIND_SIM_BUILD_WHEEL_IMAGE:-1}" != "0" ]]; then
  "${repo_root}/packaging/python/build_manylinux_image.bash"
fi

build_root="$(mktemp -d "${TMPDIR:-/tmp}/mind-simulator-wheel-src.XXXXXX")"
clean_src="${build_root}/src"

cleanup() {
  rm -rf "${build_root}"
}
trap cleanup EXIT

mkdir -p "${clean_src}"

tar \
  --exclude='./.git' \
  --exclude='./.github/actions/cache' \
  --exclude='./.pytest_cache' \
  --exclude='./.wheel-test-venv' \
  --exclude='./build' \
  --exclude='./build-*' \
  --exclude='./cmake-build-*' \
  --exclude='./wheelhouse' \
  --exclude='./examples/HCP_1200' \
  --exclude='./examples/*/outputs' \
  --exclude='./**/__pycache__' \
  --exclude='./**/x86_64' \
  -C "${repo_root}" \
  -cf - . | tar -C "${clean_src}" -xf -

(
  cd "${build_root}"
  python -m cibuildwheel "${clean_src}" --platform "${platform}" --output-dir "${repo_root}/wheelhouse"
)
