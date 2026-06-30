#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image_tag="${MIND_SIM_MANYLINUX_IMAGE:-mindsim_wheel:manylinux_2_28_x86_64}"

if [[ "${MIND_SIM_FORCE_WHEEL_IMAGE_BUILD:-0}" != "1" ]] &&
   docker image inspect "${image_tag}" >/dev/null 2>&1; then
  exit 0
fi

docker build \
  -t "${image_tag}" \
  --build-arg "MANYLINUX_TAG=${MIND_SIM_MANYLINUX_TAG:-2026.06.04-1}" \
  -f "${repo_root}/packaging/python/manylinux/Dockerfile" \
  "${repo_root}/packaging/python/manylinux"
