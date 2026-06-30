ARG MANYLINUX_IMAGE=manylinux_2_28_x86_64
ARG MANYLINUX_TAG=2026.06.04-1
FROM quay.io/pypa/${MANYLINUX_IMAGE}:${MANYLINUX_TAG}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN dnf install -y \
      bison \
      flex \
      libgomp \
      make \
      curl \
      tar \
      gzip \
      findutils \
      bzip2-devel \
      expat-devel \
      libffi-devel \
      openssl-devel \
      sqlite-devel \
      xz-devel \
      zlib-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN cat > /usr/local/bin/build-shared-libpython <<'EOF' \
    && chmod +x /usr/local/bin/build-shared-libpython
#!/usr/bin/env bash
set -euo pipefail

version="$1"
major_minor="${version%.*}"
prefix="/opt/_internal/cpython-${version}"
archive="/tmp/Python-${version}.tgz"
url="https://www.python.org/ftp/python/${version}/Python-${version}.tgz"

for attempt in 1 2 3 4 5; do
  if curl -fL -o "${archive}" "${url}"; then
    break
  fi
  rm -f "${archive}"
  if [[ "${attempt}" == "5" ]]; then
    exit 1
  fi
  sleep "$((attempt * 3))"
done
tar -xzf "${archive}" -C /tmp
cd "/tmp/Python-${version}"
./configure \
  --prefix="${prefix}" \
  --enable-shared \
  --with-static-libpython=no \
  --without-ensurepip
make -j"${CPYTHON_BUILD_JOBS:-$(nproc)}" "libpython${major_minor}.so"
cp "libpython${major_minor}.so.1.0" "${prefix}/lib/"
ln -sf "libpython${major_minor}.so.1.0" "${prefix}/lib/libpython${major_minor}.so"
strip --strip-debug "${prefix}/lib/libpython${major_minor}.so.1.0" || true
cd /
rm -rf "/tmp/Python-${version}" "${archive}"
EOF

RUN build-shared-libpython 3.10.20
RUN build-shared-libpython 3.11.15
RUN build-shared-libpython 3.12.13
RUN build-shared-libpython 3.13.13
