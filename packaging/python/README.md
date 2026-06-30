# Python Wheel Builds

The first MIND_Sim release builds Linux x86_64 CPU wheels only:

- normal CPU installs use the prebuilt wheel: `pip install mind-simulator`
- NVHPC/GPU installs use direct CMake configure/build/install, not Python wheels

Build the default CPU wheels:

```bash
bash packaging/python/build_wheels.bash linux
```

Build one Python version:

```bash
bash packaging/python/build_wheels.bash linux 311
```

Test an installed wheel with the repository test suite:

```bash
bash packaging/python/test_wheel.sh "$(command -v python)" "wheelhouse/mind_simulator-*-cp311-*.whl"
```

The release workflow runs this wheel test for each published Python wheel before
uploading to PyPI.

This helper intentionally has no NVHPC or GPU wheel mode. Those builds depend on
the local compiler, CUDA driver/runtime, and GPU node policy, so they should be
installed with CMake in the target environment.

## NVHPC/GPU CMake Installs

Load or activate the NVIDIA HPC SDK first. Module names are site-specific:

```bash
module load nvidia-hpc-sdk cuda
```

Install an NVHPC CPU build under the active Python environment prefix:

```bash
CMAKE_BIN="$(command -v cmake)"
PYTHON_BIN="$(command -v python)"
NVCXX_BIN="$(command -v nvc++)"
unset CC CXX LD CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX
export PATH="/usr/bin:/bin:${PATH}"

"${CMAKE_BIN}" -S . -B build-nvhpc-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCXX_BIN}" \
  -DMIND_SIM_ENABLE_GPU=OFF \
  -DPython_EXECUTABLE="${PYTHON_BIN}"
"${CMAKE_BIN}" --build build-nvhpc-cpu -j
"${CMAKE_BIN}" --install build-nvhpc-cpu --prefix "$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
```

Install a GPU/OpenACC build under the active Python environment prefix:

```bash
CMAKE_BIN="$(command -v cmake)"
PYTHON_BIN="$(command -v python)"
NVCXX_BIN="$(command -v nvc++)"
unset CC CXX LD CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX
export PATH="/usr/bin:/bin:${PATH}"

"${CMAKE_BIN}" -S . -B build-nvhpc-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCXX_BIN}" \
  -DMIND_SIM_ENABLE_GPU=ON \
  -DPython_EXECUTABLE="${PYTHON_BIN}"
"${CMAKE_BIN}" --build build-nvhpc-gpu -j
"${CMAKE_BIN}" --install build-nvhpc-gpu --prefix "$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
```

Direct CMake installs place the Python package under the selected Python
environment's `site-packages` directory when installed into that environment
prefix.

Use a real GPU node to validate GPU builds.

The relevant CMake switches are:

```text
CMAKE_CXX_COMPILER=/path/to/nvc++     use NVIDIA HPC SDK explicitly
MIND_SIM_ENABLE_GPU=ON                enable the CoreNEURON OpenACC GPU path
MIND_SIM_ENABLE_NATIVE_OPT=ON         tune CPU code for the build host
```
