<p align="center">
  <img
    alt="MIND_Sim logo"
    src="https://github.com/user-attachments/assets/a24961f9-10a7-4d08-953a-24a38e5a9612"
    height="256"
  >
</p>

<h3 align="center">
  MIND_Sim is designed for studying Multiscale Integrative Neuronal Dynamics
</h3>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#performance">Performance</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<br>

## Installation

### CPU wheel

```bash
pip install mind-simulator
```

### MOD files

```bash
mind-nrnivmodl path/to/mods
```

Requires: C++ compiler.

### NVHPC builds

```bash
conda activate mind_sim
git submodule update --init --recursive

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

Requires: CMake, Bison, Flex, nanobind, and NVIDIA HPC SDK.
Use `-DMIND_SIM_ENABLE_GPU=ON` for the OpenACC GPU build.
Use `-DMIND_SIM_ENABLE_NATIVE_OPT=ON` only for host-specific CPU builds.

## Overview

Currently, it is intended for personal research use, with features added as needed.

MIND_Sim is an extension simulator based on the [NEURON Simulator](https://github.com/neuronsimulator/nrn) for studying Multiscale-Integrative Neuronal Dynamics. It is organized around three main ideas:

1. It rewrites the frontend network-modeling layer of NEURON in C++, while preserving a NEURON-style API, and improves network construction speed by more than 10x.
2. It extends the MOD DSL to describe neural population dynamics and micro2macro transformations, providing a flexible way to build hybrid models and addressing the current limitation that [the TVB platform does not yet support hybrid models](https://github.com/the-virtual-brain/tvb-root/pull/771).
3. It treats regions of interest (ROIs) as first-class modeling objects, so users can freely choose the brain regions and scales they want to simulate.

A demo blog post introducing the MIND_Sim workflow is available [here](https://hengyezhu.github.io/mind-simulator-demo.html).

## Architecture

### Execution model

MIND_Sim is designed for multiscale modeling, but it also supports micro-only and macro-only simulations. The micro-scale backend is built on CoreNEURON, so detailed neuron simulations can use CPU multi-threading and CoreNEURON GPU execution. The macro-scale backend is designed around an overlap pipeline: while the micro simulation advances one exchange window, the CPU-side macro and transform work can be prepared or executed around that window. The macro layer is currently single-threaded; for the neural mass models used so far, this is sufficient because macro computation is small enough to be hidden behind micro execution.

### ROI-centered modeling

At the macro scale, users load a labelled connectivity matrix, and MIND_Sim automatically creates one ROI object for each label. Each ROI can then choose the scale that is meaningful for the model: a detailed microcircuit or a neural mass model. Different ROIs may use different equations, different exposed variables, and different coupling rules.

### MOD-based model definitions

MIND_Sim extends the MOD language used by the NEURON Simulator from micro-scale mechanisms to neural mass models and cross-scale transform modules. This keeps the modeling style aligned with NEURON/NMODL, so users who are already familiar with micro-scale mechanisms can move to micro2macro transforms without learning a completely separate model-description language. It also keeps the flexibility of modular mechanism definitions: equations and transform rules can be written independently, combined with different ROI models, and replaced without changing the rest of the simulation.

Based on the same idea, macro-scale network construction also follows a NEURON-like syntax.

The current micro-macro transformation is event based. This follows the same general direction as recent Arbor-TVB and TVB-NEST co-simulation work, where spiking activity and whole-brain variables are exchanged through explicit transformation modules: [Hater, Courson, Lu, Diaz-Pier, and Manos (2026), Arbor-TVB: a novel multi-scale co-simulation framework with a case study on neural-level seizure generation and whole-brain propagation](https://doi.org/10.3389/fncom.2025.1731161), and [Kusch, Diaz-Pier, Klijn, Sontheimer, Bernard, Morrison, and Jirsa (2024), Multiscale co-simulation design pattern for neuroscience applications](https://doi.org/10.3389/fninf.2024.1156683). From a NEURON Simulator perspective, micro2macro transformation is similar to synaptic event handling: each spike is delivered as an event and contributes to a macro exposure or state variable. Macro2micro transformation is NetStim-like: macro variables are converted into generated spike events that feed back into selected micro synapses.

### Mapping microcircuits to ROIs

A single microcircuit can contribute selected exposures for one or more ROIs. With this interface, connectome-based models can perform ROI-to-ROI coupling through exposures. Region equations, ROI-to-ROI coupling, and micro2macro transforms write ROI exposures according to the mechanisms attached to each ROI.

## Performance

By rewriting the NEURON simulator frontend in C++, MIND_Sim improves micro-network construction speed by more than 10x. This has been observed across multiple models.

### Cosim

For full cosimulation, runtime is usually dominated by the micro-scale simulation itself. Therefore, although MIND_Sim provides an overlap pipeline, its benefit is limited in this example because macro simulation and transformation together take only a small fraction of the total runtime. The [CA3 epilepsy co-simulation example](https://github.com/HengyeZhu/MIND_Sim/tree/main/examples/ca3_epilepsy_cosim) compares MIND_Sim with a TVB+NEURON reference. The following timings are for 1s simulation time.

| Workflow | Threads | Pre-run | Run | Speedup |
| --- | ---: | ---: | ---: | ---: |
| MIND_Sim async | 1 | 0.187s | 16.702s | 3.97x |
| TVB+NEURON | 1 | 0.591s | 66.327s | 1.00x |
| MIND_Sim async | 4 | 0.169s | 6.672s | 4.88x |
| TVB+NEURON | 4 | 0.558s | 32.531s | 1.00x |

For the same 1s runs, using the TVB+NEURON reference with TVB's official macro APIs, the maximum absolute differences between MIND_Sim and the reference are `1.28464e-06` for macro `x`, `1.4922e-09` for macro `z`, and `8.98019e-11 mV` for representative PYR, BAS, OLM, and PYR Adend3 voltage traces. The macro comparison includes the precision boundary between TVB's single-precision (`float32`) state/history storage and MIND_Sim's double-precision macro state. Spike sample indices are exactly equal for the representative PYR, BAS, and OLM cells. This is an example-level comparison, not a standardized benchmark.

The CA3 all-in-one script uses TVB+NEURON as the reference. A direct TVB+CoreNEURON baseline is not used in the table because, in a short-window TVB loop, each `pc.psolve()` call repeatedly pays NEURON-side model preparation and CoreNEURON-side model loading costs. These costs made TVB+CoreNEURON slower than TVB+NEURON by 3.68x with one thread and 5.65x with four threads. This motivates a co-simulation simulator built directly on CoreNEURON: MPI-level coupling alone cannot efficiently use CoreNEURON's GPU mode for future scaling.

### Frontend Build

The [HL23 frontend acceleration example](https://github.com/HengyeZhu/MIND_Sim/tree/main/examples/hl23_frontend_acceleration) compares MIND_Sim's Python frontend with a CoreNEURON baseline. The following timings are for 100ms simulation time.

| Workflow | Threads | Pre-run | Run |
| --- | ---: | ---: | ---: |
| MIND_Sim frontend (CPU) | 4 | 0.751s | 97.715s |
| MIND_Sim frontend (GPU) | GPU | 0.992s | 19.054s |
| CoreNEURON | 4 | 16.001s | 102.338s |

For the same 100ms run, the maximum absolute voltage difference is `1.88621e-07 mV` across the recorded soma traces. Spike times are exactly equal.

Frontend modeling speedup depends on two main factors. More biophysically detailed neuron models benefit more from MIND_Sim's faster frontend construction. Networks with heavier connection construction benefit less, because connection setup is still limited by loop-based creation patterns.

## Acknowledgements

I sincerely acknowledge the contributions of simulators such as [NEURON](https://github.com/neuronsimulator/nrn), [Arbor](https://github.com/arbor-sim/arbor/), [GeNN](https://github.com/genn-team/genn), and [TVB](https://github.com/the-virtual-brain/tvb-root) to computational neuroscience.
