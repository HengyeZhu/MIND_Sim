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

MIND_Sim is currently distributed from source. pip install support is coming soon.

## Overview

Currently, it is intended for personal research use, with features added as needed.

MIND_Sim is an extension simulator based on the [NEURON Simulator](https://github.com/neuronsimulator/nrn) for studying Multiscale-Integrative Neuronal Dynamics. It is organized around three main ideas:

1. It rewrites the frontend network-modeling layer of NEURON in C++, while preserving a NEURON-style API, and improves network construction speed by more than 10x.
2. It extends the MOD DSL to describe neural population dynamics and micro2macro transformations, providing a flexible way to build hybrid models and addressing the current limitation that [the TVB platform does not yet support hybrid models](https://github.com/the-virtual-brain/tvb-root/pull/771).
3. It treats regions of interest (ROIs) as first-class modeling objects, so users can freely choose the brain regions and scales they want to simulate.

This is my first public release of a simulator. I have made efforts to comply with the licenses of the open-source projects on which this work depends, but I may still have overlooked some details. If you notice that any reuse requirement has not been handled correctly, please contact me at [gluciferd@gmail.com](mailto:gluciferd@gmail.com).

## Architecture

### Execution model

MIND_Sim is designed for multiscale modeling, but it also supports micro-only and macro-only simulations. The micro-scale backend is built on CoreNEURON, so detailed neuron simulations can use CPU multi-threading and CoreNEURON GPU execution. The macro-scale backend is designed around an overlap pipeline: while the micro simulation advances one exchange window, the CPU-side macro and transform work can be prepared or executed around that window. The macro layer is currently single-threaded; for the neural mass models used so far, this is sufficient because macro computation is small enough to be hidden behind micro execution.

### ROI-centered modeling

At the macro scale, users load a labelled connectivity matrix, and MIND_Sim automatically creates one ROI object for each label. Each ROI can then choose the scale that is meaningful for the model: a detailed microcircuit, a neural field model, or a neural mass model. Different ROIs may use different equations, different exposed variables, and different coupling rules.

### MOD-based model definitions

MIND_Sim extends the MOD language used by the NEURON Simulator from micro-scale mechanisms to neural mass models and cross-scale transform modules. This keeps the modeling style aligned with NEURON/NMODL, so users who are already familiar with micro-scale mechanisms can move to micro2macro transforms without learning a completely separate model-description language. It also keeps the flexibility of modular mechanism definitions: equations and transform rules can be written independently, combined with different ROI models, and replaced without changing the rest of the simulation.

Based on the same idea, macro-scale network construction also follows a NEURON-like syntax.

The current micro-macro transformation is event based. This follows the same general direction as recent Arbor-TVB and TVB-NEST co-simulation work, where spiking activity and whole-brain variables are exchanged through explicit transformation modules: [Hater, Courson, Lu, Diaz-Pier, and Manos (2026), Arbor-TVB: a novel multi-scale co-simulation framework with a case study on neural-level seizure generation and whole-brain propagation](https://doi.org/10.3389/fncom.2025.1731161), and [Kusch, Diaz-Pier, Klijn, Sontheimer, Bernard, Morrison, and Jirsa (2024), Multiscale co-simulation design pattern for neuroscience applications](https://doi.org/10.3389/fninf.2024.1156683). From a NEURON Simulator perspective, micro2macro transformation is similar to synaptic event handling: each spike is delivered as an event and contributes to a macro exposure or state variable. Macro2micro transformation is NetStim-like: macro variables are converted into generated spike events that feed back into selected micro synapses.

### Mapping microcircuits to ROIs

A single microcircuit can own or contribute to any number of ROIs. This matters in connectome-based modeling because every ROI has its own exposures and inputs, even when several ROIs are represented by different parts or projections of the same detailed micro model. Instead of forcing one micro model to collapse into one macro node, MIND_Sim lets each ROI declare its own exposed variables and accepted inputs. Different subsets of the same micro simulation can therefore transform their spikes or states into different ROI-level values, and different ROI inputs can be routed back to different micro targets.

## Performance

By rewriting the NEURON simulator frontend in C++, MIND_Sim improves micro-network construction speed by more than 10x. This has been observed across multiple models.

For full cosimulation, runtime is usually dominated by the micro-scale simulation itself. Therefore, although MIND_Sim provides an overlap pipeline, its benefit is limited in this example because macro simulation and transformation together take only a small fraction of the total runtime. The [CA3 epilepsy co-simulation example](https://github.com/HengyeZhu/MIND_Sim/tree/main/examples/ca3_epilepsy_cosim) gives the following timings (1s simulation time).

| Workflow | Threads | Pre-run | Run | Speedup |
| --- | ---: | ---: | ---: | ---: |
| MIND_Sim async | 1 | 0.281s | 14.286s | 21.40x |
| MIND_Sim async | 4 | 0.288s | 5.581s | 21.58x |
| TVB-NetPyNE | 1 | 11.060s | 305.707s | 1.00x |
| TVB-NetPyNE | 4 | 10.646s | 120.413s | 1.00x |

This is an example-level comparison, not a standardized benchmark. A broader benchmark and full validation are coming soon.

## Acknowledgements

I sincerely acknowledge the contributions of simulators such as [NEURON](https://github.com/neuronsimulator/nrn), [Arbor](https://github.com/arbor-sim/arbor/), [GeNN](https://github.com/genn-team/genn), and [TVB](https://github.com/the-virtual-brain/tvb-root) to computational neuroscience.
