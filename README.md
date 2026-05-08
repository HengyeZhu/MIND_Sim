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
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#performance">Performance</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<br>

## Overview

Currently, it is intended for personal research use, with features added as needed.

The central modeling semantics are ROI-first. In this view, an ROI is the public coupling interface of the network, not a fixed commitment to one internal scale of description. Users can choose the scale that is meaningful for each ROI, such as a neural mass model, a neural field model, a detailed microcircuit. Coupling between ROIs is expressed through ROI-level exposures, while the values of those exposures are produced entirely by the internal dynamics owned by each ROI. This keeps the network-level interface simple and explicit, while allowing high freedom in how each ROI is modeled internally. Bridge rules then translate ROI-level exposures into the concrete values required by each downstream dynamical simulation.

This is my first public release of a simulator project. Its main motivation is the lack of a unified framework for multiscale simulation. While learning from the existing ecosystem, I tried to formulate my own frontend modeling approach. For backend computation, I reused established workflows wherever possible: the micro-scale execution is built on CoreNEURON, and the macro-scale layer was informed by TVB-style computation. I have made efforts to comply with the licenses of the open-source projects on which this work depends, but I may still have overlooked some details. If you notice any issue or believe any attribution or reuse requirement has not been handled correctly, please contact me at [gluciferd@gmail.com](mailto:gluciferd@gmail.com).

## Architecture

### Frontend modeling

At the micro scale, the frontend follows the style of the NEURON simulator. In contrast to a general-purpose interactive simulator frontend, this layer is intentionally focused on the computational objects required for execution.

At the macro scale, brain regions are represented as first-class region-of-interest (ROI) objects, and inter-regional interactions are created explicitly as connections between ROI objects. Neural mass models specify the intrinsic dynamics of each ROI using a code-generation style similar in spirit to GeNN: equations are provided at the frontend and compiled into backend kernels. Coupling rules between ROIs, as well as bridge rules between macro-scale and micro-scale components, are supplied by `.mod` files and processed by a dedicated MIND_Sim translator.

This separation between intrinsic ROI dynamics, inter-ROI coupling, and cross-scale exchange makes the modeling interface more flexible, particularly in a field that is still evolving and does not yet have fully settled modeling conventions.

### Backend

The backend currently supports CPU execution on a single core, as well as a GPU mode. Because the dominant computational cost lies in the micro-scale simulation, GPU acceleration is applied only to the micro component. The remaining backend pipeline runs on the CPU, and in the examples tested so far its throughput is sufficient to overlap the macro-scale computation. Micro computation is handled by CoreNEURON, while the macro layer currently supports only discrete-node ROIs.

## Performance

By rewriting the NEURON simulator frontend in C++, MIND_Sim can speed up model construction by more than 10x while still using CoreNEURON as the backend. This keeps backend performance and numerical results aligned with CoreNEURON.

The macro and bridge components are also implemented in C++, so for small-scale micro simulations the total runtime is significantly faster than TVB-Multiscale. A GPU backend, also based on CoreNEURON, is planned and is expected to provide substantial acceleration for large-scale micro simulations.

The current examples and tests are still incomplete, and the macro and bridge APIs remain under active development.

## Acknowledgements

I acknowledge the contributions of simulators such as **[NEURON](https://github.com/neuronsimulator/nrn)**, **[Arbor](https://github.com/arbor-sim/arbor/)**, **[GeNN](https://github.com/genn-team/genn)** and **[TVB](https://github.com/the-virtual-brain/tvb-root)** to computational neuroscience.

This is a project developed during my learning process, and it draws on many existing ideas and implementations. In the future, I plan to redevelop the entire framework based on my own understanding.
