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

## Architecture

coming soon...

## Performance

By rewriting the NEURON simulator frontend in C++, MIND_Sim can speed up model construction by more than 10x while still using CoreNEURON as the backend. This keeps backend performance and numerical results aligned with CoreNEURON.

The macro and bridge components are also implemented in C++, so for small-scale micro simulations the total runtime is significantly faster than TVB-Multiscale. A GPU backend, also based on CoreNEURON, is planned and is expected to provide substantial acceleration for large-scale micro simulations.

The current examples and tests are still incomplete, and the macro and bridge APIs remain under active development.

## Acknowledgements

I acknowledge the contributions of simulators such as **[NEURON](https://github.com/neuronsimulator/nrn)**, **[GeNN](https://github.com/genn-team/genn)** and **[TVB](https://github.com/the-virtual-brain/tvb-root)** to computational neuroscience.

This is a project developed during my learning process, and it draws on many existing ideas and implementations. In the future, I plan to redevelop the entire framework based on my own understanding.
