# Examples

Run commands from the `examples/` directory.

This repository currently keeps one CA3 epilepsy whole-brain cosimulation example:

```text
ca3_epilepsy_cosim/
```

## Data

The full workflow was designed around a subject-specific connectivity matrix built from real HCP subject `100206` data. Because the original imaging data are distributed under their own access and usage terms, this repository uses a synthetic connectivity matrix for the public example.

The synthetic file keeps the same labelled matrix format used by the real workflow, but its edge weights, delays, and topology are not subject-specific data:

```text
ca3_epilepsy_cosim/data/synthetic_hybrid_ca3_connectivity.csv
```

Users who have access to the original HCP data can regenerate the real connectivity matrix with the preprocessing script:

```text
ca3_epilepsy_cosim/mind_sim/prepare_hcp100206_ca3.py
```

That script prepares the HCP/HippUnfold-derived CA3 parcellation and converts the processed connectome into the labelled matrix CSV format expected by the simulator.

## Model Sources

The micro model is based on the CA3 epilepsy network from ModelDB 186768.

The macro model is based on the TVB `Epileptor2D` neural mass model. 

## Compile Mechanisms

The example uses one mechanism directory:

```text
ca3_epilepsy_cosim/mind_sim/mod/
```

Run the following commands from `examples/`.

`mod/` contains the standard NEURON/CoreNEURON mechanisms used by the CA3 microcircuit and the MIND_Sim extended MOD modules for macro dynamics, macro coupling, micro2macro transforms, and macro2micro transforms. Compile it with the MIND_Sim compiler:

```bash
mind_nrnivmodl ca3_epilepsy_cosim/mind_sim/mod
```

## MIND_Sim Workflow

The complete MIND_Sim modeling flow for this example is in:

```text
ca3_epilepsy_cosim/mind_sim/run_vep_ca3_cosim.py
```

A detailed [blog](https://hengyezhu.github.io/mind-simulator-demo.html) is here.

## Validation

The validation workflow is still being refined. A clearer comparison guide and reference validation procedure are coming soon.
