# Source Note

Vendored model:

- ModelDB accession: 186768
- Repository: `https://github.com/ModelDBRepository/186768`
- Vendored commit: `e60d5e99eb2a6836485b35cc3f8aae0085bee79f`
- Paper: Sanjay M, Neymotin SA, Krothapalli SB. Impaired dendritic inhibition leads to epileptic activity in a computer model of CA3. Hippocampus 25:1336-1350, 2015.

The original upstream files are kept under `modeldb_186768/` without local edits. Example-specific scripts live one directory above the vendored model. The MIND Sim rewrite uses `mind_mod_ca3/`, a selected copy of the CA3 channel and synapse mechanisms. The upstream utility mods `stats.mod`, `vecst.mod`, `wrap.mod`, and `misc.mod` are not part of the MIND Sim mechanism set because they are old NEURON helper mechanisms and are not needed for the deterministic CA3 cell/synapse dynamics.

Runtime note:

NEURON 9 requires `Random123` for `NetStim.noiseFromRandom()`. The upstream code uses older `MCellRan4` initialization. The example runners install a runtime monkey patch for NetStim random streams so the vendored source can stay unchanged.

Validation note:

`ca3_mind_sim_api.py` is a native MIND Sim API rewrite, not a wrapper around the original `networkmsj.py`. `run_ca3_original_validation.py` builds a deterministic original-NEURON baseline from the upstream `geom.py` classes, and `compare_ca3_validation.py` checks recorded voltages plus spike ids/times. The 200 ms full-connectivity local check matches with maximum voltage error `8.113687499644584e-11 mV` and exact spike times.

`run_macro_only_validation.py` validates the reduced VEP macro owner separately against a NumPy reference with the same delayed coupling and equation evaluation order. The 200 ms local check matches with `x_max_abs = 5.551115123125783e-16` and `z_max_abs = 1.1102230246251565e-16`.

`run_tvb_netpyne_multiscale_validation.py` runs the local `tvb-multiscale` NetPyNE-TVB serial orchestrator as an external interface sanity check. Its upstream model is `ReducedWongWangExcIOInhI` plus the default NetPyNE HH-soma network, so it is not a same-model precision reference for the reduced VEP plus ModelDB 186768 CA3 MIND Sim cosimulation. The script records this mismatch in its JSON report.

`run_tvb_multiscale_vep_ca3_reference.py` is the same-model TVB-multiscale migration: CA3 PYR/BAS/OLM cells and deterministic projections are rebuilt through NetPyNE/NEURON under `tvb_multiscale.tvb_netpyne.netpyne.module.NetpyneModule`, while the reduced VEP macro scaffold and 10 ms bridge semantics are matched to the MIND Sim example. `compare_mindsim_tvb_multiscale.py` compares this migrated reference to the MIND Sim NPZ output. The 200 ms local check passes with exact spike ids/times and macro/bridge differences below `2.2e-14`.

The model is not human. It is still the current target for this MIND_Sim example because it is:

- epilepsy-specific
- CA3-specific
- conductance based
- runnable through NEURON
- larger than the small focal seizure examples checked earlier

Atlas note:

Use a whole-brain atlas with hippocampal subfields. The implemented data-prep
path follows the current NEST-TVB hippocampus cosimulation strategy at the
macro-data level: derive the macro scaffold from subject MRI/dMRI, keep a
whole-brain parcellation for structural coupling, and expose the hippocampal
target as a high-resolution local replacement. The relevant 2026 CA1 NEST-TVB
paper used HCP subject `100206`, a VEP-atlas high-resolution TVB/Spatial
Epileptor scaffold, tractography-derived connectivity, and hippocampal topology
mapping. This example keeps the current region-level reduced VEP owner but adds
scripts to build a whole-brain + CA3-subfield connectivity archive from
FreeSurfer and MRtrix outputs.

FreeSurfer's hippocampal CA hierarchy documents that CA2 is included in CA3, so
`Left-CA3` and `Right-CA3` should be interpreted as CA2/CA3 unless manually
refined.
