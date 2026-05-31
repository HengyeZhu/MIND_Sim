# CA3 Epilepsy MIND Sim Cosim Example

This example vendors the CA3 epilepsy NEURON model from ModelDB 186768, rewrites its deterministic microcircuit through the MIND Sim micro API, validates the rewrite against the original ModelDB/NEURON implementation, and binds the CA3 microcircuit to a reduced VEP/Epileptor-style neural-mass scaffold.

Scientific target:

- macro ROI: `Left-CA3` or `Right-CA3` in a whole-brain atlas with hippocampal subfields
- macro model: reduced VEP/Epileptor-style neural mass, no neural field model
- micro model: ModelDB 186768 CA3 epilepsy network
- micro scale: 800 pyramidal cells, 200 basket cells, 200 OLM cells
- MIND Sim bridge: macro input drives PYR `Adend3AMPAf`; CA3 ROI output is aggregate PYR+BAS+OLM spike-derived activity

## Files

- `modeldb_186768/`: upstream ModelDB source at commit `e60d5e99eb2a6836485b35cc3f8aae0085bee79f`
- `mind_mod_ca3/`: selected CA3 channel and synapse mechanisms that compile through MIND Sim/CoreNEURON
- `mind_mod/`: MIND macro/coupling/micro bridge rules for the VEP-CA3 cosim
- `ca3_mind_sim_api.py`: MIND Sim API rewrite of the CA3 cells, synapses, and deterministic connectivity
- `run_ca3_original_validation.py`: original ModelDB/NEURON deterministic baseline
- `compare_ca3_validation.py`: voltage and spike comparison between original and MIND Sim
- `run_ca3_nmda_scan_validation.py`: batch original-vs-MIND NMDA/AMPA ratio scan
- `run_macro_only_validation.py`: MIND Sim reduced VEP macro-only validation against a NumPy reference
- `run_vep_ca3_cosim.py`: MIND Sim VEP-CA3 cosimulation
- `run_tvb_netpyne_multiscale_validation.py`: external `tvb-multiscale` NetPyNE-TVB interface sanity check
- `run_tvb_multiscale_vep_ca3_reference.py`: TVB-multiscale NetPyNE migration of the same reduced VEP + CA3 cosim
- `compare_mindsim_tvb_multiscale.py`: numerical comparison of MIND Sim and TVB-multiscale migrated cosim outputs
- `build_hybrid_ca3_parcellation.py`: FreeSurfer whole-brain + hippocampal CA3 subfield parcellation builder
- `connectome_csv_to_mindsim_npz.py`: MRtrix connectome CSV to MIND Sim `connectivity.npz` converter
- `WHOLE_BRAIN_CA3_PIPELINE.md`: HCP/FreeSurfer/MRtrix data-prep workflow for a whole-brain scaffold with CA3 ROI
- `run_ca3_neuron_export.py`: legacy direct-NEURON export helper

## Environment

```bash
conda activate test
cd examples/ca3_epilepsy_cosim
export PYTHONPATH=/home/gluciferd/MIND_Sim/build:/home/gluciferd/MIND_Sim/src/python_api:$PYTHONPATH
export MIND_SIM_CODEGEN_CACHE=/tmp/mind_sim_codegen
```

Compile the original ModelDB mechanisms only for the NEURON baseline:

```bash
cd modeldb_186768
nrnivmodl *.mod
cd ..
```

The MIND Sim rewrite loads mechanisms from `mind_mod_ca3/`; it does not use the old utility mods (`stats.mod`, `vecst.mod`, `wrap.mod`, `misc.mod`) that are present in the upstream directory.

## Validate The Rewrite

Run the original 200 ms deterministic full-connectivity baseline:

```bash
./modeldb_186768/x86_64/special -python run_ca3_original_validation.py \
  --duration-ms 200 \
  --connections \
  --output outputs/ca3_original_200ms_conn.npz
```

Run the MIND Sim API rewrite:

```bash
python ca3_mind_sim_api.py \
  --duration-ms 200 \
  --connections \
  --device cpu \
  --output outputs/ca3_mindsim_200ms_conn.npz
```

Compare:

```bash
python compare_ca3_validation.py \
  outputs/ca3_original_200ms_conn.npz \
  outputs/ca3_mindsim_200ms_conn.npz \
  --output outputs/ca3_validate_200ms_conn.json
```

Current local validation result:

- `global_max_abs_mv`: `8.113687499644584e-11`
- `global_rms_mv`: `1.099467307623214e-12`
- `original_spikes`: `200`
- `mind_sim_spikes`: `200`
- `spike_gid_equal`: `true`
- `spike_time_max_abs_ms`: `0.0`

## Validate Paper-Protocol Micro Inputs

The upstream paper-style micro-only protocol can now be enabled on both runners:

- recurrent CA3 connectivity
- background `NetStim` inputs
- deterministic medial septal rhythmic input
- 5 s default duration
- optional washin/washout of OLM NMDA ratio

For numerical validation, disable stochastic background noise. In this mode the
MIND Sim runner uses direct micro-level `NetStim(noise=0)` artificial cells and
`network.event_connect(...)` into the same synapses as the original NEURON
runner; it does not use the `MICRO_INPUT` bridge generator.

Run the original deterministic NEURON protocol:

```bash
./modeldb_186768/x86_64/special -python run_ca3_original_validation.py \
  --paper-protocol \
  --no-background-noise \
  --output outputs/ca3_original_paper_protocol_5s_periodic_micro_direct.npz
```

Run the deterministic MIND Sim rewrite:

```bash
python ca3_mind_sim_api.py \
  --paper-protocol \
  --no-background-noise \
  --device cpu \
  --output outputs/ca3_mindsim_paper_protocol_5s_periodic_micro_direct.npz
```

Compare spike counts, population rates, binned spike counts, and recorded
voltages:

```bash
python compare_ca3_validation.py \
  outputs/ca3_original_paper_protocol_5s_periodic_micro_direct.npz \
  outputs/ca3_mindsim_paper_protocol_5s_periodic_micro_direct.npz \
  --output outputs/ca3_validate_paper_protocol_5s_periodic_micro_direct.json
```

For a faster exact event-raster smoke check:

```bash
./modeldb_186768/x86_64/special -python run_ca3_original_validation.py \
  --duration-ms 20 \
  --connections \
  --background-inputs \
  --no-background-noise \
  --medial-septal \
  --output outputs/ca3_original_20ms_periodic_micro_direct.npz

python ca3_mind_sim_api.py \
  --duration-ms 20 \
  --connections \
  --background-inputs \
  --no-background-noise \
  --medial-septal \
  --device cpu \
  --output outputs/ca3_mindsim_20ms_periodic_micro_direct.npz
```

Current local periodic-background check:

- 20 ms smoke:
  - `original_spikes`: `1200`
  - `mind_sim_spikes`: `1200`
  - population counts: `PYR=800`, `BAS=200`, `OLM=200` on both
  - `spike_gid_equal`: `true`
  - `spike_time_max_abs_ms`: `0.0`
  - `global_max_abs_mv`: `2.892264205911488e-10`
- 5 s paper protocol:
  - `original_spikes`: `52400`
  - `mind_sim_spikes`: `52400`
  - population counts: `PYR=17600`, `BAS=30400`, `OLM=4400` on both
  - `spike_gid_equal`: `true`
  - `spike_time_max_abs_ms`: `0.0`
  - `global_max_abs_mv`: `7.489099118629383e-09`
  - `original_run_s`: `89.938633`
  - `mind_sim_run_s`: `45.714104`

For stochastic background noise, the original uses NEURON `NetStim` random
streams and the MIND runner still uses the generated MIND input process through
the macro/micro bridge, so the comparison is statistical rather than bitwise
exact. The stochastic bridge currently records final selected voltages plus
spikes; the direct deterministic micro runner records full voltage trajectories.
MIND washin/washout is implemented inside the CA3 `MyExp2SynNMDABB` synapse copy
by making the NMDA/AMPA ratio used in `NET_RECEIVE` time dependent. This is
mathematically equivalent to the original segmented NEURON parameter update
because `r` is read only when an event arrives.

## Scan NMDA/AMPA Ratios

Run a small original-vs-MIND scan over the four paper ratio parameters:
`olm_somaNMDA`, `bas_somaNMDA`, `pyr_BdendNMDA`, and `pyr_Adend3NMDA`.

```bash
python run_ca3_nmda_scan_validation.py \
  --values 0,1 \
  --duration-ms 20 \
  --connections \
  --no-background-inputs \
  --output-dir outputs/nmda_scan_20ms
```

This runs `2^4 = 16` combinations, writes one original NPZ, one MIND NPZ, one
comparison JSON per combination, and writes
`outputs/nmda_scan_20ms/ca3_nmda_scan_summary.json`.

Use per-parameter grids when needed:

```bash
python run_ca3_nmda_scan_validation.py \
  --olm-values 0,0.5,1 \
  --bas-values 1 \
  --pyr-bdend-values 1 \
  --pyr-adend3-values 1 \
  --duration-ms 200 \
  --connections \
  --output-dir outputs/nmda_scan_olm
```

## Validate Macro Only

Run the reduced VEP neural-mass model without any micro replacement and compare MIND Sim `MacroSimulator` to a NumPy reference:

```bash
python run_macro_only_validation.py \
  --duration-ms 200 \
  --macro-dt-ms 0.1 \
  --exchange-window-ms 10 \
  --output outputs/macro_only_validation.npz \
  --report outputs/macro_only_validation.json
```

Current local macro-only validation result:

- `passed`: `true`
- `x_max_abs`: `5.551115123125783e-16`
- `z_max_abs`: `1.1102230246251565e-16`
- `time_max_abs_ms`: `0.0`

## Validate The NetPyNE-TVB Interface

This runs the local `tvb-multiscale` NetPyNE-TVB serial orchestrator. It is an
interface sanity check, not a CA3 numerical-precision reference, because the
upstream example uses `ReducedWongWangExcIOInhI` plus `DefaultExcIOInhIBuilder`
rather than the reduced VEP plus ModelDB 186768 CA3 microcircuit used above.

```bash
python run_tvb_netpyne_multiscale_validation.py \
  --duration-ms 20 \
  --population-order 2 \
  --spiking-proxy-inds 0 \
  --output outputs/tvb_netpyne_multiscale_validation.npz \
  --report outputs/tvb_netpyne_multiscale_validation.json
```

The JSON report explicitly marks `comparison_to_mind_vep_ca3.same_model` as
`false`; a valid MIND-vs-NetPyNE precision comparison needs the same macro
equations, same CA3 microcircuit, same coupling transforms, same seeds, and same
initial conditions on both sides.

## Validate TVB-Multiscale Migration

This migrates the same reduced VEP macro scaffold and ModelDB 186768 CA3 micro
network to a TVB-multiscale NetPyNE reference. It uses
`tvb_multiscale.tvb_netpyne.netpyne.module.NetpyneModule` for the CA3
microcircuit, preserves the MIND Sim macro equations and 10 ms exchange-window
bridge semantics, and writes the same output schema as `run_vep_ca3_cosim.py`.

```bash
python run_tvb_multiscale_vep_ca3_reference.py \
  --duration-ms 200 \
  --connections \
  --quiet \
  --output outputs/tvb_multiscale_vep_ca3_reference_200ms.npz

python compare_mindsim_tvb_multiscale.py \
  outputs/vep_ca3_mindsim_200ms.npz \
  outputs/tvb_multiscale_vep_ca3_reference_200ms.npz \
  --atol 1e-12 \
  --output outputs/mindsim_vs_tvb_multiscale_200ms.json
```

Current local 200 ms comparison result:

- `passed`: `true`
- `macro_x max_abs`: `2.3869795029440866e-15`
- `macro_z max_abs`: `2.6645352591003757e-15`
- `ca3_macro_input max_abs`: `6.661338147750939e-16`
- `ca3_drive_hz max_abs`: `2.1316282072803006e-14`
- `spike_times_ms exact_equal`: `true`
- `spike_gids exact_equal`: `true`

## Run VEP-CA3 Cosimulation

```bash
python run_vep_ca3_cosim.py \
  --duration-ms 200 \
  --exchange-window-ms 10 \
  --macro-dt-ms 0.1 \
  --device cpu \
  --output outputs/vep_ca3_mindsim_200ms.npz
```

For a subject-specific VEP-HC-style scaffold, pass a NumPy archive with:

- `labels`: ROI labels containing the chosen CA3 label
- `weights`: target-by-source connectivity matrix
- `delays`: optional target-by-source delays in ms; if absent, positive weights use `--exchange-window-ms`

```bash
python run_vep_ca3_cosim.py \
  --connectivity-npz /path/to/vep_hc_connectivity.npz \
  --ca3-label Left-CA3 \
  --output outputs/subject_left_ca3_mindsim_cosim.npz
```

To build such a connectivity archive from HCP/FreeSurfer/MRtrix data, follow
`WHOLE_BRAIN_CA3_PIPELINE.md`. The short form is:

```bash
python build_hybrid_ca3_parcellation.py \
  --base-parcellation "$SUBJECTS_DIR/100206/mri/aparc+aseg.mgz" \
  --left-hippo-labels "$SUBJECTS_DIR/100206/mri/lh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz" \
  --right-hippo-labels "$SUBJECTS_DIR/100206/mri/rh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz" \
  --lut "$FREESURFER_HOME/FreeSurferColorLUT.txt" \
  --output-volume outputs/100206_hybrid_ca3.mgz \
  --output-labels outputs/100206_hybrid_ca3_labels.tsv

python connectome_csv_to_mindsim_npz.py \
  --weights-csv outputs/100206_hybrid_ca3_weights.csv \
  --lengths-csv outputs/100206_hybrid_ca3_lengths.csv \
  --labels-tsv outputs/100206_hybrid_ca3_labels.tsv \
  --ca3-label Left-CA3 \
  --output outputs/100206_hybrid_ca3_connectivity.npz
```

## Output Schema

`run_vep_ca3_cosim.py` writes:

- `labels`, `weights`, `delays`
- `time_ms`
- `macro_x`, `macro_z`
- `ca3_macro_input`
- `ca3_drive_hz`
- `ca3_macro_output_x`, `ca3_macro_output_z`
- `ca3_micro_rate_proxy_hz`
- `ca3_micro_population_rate_hz`, `rate_time_ms`
- `spike_times_ms`, `spike_gids`
- `metadata_json`

These arrays are the ROI input/output traces intended for later micro-parameter optimization.
