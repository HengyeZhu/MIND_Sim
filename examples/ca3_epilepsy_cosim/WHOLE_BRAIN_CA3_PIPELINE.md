# Whole-Brain + CA3 Subfield Pipeline

This is the data-prep path for replacing the current 5-ROI toy scaffold with a
subject-specific whole-brain macro model that exposes CA3 as a separate ROI.

The design follows the recent NEST-TVB hippocampus cosimulation pattern:

- build the macro model from subject MRI/dMRI rather than a hand-written 5x5 matrix;
- use a whole-brain parcellation for the TVB/VEP-scale structural scaffold;
- expose the hippocampal target region as a high-resolution local replacement;
- exchange rate-coded macro input and aggregate micro output across scales.

The 2026 CA1 NEST-TVB paper used HCP subject `100206`, a VEP-atlas
high-resolution TVB/Spatial Epileptor scaffold, tractography-derived
connectivity, and a hippocampal micro model mapped onto the local hippocampal
topology. For this CA3/NEURON example we keep the same macro-data logic but use
FreeSurfer hippocampal subfields to split out `Left-CA3` / `Right-CA3` and bind
the MIND Sim CA3 microcircuit to that ROI.

## Inputs

You need a subject with:

- FreeSurfer whole-brain output, usually `mri/aparc+aseg.mgz`;
- FreeSurfer hippocampal subfields, preferably the CA hierarchy in FS voxel
  space:
  - `mri/lh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz`
  - `mri/rh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz`
- diffusion tractography in the same anatomical space, e.g. MRtrix `.tck`;
- a 5TT/ACT setup if you want production-quality tractography.

HCP data are suitable, but the raw/preprocessed HCP downloads require an HCP/NDA
account and data-use agreement, so this repository does not vendor the MRI/dMRI
data.

## 1. FreeSurfer

If the subject is not already processed:

```bash
recon-all -all -s 100206
segmentHA_T1.sh 100206 "$SUBJECTS_DIR"
```

FreeSurfer notes:

- The official hippocampal-subfield tool requires a completed `recon-all`.
- The `.CA.FSvoxelSpace.mgz` hierarchy is useful here because it merges internal
  molecular/GC-DG labels into CA fields and lives in the normal FreeSurfer voxel
  space.
- FreeSurfer documents that CA2 is included in CA3, so our `CA3` ROI should be
  interpreted as `CA2/CA3` unless manually refined.

## 2. Build A Whole-Brain + CA3 Label Volume

```bash
python build_hybrid_ca3_parcellation.py \
  --base-parcellation "$SUBJECTS_DIR/100206/mri/aparc+aseg.mgz" \
  --left-hippo-labels "$SUBJECTS_DIR/100206/mri/lh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz" \
  --right-hippo-labels "$SUBJECTS_DIR/100206/mri/rh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz" \
  --lut "$FREESURFER_HOME/FreeSurferColorLUT.txt" \
  --output-volume outputs/100206_hybrid_ca3.mgz \
  --output-labels outputs/100206_hybrid_ca3_labels.tsv \
  --output-metadata outputs/100206_hybrid_ca3_metadata.json
```

The output volume has contiguous integer labels, so MRtrix does not create a
huge sparse connectome indexed by raw FreeSurfer label IDs. The whole-brain
parcellation is preserved, but the left and right hippocampus labels are split
into residual non-CA3 hippocampus plus explicit `Left-CA3` and `Right-CA3`.

If a FreeSurfer release uses different CA3 label values, pass them explicitly:

```bash
python build_hybrid_ca3_parcellation.py ... \
  --left-ca3-values 208 \
  --right-ca3-values 208
```

## 3. Tractography Connectome

Example MRtrix commands; adapt the preprocessing to the actual HCP diffusion
layout you use.

```bash
mrconvert outputs/100206_hybrid_ca3.mgz outputs/100206_hybrid_ca3.nii.gz

tck2connectome tracks.tck \
  outputs/100206_hybrid_ca3.nii.gz \
  outputs/100206_hybrid_ca3_weights.csv \
  -zero_diagonal \
  -symmetric

tck2connectome tracks.tck \
  outputs/100206_hybrid_ca3.nii.gz \
  outputs/100206_hybrid_ca3_lengths.csv \
  -scale_length \
  -stat_edge mean \
  -zero_diagonal \
  -symmetric
```

This mirrors the NEST-TVB paper at the conceptual level: macro connectivity is
tractography-derived, while the hippocampal target is represented at finer
scale. We are region-based rather than vertex-based in the current MIND Sim
example; moving to the paper's vertex/SEM setup would be a separate macro-model
upgrade.

## 4. Convert To MIND Sim Connectivity

```bash
python connectome_csv_to_mindsim_npz.py \
  --weights-csv outputs/100206_hybrid_ca3_weights.csv \
  --lengths-csv outputs/100206_hybrid_ca3_lengths.csv \
  --labels-tsv outputs/100206_hybrid_ca3_labels.tsv \
  --ca3-label Left-CA3 \
  --output outputs/100206_hybrid_ca3_connectivity.npz \
  --metadata-json outputs/100206_hybrid_ca3_connectivity.json
```

Then run the cosimulation with the real whole-brain scaffold:

```bash
python run_vep_ca3_cosim.py \
  --connectivity-npz outputs/100206_hybrid_ca3_connectivity.npz \
  --ca3-label Left-CA3 \
  --duration-ms 2000 \
  --exchange-window-ms 10 \
  --macro-dt-ms 0.1 \
  --device cpu \
  --output outputs/100206_left_ca3_mindsim_cosim.npz
```

## Current Limitations

- The current macro equations are the reduced region-level VEP-like model, not
  the 6D Spatial Epileptor Model on cortical/hippocampal mesh vertices used in
  the 2026 NEST-TVB CA1 paper.
- The current bridge is bidirectional at ROI level: macro input drives CA3
  external AMPA events, and CA3 spikes drive the CA3 ROI output.
- dMRI estimates for CA3 are small-ROI tractography estimates. They are usable
  as a scaffold, but they should be treated as noisy and should be sensitivity
  tested against thresholding, streamline count normalization, and conduction
  speed.

## References

- Tartarini et al. 2026, *Co-simulation framework combining a microscopically
  detailed point neuron model of the hippocampal CA1 region with the macroscopic
  high-resolution virtual brain model*, Journal of Computational Neuroscience,
  https://doi.org/10.1007/s10827-026-00925-w
- Kusch et al. 2024, *Multiscale co-simulation design pattern for neuroscience
  applications*, Frontiers in Neuroinformatics,
  https://doi.org/10.3389/fninf.2024.1156683
- FreeSurfer hippocampal subfields and amygdala nuclei documentation,
  https://freesurfer.net/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala
