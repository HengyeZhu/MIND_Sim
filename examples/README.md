# Examples

Run commands from the repository root:

```bash
cd MIND_Sim
```

## Prepare MIND_Sim

```bash
conda create -n mind_sim python=3.11 -y
conda activate mind_sim
conda install -c conda-forge cmake ninja bison flex -y
pip install -e .
```

Compile the MIND_Sim mechanisms:

```bash
conda activate mind_sim
mind_nrnivmodl examples/ca3_epilepsy_cosim/mind_sim/mod
ldd examples/ca3_epilepsy_cosim/mind_sim/mod/x86_64/libcorenrnmech.so | grep -E 'libnvc|libnvomp|libnvhpc|libacc'
```

## Prepare TVB+NEURON

Create the reference environment:

```bash
conda create -n test python=3.11 -y
conda activate test
pip install numpy scipy matplotlib networkx numba numexpr six Deprecated mako docutils h5py traitlets neuron
```

Download TVB:

```bash
git clone https://github.com/the-virtual-brain/tvb-root.git ~/tvb-root
```

Check the reference imports:

```bash
conda activate test
python -c "import numpy, matplotlib, neuron"
python -c "import sys, pathlib; root = pathlib.Path.home() / 'tvb-root'; [sys.path.insert(0, str(root / name)) for name in ['tvb_library', 'tvb_contrib', 'tvb_storage', 'tvb_framework']]; from tvb.simulator.history import SparseHistory; from tvb.simulator.models.epileptor import Epileptor2D"
```

Compile the TVB+NEURON mechanisms:

```bash
conda activate test
cd examples/ca3_epilepsy_cosim/neuron_tvb/mod
nrnivmodl .
cd -
```

## Public CA3 Run

Use the included synthetic connectivity:

```text
examples/ca3_epilepsy_cosim/data/synthetic_hybrid_ca3_connectivity.csv
```

Run MIND_Sim, TVB+NEURON, comparison, and plots:

```bash
DURATION_MS=1000 ./examples/ca3_epilepsy_cosim/allinone.sh
```

The all-in-one workflow rebuilds MIND_Sim mechanisms with `mind_nrnivmodl` and TVB+NEURON mechanisms with the `nrnivmodl` command from the `test` environment.

Run only the comparison step on existing outputs:

```bash
RUN_EXPERIMENTS=0 DURATION_MS=1000 ./examples/ca3_epilepsy_cosim/allinone.sh
```

Set a custom output directory:

```bash
OUTDIR=examples/ca3_epilepsy_cosim/outputs/my_run DURATION_MS=1000 ./examples/ca3_epilepsy_cosim/allinone.sh
```

Generated files:

```text
examples/ca3_epilepsy_cosim/outputs/allinone_1000ms/
examples/ca3_epilepsy_cosim/outputs/allinone_1000ms/compare_1000ms.json
examples/ca3_epilepsy_cosim/outputs/allinone_1000ms/compare_1000ms.md
examples/ca3_epilepsy_cosim/outputs/allinone_1000ms/plots/
```

## Public Data

Included:

```text
examples/ca3_epilepsy_cosim/data/synthetic_hybrid_ca3_connectivity.csv
examples/ca3_epilepsy_cosim/data/freesurfer_color_lut.tsv
examples/ca3_epilepsy_cosim/data/hippunfold_multihist7_subfields.tsv
examples/ca3_epilepsy_cosim/mind_sim/mod/
examples/ca3_epilepsy_cosim/neuron_tvb/mod/
```

Download:

```text
~/tvb-root
```

No HCP download is needed for the public synthetic run.

## HCP 100206 Data

Place HCP subject data here:

```text
examples/HCP_1200/100206/
```

Prepare a data environment:

```bash
conda create -n hcp_prep python=3.11 -y
conda activate hcp_prep
conda install -c conda-forge numpy nibabel -y
```

Check required files:

```bash
conda activate hcp_prep
python examples/ca3_epilepsy_cosim/mind_sim/prepare_hcp100206_ca3.py check \
  --hcp-root examples/HCP_1200 \
  --subject 100206
```

Create BIDS symlinks for HippUnfold:

```bash
python examples/ca3_epilepsy_cosim/mind_sim/prepare_hcp100206_ca3.py prepare-bids \
  --hcp-root examples/HCP_1200 \
  --subject 100206 \
  --output-bids examples/ca3_epilepsy_cosim/outputs/hcp_bids \
  --replace
```

Use these inputs for the real-connectivity path:

```text
examples/HCP_1200/100206/T1w/T1w_acpc_dc_restore.nii.gz
examples/HCP_1200/100206/T1w/T2w_acpc_dc_restore.nii.gz
examples/HCP_1200/100206/T1w/aparc+aseg.nii.gz
examples/HCP_1200/100206/T1w/wmparc.nii.gz
examples/ca3_epilepsy_cosim/outputs/hippunfold/sub-100206/anat/
MRtrix weights CSV
MRtrix lengths CSV
ROI labels TSV
```
