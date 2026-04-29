# Canonical Arbor-HHPlus Cosimulation Benchmark

This directory contains the single MIND_Sim cosimulation benchmark configuration
currently used for performance work. It intentionally fixes all numerical and
network parameters so future timing changes are not mixed with discarded
comparison configurations.

Fixed configuration:

- `cells = 100`
- `duration_ms = 2000.0`
- `dt_micro_ms = 0.01`
- `dt_macro_ms = 0.01`
- `batch_window_ms = 0.25`
- `connectivity = ~/arbor-tvb-cosim/connectivity_mouse.zip`
- `micro_roi = 72`
- `micro mechanism = hhplus`
- `pathological_fraction = 1.0`
- `k_bath_ok = 9.5`
- `k_bath_bad = 17.0`
- `recurrent_weight = 0.5`
- `recurrent_delay_ms = 0.5`
- `exposure_gain = 10.0`
- `exposure_tau_ms = 100.0`

Run against the current build tree:

```bash
cd /home/gluciferd/MIND_Sim
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mind_sim

PYTHONPATH=/home/gluciferd/MIND_Sim/build:/home/gluciferd/miniconda3/envs/mind_sim/lib/python3.11/site-packages \
python -S examples/cosim_arbor_hhplus/run_mind_sim.py
```

The script writes exactly one benchmark result path:

```text
result/cosim_arbor_hhplus/mind_sim_100cells_2s.h5
```

Use this file as the fixed result artifact for benchmark comparisons.
