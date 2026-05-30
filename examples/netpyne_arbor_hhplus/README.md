# NetPyNE Arbor-HHPlus Baseline

This directory contains the NetPyNE/NEURON baseline port of the fixed
`examples/cosim_arbor_hhplus` MIND_Sim benchmark.

The numerical configuration is intentionally fixed to the current MIND_Sim
benchmark口径:

- `duration_ms = 2000.0`
- `dt_ms = 0.01`
- `macro_dt_ms = 0.01`
- `exchange_window_ms = 0.25`
- `connectivity = ~/arbor-tvb-cosim/connectivity_mouse.zip`
- `micro_roi = 72`
- `pathological_fraction = 1.0`
- `k_bath_ok = 9.5`
- `k_bath_bad = 17.0`
- `recurrent_weight = 0.5`
- `recurrent_delay_ms = 0.5`
- `macro_event_weight = 0.01`
- `macro_event_delay_ms = 1.0`
- `exposure_gain = 10.0`
- `exposure_tau_ms = 100.0`

The micro network is created with NetPyNE cell construction, then the event
path is wired manually with NEURON `ParallelContext.gid_connect`:

- one reused `ExpSyn` per postsynaptic cell for all recurrent fan-in
- one reused `ExpSyn` per cell for macro input
- recurrent `NetCon`s are created via `pc.gid_connect`
- macro input uses source-less `NetCon(None, ExpSyn)` and explicit `netcon.event(t)`

Lightweight configuration checks do not compile MOD files or run simulation:

```bash
cd /home/gluciferd/MIND_Sim
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

python examples/netpyne_arbor_hhplus/run_netpyne.py --check-config --cells 100
python examples/netpyne_arbor_hhplus/run_netpyne.py --check-config --cells 1000
```

Before a real run, compile the local mechanisms in the `test` environment:

```bash
cd /home/gluciferd/MIND_Sim/examples/netpyne_arbor_hhplus/mod
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test
nrnivmodl
```

The run commands are intentionally explicit:

```bash
cd /home/gluciferd/MIND_Sim
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

python examples/netpyne_arbor_hhplus/run_netpyne.py --cells 100 --quiet
python examples/netpyne_arbor_hhplus/run_netpyne.py --cells 1000 --quiet
```

Default outputs:

```text
result/netpyne_arbor_hhplus/netpyne_100cells_2s.h5
result/netpyne_arbor_hhplus/netpyne_1000cells_2s.h5
```
