# HL23 Frontend Acceleration Example

This example builds four Human L2/3 cell types directly from SWC files through
MIND_Sim's Python frontend and compares against a CPU CoreNEURON baseline that
loads HOC files.

Default performance settings:

- 850 PYR, 50 SST, 50 PV, and 50 VIP cells
- `tstop = 50 ms`
- CPU only
- 4 threads

For performance comparisons, compile both MIND_Sim and CoreNEURON with NVHPC.
This enables compiler-side automatic optimization even when the benchmark runs
on CPU.

## Run

```bash
cd /home/gluciferd/MIND_Sim
bash examples/hl23_frontend_acceleration/allinone.sh
```

The workflow compiles mechanisms, runs MIND_Sim and CPU CoreNEURON
serially, and writes the performance comparison to:

`examples/hl23_frontend_acceleration/outputs/allinone_50ms/comparison.json`.
