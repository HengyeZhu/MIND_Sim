# CoreNEURON Source Integration

This directory contains copied CoreNEURON source code from the local NEURON tree:

- Upstream source path during import: `/home/gluciferd/nrn/src/coreneuron`
- Upstream license/copyright file: `/home/gluciferd/nrn/Copyright`
- Intended MIND_Sim role: embedded CPU CoreNEURON runtime source for direct copy-modify integration.

MIND_Sim does not treat this directory as a third-party binary dependency and does not call an external CoreNEURON runtime API. The source is kept here so MIND_Sim can adapt the CPU runtime and data layout internally while preserving CoreNEURON's open-source license requirements.
