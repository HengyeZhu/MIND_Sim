from __future__ import annotations

import numpy as np

import mind_sim as ms
from style_ion import single_section_sim, useion_mod_dir


nseg = 1

sim, group, seg = single_section_sim(useion_mod_dir())
group.insert("ionic")
sim.build_microcircuit()

ena_hoc = ms.Vector().record(seg._ref_ena)
t_hoc = ms.Vector().record(sim._ref_t)

sim.finitialize(-65.0)
sim.run(5.0)

ena = np.array(ena_hoc.to_python())
t = np.array(t_hoc.to_python())

ena_exact = np.full(t.shape, 42.0)
ena_exact[0] = 0.0

abs_err = np.abs(ena - ena_exact)
assert np.all(abs_err < 1e-12), abs_err
