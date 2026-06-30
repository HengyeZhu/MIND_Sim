import mind_sim as ms
from tests.micro.nmodl.transpiler.usecases.useion.style_ion import useion_mod_dir


def test_valence():
    ms.load_mech(useion_mod_dir())
    sim = ms.micro.sim()

    assert sim.ion_charge("K_ion") == 222.0
