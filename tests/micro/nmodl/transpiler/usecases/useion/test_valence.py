import mind_sim as ms
from tests.micro.nmodl.transpiler.usecases.useion.style_ion import useion_mod_dir


def test_valence():
    sim = ms.Sim()
    sim.load_mech(str(useion_mod_dir()))

    assert sim.ion_charge("K_ion") == 222.0
