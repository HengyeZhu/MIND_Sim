from __future__ import annotations

import numpy as np

import mind_sim as ms




def _copy(src, result, *args, dest=None):
    if dest is None:
        dest = ms.Vector()
    dest.copy(src, *args)
    assert dest.to_python() == result


def _vector_construction_and_basic_mutation_style():
    vi = ms.Vector((1, 2, 3))
    assert vi.to_python() == [1.0, 2.0, 3.0]
    assert vi.get(1) == 2.0
    vi.set(1, 2.1)
    assert vi.to_python() == [1.0, 2.1, 3.0]

    v = ms.Vector(np.array([5, 1, 6], "d"))
    assert v.to_python() == [5.0, 1.0, 6.0]
    v.clear()
    assert v.size() == 0

    assert ms.Vector(3).to_python() == [0.0, 0.0, 0.0]
    assert ms.Vector(3, 1).to_python() == [1.0, 1.0, 1.0]
    assert ms.Vector().from_python((1, 2, 3)).to_python() == [1.0, 2.0, 3.0]

    v = ms.Vector()
    v.append(3, 3)
    v.append(2)
    v.append(1)
    v.append(5)

    assert v.to_python() == [3.0, 3.0, 2.0, 1.0, 5.0]
    assert v.size() == 5
    v.buffer_size(v.size())
    assert v.buffer_size() >= v.size()
    v.buffer_size(6)
    assert v.buffer_size() >= 6
    assert v.eq(v.c())


def _vector_statistics_style():
    v = ms.Vector([3, 3, 2, 1, 5])
    v3 = ms.Vector([4, 2, 61, 17, 13])

    assert v.median() == 3.0
    assert v.mean() == 2.8
    assert v.mean(1, 3) == 2.0
    assert np.allclose(v.stdev(), 1.4832396974191326)
    assert np.allclose(v.stdev(0, 3), 0.9574271077563381)
    assert np.allclose(v.stderr(), 0.6633249580710799)
    assert np.allclose(v.stderr(1, 3), 0.5773502691896258)
    assert v.sum() == 14.0
    assert v.sum(1, 3) == 6.0
    assert v.sumsq() == 48.0
    assert v.sumsq(2, 4) == 30.0
    assert v.var() == 2.2
    assert v.var(2, 3) == 0.5
    assert v.min() == 1.0
    assert v.min(0, 2) == 2.0
    assert ms.Vector().min() == 0.0
    assert v.min_ind() == 3
    assert v.min_ind(0, 2) == 2
    assert ms.Vector().min_ind() == -1.0
    assert v.max() == 5.0
    assert v.max(0, 2) == 3.0
    assert ms.Vector().max() == 0.0
    assert v.max_ind() == 4
    assert v.max_ind(0, 2) == 0
    assert v3.max_ind(2, 4) == 2
    assert v3.max_ind(1, 2) == 2
    assert v3.max_ind(3, 4) == 3
    assert ms.Vector().max_ind() == -1.0
    assert v.dot(ms.Vector((1, 2, 3, 4, 5))) == 44.0
    assert np.allclose(v.mag(), 6.928203230275509)
    assert v.c().reverse().meansqerr(v) == 3.2
    assert v.c().reverse().meansqerr(v, ms.Vector((1, 2, 5, 4, 3))) == 8.0


def _vector_copy_and_data_morphing_style():
    v = ms.Vector([3, 3, 2, 1, 5])

    _copy(v, [3.0, 3.0, 2.0, 1.0, 5.0])
    _copy(v, [0.0, 0.0, 3.0, 3.0, 2.0, 1.0, 5.0], 2)
    _copy(v, [2.0, 1.0], 2, 3)
    _copy(v, [2.0, 1.0, 5.0], 2, -1)
    _copy(v, [0.0, 2.0, 1.0], 1, 2, 3)
    _copy(v, [0.0, 3.0, 2.0], ms.Vector((1, 2)), dest=ms.Vector(3, 0.0))
    _copy(
        v,
        [3.0, 2.0, 0.0],
        ms.Vector((1, 2)),
        ms.Vector((0, 1)),
        dest=ms.Vector(3, 0.0),
    )

    assert v.c().to_python() == v.to_python()
    assert v.c(1).to_python() == [3.0, 2.0, 1.0, 5.0]
    assert v.c(1, 3).to_python() == [3.0, 2.0, 1.0]
    assert v.at().to_python() == v.to_python()
    assert v.at(1).to_python() == [3.0, 2.0, 1.0, 5.0]
    assert v.at(1, 3).to_python() == [3.0, 2.0, 1.0]

    assert v.resize(4).size() == 4
    assert v.fill(1.0).to_python() == [1.0, 1.0, 1.0, 1.0]
    assert v.fill(1.1, 1, 2).to_python() == [1.0, 1.1, 1.1, 1.0]
    assert v.indgen().to_python() == [0.0, 1.0, 2.0, 3.0]
    assert v.indgen(2).to_python() == [0.0, 2.0, 4.0, 6.0]
    assert v.indgen(2, 5).to_python() == [2.0, 7.0, 12.0, 17.0]
    assert v.indgen(1, 20, 5).to_python() == [1.0, 6.0, 11.0, 16.0]
    assert v.append(ms.Vector(1, 17.0), 18.0, 19.0).to_python() == [
        1.0,
        6.0,
        11.0,
        16.0,
        17.0,
        18.0,
        19.0,
    ]
    assert v.insrt(1, 3.0).to_python() == [1.0, 3.0, 6.0, 11.0, 16.0, 17.0, 18.0, 19.0]
    assert v.insrt(3, ms.Vector(1, 9.0)).to_python() == [
        1.0,
        3.0,
        6.0,
        9.0,
        11.0,
        16.0,
        17.0,
        18.0,
        19.0,
    ]
    assert v.remove(4).to_python() == [1.0, 3.0, 6.0, 9.0, 16.0, 17.0, 18.0, 19.0]
    assert v.remove(1, 5).to_python() == [1.0, 18.0, 19.0]


def _vector_where_and_transform_style():
    v = ms.Vector().indgen(1, 30, 5)
    assert v.to_python() == [1.0, 6.0, 11.0, 16.0, 21.0, 26.0]
    assert v.contains(6.0)
    assert not v.contains(7.0)
    assert ms.Vector().where(v, ">=", 10).to_python() == [11.0, 16.0, 21.0, 26.0]
    assert ms.Vector().where(v, "<=", 11).to_python() == [1.0, 6.0, 11.0]
    assert ms.Vector().where(v, "!=", 11).to_python() == [1.0, 6.0, 16.0, 21.0, 26.0]
    assert ms.Vector().where(v, "==", 11).to_python() == [11.0]
    assert ms.Vector().where(v, "<", 11).to_python() == [1.0, 6.0]
    assert ms.Vector().where(v, ">", 11).to_python() == [16.0, 21.0, 26.0]
    assert ms.Vector().where(v, "[)", 9, 21).to_python() == [11.0, 16.0]
    assert ms.Vector().where(v, "[]", 9, 21).to_python() == [11.0, 16.0, 21.0]
    assert ms.Vector().where(v, "(]", 11, 21).to_python() == [16.0, 21.0]
    assert ms.Vector().where(v, "()", 11, 21).to_python() == [16.0]
    assert v.where(">", 1.0).to_python() == [6.0, 11.0, 16.0, 21.0, 26.0]
    assert v.where("[)", 6.0, 26.0).to_python() == [6.0, 11.0, 16.0, 21.0]
    assert v.indwhere(">", 11.0) == 2
    assert v.indwhere("<", 11.0) == 0
    assert v.indwhere("!=", 11.0) == 0
    assert v.indwhere(">=", 11.0) == 1
    assert v.indwhere("<=", 11.0) == 0
    assert v.indwhere("[)", 11.1, 16.0) == -1
    assert v.indwhere("[)", 11.0, 16.0) == 1
    assert v.indwhere("(]", 11.0, 16.0) == 2
    assert v.indwhere("[]", 11.0, 16.0) == 1
    assert v.indwhere("()", 16.0, 11.0) == -1
    assert ms.Vector().indvwhere(v, "()", 11, 21).to_python() == [2.0]
    assert ms.Vector().indvwhere(v, "==", 11).to_python() == [1.0]
    assert ms.Vector().indvwhere(v, "[]", 1, 17).to_python() == [0.0, 1.0, 2.0]
    assert ms.Vector().indvwhere(v, "(]", 1, 16).to_python() == [0.0, 1.0, 2.0]
    assert ms.Vector().indvwhere(v, "[)", 1, 16).to_python() == [0.0, 1.0]
    assert ms.Vector().indvwhere(v, "!=", 11).to_python() == [0.0, 2.0, 3.0]
    assert ms.Vector().indvwhere(v, "<", 11).to_python() == [0.0]
    assert ms.Vector().indvwhere(v, "<=", 11).to_python() == [0.0, 1.0]
    assert ms.Vector().indvwhere(v, ">", 16).to_python() == [3.0]
    assert ms.Vector().indvwhere(v, ">=", 16).to_python() == [2.0, 3.0]
    assert v.histogram(1.0, 20.0, 10).to_python() == [0.0, 1.0, 2.0]
    assert ms.Vector().hist(v, 1.0, 2.0, 10).to_python() == [1.0, 2.0]
    assert v.ind(ms.Vector((1, 3))).to_python() == [11.0, 21.0]
    assert ms.Vector().spikebin(v.c(), 12.0).to_python() == [0.0, 0.0, 1.0, 1.0]
    assert v.label() == ""
    v.label("v")
    assert v.label() == "v"
    assert v.cl().label() == "v"
    v.label("v2")
    assert v.label() == "v2"

    x = ms.Vector((3, 2, 15, 16))
    assert np.allclose(
        x.c().apply("sin").to_python(),
        [
            0.1411200080598672,
            0.9092974268256817,
            0.6502878401571169,
            -0.2879033166650653,
        ],
    )
    assert np.allclose(x.c().apply("sin", 1, 2).to_python(), [3.0, 0.9092974268256817, 0.6502878401571169, 16.0])
    assert np.allclose(x.c().apply("sq").to_python(), [9.0, 4.0, 225.0, 256.0])
    assert x.reduce("sq", 100) == 594.0
    assert ms.Vector().deriv(x, 0.1).to_python() == [-10.0, 60.0, 70.0, 10.0]
    assert ms.Vector().deriv(x, 1, 1).to_python() == [-1.0, 13.0, 1.0]
    assert ms.Vector().deriv(x, 1, 2).to_python() == [-1.0, 6.0, 7.0, 1.0]
    assert ms.Vector().integral(x).to_python() == [3.0, 5.0, 20.0, 36.0]
    assert np.allclose(ms.Vector().integral(x, 0.1).to_python(), [3.0, 3.2, 4.7, 6.3])
    assert x.c().medfltr().to_python() == [3.0, 3.0, 3.0, 3.0]
    assert x.c().medfltr(ms.Vector((1, 2, 3, 4))).to_python() == [2.0, 2.0, 2.0, 2.0]
    assert x.c().sort().to_python() == [2.0, 3.0, 15.0, 16.0]
    assert x.sortindex().to_python() == [1.0, 0.0, 2.0, 3.0]
    assert x.c().reverse().to_python() == [16.0, 15.0, 2.0, 3.0]
    assert x.c().rotate(3).to_python() == [2.0, 15.0, 16.0, 3.0]
    assert x.c().rotate(3, 0).to_python() == [0.0, 0.0, 0.0, 3.0]
    assert ms.Vector().rebin(x, 2).to_python() == [5.0, 31.0]
    assert x.c().rebin(2).to_python() == [5.0, 31.0]
    assert ms.Vector().pow(x, 2).to_python() == [9.0, 4.0, 225.0, 256.0]
    assert np.allclose(x.c().pow(x, 0).to_python(), [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(x.c().pow(x, 0.5).to_python(), [1.7320508075688772, 1.4142135623730951, 3.872983346207417, 4.0])
    assert np.allclose(x.c().pow(x, -1).to_python(), [0.3333333333333333, 0.5, 0.06666666666666667, 0.0625])
    assert np.allclose(x.c().pow(x, 1).to_python(), [3.0, 2.0, 15.0, 16.0])
    assert np.allclose(x.c().pow(x, 3).to_python(), [27.0, 8.0, 3375.0, 4096.0])
    assert x.c().pow(2).to_python() == [9.0, 4.0, 225.0, 256.0]
    assert np.allclose(ms.Vector().sqrt(x).to_python(), [1.7320508075688772, 1.4142135623730951, 3.872983346207417, 4.0])
    assert np.allclose(x.c().sqrt().to_python(), [1.7320508075688772, 1.4142135623730951, 3.872983346207417, 4.0])
    assert np.allclose(ms.Vector().log(x).to_python(), [1.0986122886681098, 0.6931471805599453, 2.70805020110221, 2.772588722239781])
    assert np.allclose(x.c().log().to_python(), [1.0986122886681098, 0.6931471805599453, 2.70805020110221, 2.772588722239781])
    assert np.allclose(ms.Vector().log10(x).to_python(), [0.47712125471966244, 0.3010299956639812, 1.1760912590556813, 1.2041199826559248])
    assert np.allclose(x.c().log10().to_python(), [0.47712125471966244, 0.3010299956639812, 1.1760912590556813, 1.2041199826559248])
    assert np.allclose(ms.Vector().tanh(x).to_python(), [0.9950547536867305, 0.9640275800758169, 0.9999999999998128, 0.9999999999999747])
    assert np.allclose(x.c().tanh().to_python(), [0.9950547536867305, 0.9640275800758169, 0.9999999999998128, 0.9999999999999747])
    assert x.c().add(ms.Vector((1.1, 2.2, 3.3, 4.4))).to_python() == [4.1, 4.2, 18.3, 20.4]
    assert x.c().add(1.3).to_python() == [4.3, 3.3, 16.3, 17.3]
    assert x.c().sub(ms.Vector((1.1, 2, 3.3, 4))).to_python() == [1.9, 0.0, 11.7, 12.0]
    assert x.c().sub(1.3).to_python() == [1.7, 0.7, 13.7, 14.7]
    assert x.c().mul(ms.Vector((1.5, 2, 3, 4))).to_python() == [4.5, 4.0, 45.0, 64.0]
    assert x.c().mul(2.5).to_python() == [7.5, 5.0, 37.5, 40.0]
    assert x.c().div(ms.Vector((1.5, 2, 3, 4))).to_python() == [2.0, 1.0, 5.0, 4.0]
    assert x.c().div(2.5).to_python() == [1.2, 0.8, 6.0, 6.4]
    scaled = x.c()
    assert np.allclose(scaled.scale(2, 5), 0.21428571428571427)
    assert np.allclose(scaled.to_python(), [2.2142857142857144, 2.0, 4.785714285714286, 5.0])
    assert np.allclose(x.c().sin(1, 1).to_python(), [0.8414709848078965, 0.844849172063764, 0.8481940061209319, 0.8515053549310787])
    assert np.allclose(x.c().sin(1, 1, 2).to_python(), [0.8414709848078965, 0.8481940061209319, 0.8547830877678237, 0.8612371892561972])
    assert ms.Vector([1.2312414, 3.1231, 5.49554, 6.5000000001]).floor().to_python() == [1.0, 3.0, 5.0, 6.0]
    assert ms.Vector([-1.0, -3.0, -5.0, -6.0]).abs().to_python() == [1.0, 3.0, 5.0, 6.0]
    assert ms.Vector(x.size()).correl(x).to_python() == [494.0, 324.0, 154.0, 324.0]
    assert ms.Vector(x.size()).convlv(x, x.c().reverse()).to_python() == [294.0, 122.0, 318.0, 490.0]
    assert np.allclose(ms.Vector(x.size()).convlv(x, x.c().reverse(), -1).to_python(), [305.9999866504336, 306.0, 306.0000133495664, 306.0])
    assert x.c().spctrm(x).to_python() == [60.625, 2.0, 15.0, 16.0]
    assert ms.Vector(x.size()).filter(x, x.c().reverse()).to_python() == [308.0, -66.0, 376.0, 750.0]
    assert ms.Vector(x.size()).fft(x, -1).to_python() == [17.5, 16.5, -12.5, -15.5]
    assert x.c().fft(-1).to_python() == ms.Vector(x.size()).fft(x, -1).to_python()


def _vector_record_still_uses_micro_runtime():
    sim = ms.Sim()
    sim.set_device("cpu")
    sim.set_num_threads(1)
    sim.set_dt(0.025)

    soma = ms.section("soma", "soma")
    soma.L_um = 10.0
    soma.diam_um = 10.0
    sim.build_morphology([{"name": "CELL", "num_cells": 1, "sections": [soma]}])
    cell = sim.population("CELL")[0]
    cell.v_init = -65.0
    group = cell.group("soma")
    group.Ra = 35.4
    group.cm = 1.0
    group.insert("hh")
    seg = group[0](0.5)

    sim.build_microcircuit()
    sim.finitialize(-65.0)
    t = ms.Vector().record(sim._ref_t)
    v = ms.Vector().record(seg._ref_v)
    sim.run(0.1)

    assert t.to_python() == [0.0, 0.025, 0.05, 0.07500000000000001, 0.1]
    assert v[0] == -65.0


def test_vector_api():
    _vector_construction_and_basic_mutation_style()
    _vector_statistics_style()
    _vector_copy_and_data_morphing_style()
    _vector_where_and_transform_style()
    _vector_record_still_uses_micro_runtime()
