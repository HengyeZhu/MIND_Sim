from __future__ import annotations

import numpy as np

import mind_sim as ms


ONE_POINT_SOMA_SWC = """
# One-point soma
#n,type,x,y,z,radius,parent
1  1 0 0 0 10 -1
2  3 0 10 0 1 1
3  3 0 20 0 1 2
4  3 0 -10 0 2 1
5  3 0 -20 0 2 4
6  3 10 0 0 .5 1
7  3 10 5 0 1 6
"""


def test_swc(tmp_path):
    path = tmp_path / "one_point_soma.swc"
    path.write_text(ONE_POINT_SOMA_SWC)

    sections = ms.load_swc_sections(str(path))

    assert [section.name for section in sections] == ["soma_0", "dend_0", "dend_1", "dend_2"]
    assert [section.label for section in sections] == ["soma", "dend", "dend", "dend"]
    assert np.allclose([section.L_um for section in sections], [20.0, 10.0, 10.0, 5.0])
    assert sections[0].parent is None
    assert [(section.parent, section.parentx) for section in sections[1:]] == [
        ("soma_0", 0.5),
        ("soma_0", 0.5),
        ("soma_0", 0.5),
    ]
    assert list(sections[0].pt3d) == [
        (-10.0, 0.0, 0.0, 20.0),
        (0.0, 0.0, 0.0, 20.0),
        (10.0, 0.0, 0.0, 20.0),
    ]
