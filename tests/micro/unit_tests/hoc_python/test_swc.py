from __future__ import annotations

import mind_sim as ms


def test_swc_loader(tmp_path, capsys):
    swc = """\
# index     type         X            Y            Z       radius       parent
1           1  1.049119830 -8.248093605  0.500801444  1.359369397          -1
2           1  0.945353746 -7.414265633  0.450333387  2.455268145           1
"""
    path = tmp_path / "test.swc"
    path.write_text(swc)

    sections = ms.load_swc_sections(str(path))
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert [section.label for section in sections] == ["soma"]
    assert sections[0].parent is None
