import pytest

from pyRSPthon.read.green import get_green


GREEN_INP = """# a green.inp with a band path
cluster
 Id: Fe3d

energymesh
 501 -1.0 1.0 0.005

spectrum
 Band Pband eV
 20 20 20 20
 G X W L G

matsubara
 512 1 1 1
"""


def test_get_green(tmp_path):
    (tmp_path / "green.inp").write_text(GREEN_INP)
    green = get_green(prefix=str(tmp_path))
    assert green.emesh == (501, -1.0, 1.0, 0.005)
    assert green.matsubara == (512, 1, 1, 1)
    assert "eV" in green.spectrum
    nk, ticks, labels = green.kpath
    # kpath tool convention: sum(nk_per_segment) + n_segments + 1
    assert nk == 4 * 20 + 4 + 1
    assert labels == ["G", "X", "W", "L", "G"]
    assert ticks == [0, 21, 42, 63, 84]


def test_no_energymesh_gives_none(tmp_path):
    # regression: a fabricated default emesh used to be returned silently
    (tmp_path / "green.inp").write_text("cluster\n Id: X\n")
    green = get_green(prefix=str(tmp_path))
    assert green.emesh is None
    assert len(green.clusters) == 1
