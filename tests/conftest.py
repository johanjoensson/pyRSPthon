import os

import numpy as np
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# The cell and supercell map from rspt/cub/cub.inp, matching symcof_fcc and
# the cub.k.* reference files.
CUB_CELL = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
CUB_MAP = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


@pytest.fixture
def fixtures():
    return FIXTURES


@pytest.fixture
def dos_dir(tmp_path):
    """A directory with synthetic dos.dat / pdos-Fe.dat / real- imag- files."""
    w = np.linspace(-10, 10, 101)
    tot = np.exp(-(w**2) / 4)
    up, dn = tot * 0.6, tot * 0.4
    np.savetxt(
        tmp_path / "dos.dat",
        np.column_stack([w, tot, up, dn]),
        header="  Energy       Total        Spin Up      Spin Dn",
        comments="#",
    )
    orb = np.column_stack([tot * f for f in (0.1, 0.2, 0.3, 0.2, 0.2)])
    np.savetxt(
        tmp_path / "pdos-Fe.dat",
        np.column_stack([w, tot, up, dn, orb]),
        header="  Energy       Total        Spin Up      Spin Dn      Orbitals",
        comments="#",
    )
    hyb = np.column_stack([w, np.cos(w), np.cos(w + 1)])
    header = "  Energy      Orbitals\n indexmap\n 1 0\n 0 2"
    np.savetxt(tmp_path / "real-hyb-Fe.dat", hyb, header=header, comments="#")
    np.savetxt(tmp_path / "imag-hyb-Fe.dat", hyb * 0.5, header=header, comments="#")
    return tmp_path


@pytest.fixture
def band_dir(tmp_path):
    """A directory with a synthetic binary band.data + band.gpi pair."""
    nk, ne, ncol = 40, 30, 11
    e = np.linspace(-5, 5, ne)
    k = np.arange(nk)
    disp = 3 * np.cos(2 * np.pi * k / nk)
    A = np.zeros((nk, ne, ncol), dtype=np.float32)
    for kk in range(nk):
        A[kk, :, 0] = np.exp(-((e - disp[kk]) ** 2) / 0.1)
        for o in range(10):
            A[kk, :, 1 + o] = A[kk, :, 0] * (o + 1) / 10
    A.tofile(tmp_path / "band.data")
    A.tofile(tmp_path / "pband-Fe.data")
    gpi = (
        ' #!/usr/bin/env gnuplot\n'
        ' set ylabel "Energy (eV)"\n'
        f' set xtics ("{{/Symbol G}}" 0 ,"X" 20 ,"L" {nk - 1} )\n'
        f' set ytics ("-5.0" 0 ,"0.0" {ne // 2} ,"5.0" {ne - 1} )\n'
        f" ky(x) = (int(x)%{ne})\n"
        f" kx(x) = int(x)/{ne}\n"
        f" p [0:{nk - 1}] 'band.data' binary record={nk * ne}"
        ' format="%float%*10float" u (kx($0)):(ky($0)):1 w image\n'
    )
    (tmp_path / "band.gpi").write_text(gpi)
    (tmp_path / "pband-Fe.gpi").write_text(gpi.replace("band.data", "pband-Fe.data"))
    return tmp_path
