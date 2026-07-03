import numpy as np
import pytest

from pyRSPthon.read.dat import (
    extract_dat,
    extract_dos,
    extract_pdos,
    match_columns,
    split_header_columns,
)


def test_tot_plus_minus_are_spin_columns():
    # regression: "tot" used to swallow tot+/tot- before the spin branches
    cols = match_columns(split_header_columns("# Energy  Tot  Tot+  Tot-"))
    assert cols == {"energy": 0, "total": 1, "up": 2, "down": 3}


def test_spin_up_dn_labels():
    cols = match_columns(
        split_header_columns("#  Energy   Total   Spin Up   Spin Dn   Sz")
    )
    assert cols == {"energy": 0, "total": 1, "up": 2, "down": 3, "sz": 4}


def test_extract_dos(dos_dir):
    dos = extract_dos(prefix=str(dos_dir))
    assert dos.w.shape == (101,)
    assert dos.up is not None and dos.down is not None
    np.testing.assert_allclose(dos.up + dos.down, dos.sum, rtol=1e-12)


def test_extract_pdos_orbitals(dos_dir):
    pdos = extract_pdos("Fe", prefix=str(dos_dir))
    assert pdos.orbitals is not None
    assert pdos.orbitals.shape == (101, 5)


def test_extract_dat_indexmap_and_blocks(dos_dir):
    dat = extract_dat("hyb", "Fe", prefix=str(dos_dir))
    # indexmap [[1,0],[0,2]] -> 2x2 orbital matrix, two 1-orbital blocks
    assert dat.orbitals.shape == (101, 2, 2)
    assert sorted(map(sorted, dat.blocks)) == [[0], [1]]
    assert np.iscomplexobj(dat.orbitals)
    # diagonal sum = total when no explicit total column
    np.testing.assert_allclose(
        dat.sum, dat.orbitals[:, 0, 0] + dat.orbitals[:, 1, 1]
    )


def test_missing_energy_column_raises(tmp_path):
    np.savetxt(
        tmp_path / "dos.dat",
        np.zeros((5, 2)),
        header="  Something   Total",
        comments="#",
    )
    with pytest.raises(RuntimeError, match="energy mesh"):
        extract_dos(prefix=str(tmp_path))
