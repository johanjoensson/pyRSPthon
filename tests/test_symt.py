import os

import numpy as np
import pytest

from pyRSPthon.ase import read_symt, write_symt


def test_read_legacy_nio(fixtures):
    atoms = read_symt(os.path.join(fixtures, "symt_legacy.inp"))
    assert atoms.get_chemical_formula() == "NiO"
    # labels a/b -> deterministic tags
    assert list(atoms.get_tags()) == [0, 1]
    assert not atoms.info["spinpol"]


def test_read_legacy_fulrel_spinaxis(fixtures):
    atoms = read_symt(os.path.join(fixtures, "symt_fulrel.inp"))
    assert atoms.info["spinpol"] and atoms.info["fullrel"]
    # cartesian z axis in a bcc cell = a1 + a2 in lattice coordinates
    np.testing.assert_allclose(atoms.info["spinaxis"], [1.0, 1.0, 0.0], atol=1e-12)


def test_round_trip(fixtures, tmp_path):
    a = read_symt(os.path.join(fixtures, "symt_legacy.inp"))
    write_symt(a, prefix=str(tmp_path))
    b = read_symt(tmp_path / "symt.inp")
    np.testing.assert_allclose(a.cell[:], b.cell[:], atol=1e-12)
    np.testing.assert_allclose(
        a.get_scaled_positions(), b.get_scaled_positions(), atol=1e-12
    )
    assert list(a.get_atomic_numbers()) == list(b.get_atomic_numbers())
    assert a.info["labels"] == b.info["labels"]


def test_cartesian_positions_converted(tmp_path):
    (tmp_path / "symt.inp").write_text(
        "lengthscale\n 2.0\n\n"
        "latticevectors\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n\n"
        "atoms\n 2\n"
        " 0.0 0.0 0.0  26 l a\n"
        " 0.5 0.5 0.5  26 c b\n"
    )
    atoms = read_symt(tmp_path / "symt.inp")
    # cartesian (0.5,0.5,0.5) in lengthscale units == fractional (0.5,0.5,0.5)
    np.testing.assert_allclose(
        atoms.get_scaled_positions()[1], [0.5, 0.5, 0.5], atol=1e-12
    )


def test_noncollinear_moments_rejected(tmp_path):
    from ase import Atoms

    atoms = Atoms("Fe", cell=np.eye(3) * 2.8, scaled_positions=[[0, 0, 0]], pbc=True)
    atoms.set_initial_magnetic_moments([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="collinear"):
        write_symt(atoms, prefix=str(tmp_path))


def test_up_dn_labels_set_moments_and_spinpol(tmp_path):
    (tmp_path / "symt.inp").write_text(
        "latticevectors\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n\n"
        "atoms\n 2\n"
        " 0.0 0.0 0.0  25 l up\n"
        " 0.5 0.5 0.5  25 l dn\n"
    )
    atoms = read_symt(tmp_path / "symt.inp")
    assert atoms.info["spinpol"]
    np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), [1.0, -1.0])
