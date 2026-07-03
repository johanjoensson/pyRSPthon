import os

import numpy as np

from pyRSPthon.read.symcof import read_symcof, rotations_in_lattice_basis
from conftest import CUB_CELL


def test_read_symcof_fcc(fixtures):
    group = read_symcof(os.path.join(fixtures, "symcof_fcc"))
    assert group.spins == 1
    assert group.rotations.shape == (48, 3, 3)
    # all ops orthogonal with det +-1
    for rot in group.rotations:
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert abs(abs(np.linalg.det(rot)) - 1) < 1e-10


def test_lattice_basis_integer_and_closed(fixtures):
    group = read_symcof(os.path.join(fixtures, "symcof_fcc"))
    frac = rotations_in_lattice_basis(group.rotations, CUB_CELL)
    ops = {tuple(m.flatten()) for m in frac}
    assert len(ops) == 48
    for a in frac[:8]:
        for b in frac[:8]:
            assert tuple((a @ b).flatten()) in ops
