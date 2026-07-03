"""
Parser for RSPt symcof files (written by `symt` as symt.out).

Only the header and the symmetry-operation matrices are parsed; the per-type
Euler-angle and harmonics blocks are left to the RSPt tools themselves.

The rotation matrices in symcof are in CARTESIAN coordinates. These are the
only symmetry operations RSPt knows about at runtime — any k-point mesh must
be reduced with exactly these (see pyRSPthon.kpts).
"""

from collections import namedtuple
import numpy as np

SymcofGroup = namedtuple("SymcofGroup", ["rotations", "spins"])


def read_symcof(fname="symcof"):
    """
    Read the group header and rotation matrices of a symcof file.

    Parameters:
    ===========
    fname: str - path to the symcof file (or symt.out)

    Returns:
    ========
    SymcofGroup with
      rotations: (ngrp, 3, 3) float array - CARTESIAN rotation matrices
      spins: int - 1 (non spin polarized) or 2 (spin polarized/relativistic)
    """
    with open(fname, "rt") as f:
        next(f)  # format line " (2i6)"
        fields = next(f).split()
        ngrp, spins = int(fields[0]), int(fields[1])
        next(f)  # format line " (/ 3f18.14 / ...)"
        rotations = np.empty((ngrp, 3, 3))
        for i in range(ngrp):
            comment = next(f)
            if not comment.lstrip().startswith("c"):
                raise ValueError(
                    f"Expected 'c Element' comment line in {fname}, got {comment!r}"
                )
            for row in range(3):
                rotations[i, row] = [float(x) for x in next(f).split()[:3]]
    return SymcofGroup(rotations=rotations, spins=spins)


def rotations_in_lattice_basis(rotations, cell):
    """
    Convert cartesian rotation matrices to (integer) matrices acting on
    fractional (lattice) coordinates: H = A^-1 P A with A columns = lattice
    vectors (cell rows are the lattice vectors, so A = cell.T).

    Returns an (ngrp, 3, 3) integer array; raises if any matrix is not
    integer within tolerance (which would mean ops incompatible with cell).
    """
    A = np.asarray(cell).T
    Ainv = np.linalg.inv(A)
    frac = np.einsum("ij,njk,kl->nil", Ainv, rotations, A)
    rounded = np.rint(frac)
    if not np.allclose(frac, rounded, atol=1e-6):
        raise ValueError(
            "Symmetry operations are not integer in the lattice basis; "
            "the symcof file does not match this cell."
        )
    return rounded.astype(int)


def validate_against_spglib(atoms, group: SymcofGroup, symprec=1e-5):
    """
    Cross-check a symcof group against spglib's analysis of the structure.

    Returns a dict report with the spglib space group, the two group orders
    and the number of operations spglib finds that symt weeded away (e.g. by
    the spin axis). Requires spglib.
    """
    import spglib

    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    spg_rotations = np.unique(dataset.rotations, axis=0)

    symcof_frac = rotations_in_lattice_basis(group.rotations, atoms.cell[:])
    symcof_set = {tuple(r.flatten()) for r in symcof_frac}
    spglib_set = {tuple(r.flatten()) for r in spg_rotations}

    return {
        "spacegroup": f"{dataset.international} ({dataset.number})",
        "symcof_order": len(symcof_set),
        "spglib_order": len(spglib_set),
        "weeded_ops": sorted(spglib_set - symcof_set),
        "extra_ops": sorted(symcof_set - spglib_set),
    }
