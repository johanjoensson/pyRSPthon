"""
kpLib-backed grid-vector selection.

kpLib is only used to *choose* an efficient (generalized) k-point grid for a
target minimum distance. Symmetry reduction of the resulting grid is always
redone with the operations RSPt knows (from symcof); see pyRSPthon.kpts.grid.
"""


def suggest_grid(atoms, min_distance, include_gamma="auto"):
    """
    Use kpLib to pick an efficient generalized k-point grid.

    Parameters:
    ===========
    atoms: ase.Atoms  - structure to generate k-points for
    min_distance: float - target minimum distance between lattice points (Å)
    include_gamma: "auto"|True|False - whether the grid must contain Γ

    Returns:
    ========
    dict as returned by kpLib (grid vectors, shift, full point list, ...)
    """
    from kpLib.interface import get_kpoints

    return get_kpoints(
        atoms.get_cell()[:],
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
        min_distance=min_distance,
        include_gamma=include_gamma,
    )
