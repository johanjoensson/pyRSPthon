"""
K-point mesh generation with symcof-consistent symmetry reduction.

HARD INVARIANT: meshes are reduced ONLY with the symmetry operations RSPt
itself knows — the ones in symcof (plus time reversal only when explicitly
requested). RSPt reconstructs the full Brillouin zone from the irreducible
wedge using exactly those operations; reducing with a larger group (e.g.
spglib's crystallographic group on a spin-axis-weeded fullrel run) silently
corrupts the calculation. Reducing with a subgroup is safe — it only means
more k-points — so operations under which a grid is not invariant are
dropped with a warning instead of failing.

The algorithm mirrors RSPt's cub tool: grid indices n in a (generalized)
Monkhorst-Pack mesh, X = (n + shift)/N in supercell fractional coordinates,
mapped to reciprocal-lattice-fractional x = M X mod 1, reduced, then folded
to the shortest cartesian image (inbz).
"""

import warnings
from math import lcm

import numpy as np

from ..read.symcof import read_symcof


def reciprocal_ops(rotations_cart, cell):
    """
    Convert cartesian symmetry operations to integer matrices acting on
    reciprocal-lattice-fractional coordinates.
    cell rows are the (direct) lattice vectors.
    """
    A = np.asarray(cell, dtype=float).T  # columns = lattice vectors
    B = np.linalg.inv(A).T  # columns = reciprocal basis vectors
    Binv = np.linalg.inv(B)
    H = np.einsum("ij,njk,kl->nil", Binv, rotations_cart, B)
    Hr = np.rint(H)
    if not np.allclose(H, Hr, atol=1e-6):
        raise ValueError(
            "Symmetry operations are not integer in the reciprocal basis; "
            "the symcof file does not match this cell."
        )
    return Hr.astype(int)


def generate_grid(nks, shift_num=(0, 0, 0), shift_den=(1, 1, 1), map_matrix=None):
    """
    Generate the distinct points of a (generalized) Monkhorst-Pack grid.

    Points are returned as exact integers over a common denominator D:
    x = X_int / D in reciprocal-lattice-fractional coordinates, folded into
    [0, 1). With a non-identity map_matrix M (the cub.inp supercell map) the
    grid downfolds; the downfolding multiplicity must be uniform.

    Returns:
    ========
    points: (nd, 3) int array - distinct grid points (times D)
    D: int - common denominator
    multiplicity: int - downfolding multiplicity (|det M| when commensurate)
    """
    nks = [int(n) for n in nks]
    shift_num = [int(o) for o in shift_num]
    shift_den = [int(d) for d in shift_den]
    for num, den in zip(shift_num, shift_den):
        if den <= 0 or num < 0 or num >= den:
            raise ValueError(f"Invalid shift {num}/{den}: need 0 <= num < den")
    M = (
        np.eye(3, dtype=int)
        if map_matrix is None
        else np.asarray(map_matrix, dtype=int)
    )
    if abs(round(np.linalg.det(M))) == 0:
        raise ValueError("The supercell map matrix is singular")

    D = lcm(*[n * d for n, d in zip(nks, shift_den)])
    scale = np.array([D // (n * d) for n, d in zip(nks, shift_den)], dtype=np.int64)
    indices = np.stack(
        np.meshgrid(*[np.arange(n) for n in nks], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    # (n_k * den_k + num_k) * D/(N_k den_k) = X_k * D in supercell coords
    comp = (indices * np.array(shift_den) + np.array(shift_num)) * scale
    X = comp @ M.T % D

    points, counts = np.unique(X, axis=0, return_counts=True)
    # Downfolding is a quotient by a subgroup, so the multiplicity is the
    # same for every point; this can only fire on an implementation bug.
    if counts.min() != counts.max():
        raise ValueError("Internal error: uneven downfolding multiplicities")
    return points, D, int(counts[0])


def reduce_grid(points, D, ops_frac, time_reversal=False):
    """
    Reduce distinct grid points to the irreducible wedge under ops_frac
    (integer matrices in the reciprocal-fractional basis).

    Operations that do not map the grid onto itself are dropped with a
    warning (reducing with a subgroup is always safe).

    Returns:
    ========
    reps: (nk, 3) int array - representative points (times D)
    weights: (nk,) int array - orbit sizes; sum equals len(points)
    n_used: int - number of operations actually used
    """
    ops = [np.asarray(H, dtype=int) for H in ops_frac]
    if time_reversal:
        seen = {tuple(H.flatten()) for H in ops}
        for H in list(ops):
            if tuple((-H).flatten()) not in seen:
                ops.append(-H)

    index = {tuple(p): i for i, p in enumerate(points)}
    permutations = []
    dropped = 0
    for H in ops:
        images = points @ H.T % D
        perm = np.empty(len(points), dtype=int)
        ok = True
        for i, img in enumerate(images):
            j = index.get(tuple(img))
            if j is None:
                ok = False
                break
            perm[i] = j
        if ok:
            permutations.append(perm)
        else:
            dropped += 1
    if dropped:
        warnings.warn(
            f"{dropped} symmetry operations do not leave the k-grid invariant "
            "and were not used for reduction (safe: the grid is reduced less). "
            "A symmetric shift or commensurate grid would avoid this."
        )

    visited = np.zeros(len(points), dtype=bool)
    reps = []
    weights = []
    for i in range(len(points)):
        if visited[i]:
            continue
        orbit = [i]
        visited[i] = True
        stack = [i]
        while stack:
            j = stack.pop()
            for perm in permutations:
                m = perm[j]
                if not visited[m]:
                    visited[m] = True
                    orbit.append(m)
                    stack.append(m)
        reps.append(points[i])
        weights.append(len(orbit))
    return np.array(reps), np.array(weights, dtype=int), len(permutations)


def fold_to_bz(kfrac, cell):
    """
    Fold reciprocal-fractional k-points to their shortest cartesian image,
    replicating cub's inbz: first into (-1/2, 1/2], then pick the integer
    shift in [-2, 2]^3 with strictly smaller cartesian norm (scan order as
    in cub, ties keep the current image). Components below cub's EPS are
    zeroed.
    """
    A = np.asarray(cell, dtype=float).T
    B = np.linalg.inv(A).T  # columns = reciprocal basis vectors
    eps = 512 * np.finfo(float).eps
    out = np.array(kfrac, dtype=float)
    out -= np.ceil(out - 0.5)  # into (-1/2, 1/2]
    shifts = np.array(
        [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]
    )
    for row in out:
        best = row.copy()
        norm = np.sum((B @ row) ** 2)
        for s in shifts:
            x = row + s
            u = np.sum((B @ x) ** 2)
            if u < norm - 1e-14:
                best = x
                norm = u
        row[:] = best
        row[np.abs(row) < eps] = 0.0
    return out


def reduce_explicit_grid(kfrac_full, cell, symcof_path="symcof", time_reversal=False):
    """
    Symcof-reduce an explicitly listed full k-grid (e.g. the one kpLib
    suggests). kpLib's own symmetry reduction is deliberately ignored: it
    uses the full crystallographic group, which may be larger than the ops
    RSPt knows (see the module docstring).

    Parameters:
    ===========
    kfrac_full: (n, 3) array - ALL grid points, reciprocal-fractional
    cell: (3, 3) array - direct lattice vectors as rows

    Returns:
    ========
    kpoints: (nk, 3) float array (BZ folded), weights: (nk,) int array
    """
    from fractions import Fraction

    kfrac_full = np.mod(np.asarray(kfrac_full, dtype=float), 1.0)
    D = 1
    fracs = []
    for row in kfrac_full:
        frow = [Fraction(float(x)).limit_denominator(1_000_000) for x in row]
        fracs.append(frow)
        for fr in frow:
            D = lcm(D, fr.denominator)
    points = np.array(
        [[int(fr * D) % D for fr in row] for row in fracs], dtype=np.int64
    )
    if not np.allclose(points / D, kfrac_full, atol=1e-9):
        raise ValueError("Could not represent the k-grid with exact rationals")
    points = np.unique(points, axis=0)

    group = read_symcof(symcof_path)
    ops = reciprocal_ops(group.rotations, cell)
    reps, weights, _ = reduce_grid(points, D, ops, time_reversal)
    return fold_to_bz(reps / D, cell), weights


def get_kpts_min_distance(
    atoms, min_distance, symcof_path="symcof", time_reversal=False, include_gamma="auto"
):
    """
    Pick an efficient (generalized) grid with kpLib for a target minimum
    periodic distance, then reduce it with the symcof operations.

    Requires the kpLib optional dependency.
    """
    from .kplib import suggest_grid

    suggestion = suggest_grid(atoms, min_distance, include_gamma)
    return reduce_explicit_grid(
        suggestion["coords"],
        atoms.cell[:],
        symcof_path=symcof_path,
        time_reversal=time_reversal,
    )


def get_kpts(
    cell,
    nks,
    shift_num=(0, 0, 0),
    shift_den=(1, 1, 1),
    map_matrix=None,
    symcof_path="symcof",
    time_reversal=False,
    reduce=True,
):
    """
    Generate a symcof-consistently reduced k-point mesh.

    Parameters:
    ===========
    cell: (3, 3) array - direct lattice vectors as rows (any length unit)
    nks: 3 ints - grid divisions
    shift_num, shift_den: rational grid offsets num/den per axis
    map_matrix: optional 3x3 int supercell map (cub.inp M); None = identity
    symcof_path: path to symcof; required for reduction
    time_reversal: also reduce with -k (only if RSPt is told the same!)
    reduce: set False to write the full grid (always correct, just larger)

    Returns:
    ========
    kpoints: (nk, 3) float array - reciprocal-fractional, BZ-folded
    weights: (nk,) int array - multiplicities (sum = number of grid points)
    """
    points, D, _ = generate_grid(nks, shift_num, shift_den, map_matrix)
    if reduce:
        group = read_symcof(symcof_path)
        ops = reciprocal_ops(group.rotations, cell)
        reps, weights, _ = reduce_grid(points, D, ops, time_reversal)
    else:
        reps = points
        weights = np.ones(len(points), dtype=int)
    kpoints = fold_to_bz(reps / D, cell)
    return kpoints, weights
