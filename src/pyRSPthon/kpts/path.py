"""
Band-path k-point generation (Python replacement for RSPt's kpath tool).

Produces spts.band files (all weights 1, high-symmetry points tagged with
#label comments) and the matching green.inp spectrum block.
"""

import numpy as np

from .spts import write_spts_band


def resolve_path(cell, path_string, special_points=None):
    """
    Resolve a path specification like "GXWLG" (or "G,X,W,L,G") into
    high-symmetry point coordinates.

    Parameters:
    ===========
    cell: (3, 3) array - direct lattice vectors as rows
    path_string: str - point labels; 'G' or 'Γ' is the zone center
    special_points: optional dict label -> 3 floats; if None, use ASE's
        special points for the detected Bravais lattice

    Returns:
    ========
    labels: list[str], points: (n, 3) array (reciprocal-fractional)
    """
    if special_points is None:
        from ase.cell import Cell

        special_points = Cell(np.asarray(cell)).bandpath(npoints=0).special_points
    if "," in path_string:
        labels = [lab.strip() for lab in path_string.split(",") if lab.strip()]
    else:
        labels = list(path_string)
    points = []
    for lab in labels:
        key = "G" if lab in ("G", "Γ") else lab
        if key not in special_points:
            raise ValueError(
                f"Unknown high-symmetry point {lab!r}; known: "
                f"{sorted(special_points)}"
            )
        points.append(special_points[key])
    return labels, np.array(points, dtype=float)


def interpolate_path(nodes, nk_per_segment):
    """
    Interpolate a k-path exactly like RSPt's kpath tool: each segment
    contributes its start node plus nk interior points; the final node ends
    the path. Total = sum(nk+1) + 1 points.

    Parameters:
    ===========
    nodes: (n, 3) array - high-symmetry points
    nk_per_segment: int or sequence of n-1 ints - interior points per segment

    Returns:
    ========
    kpoints: (nkb, 3) array, labels: dict[int, int] - point index -> node index
    """
    nodes = np.asarray(nodes, dtype=float)
    nseg = len(nodes) - 1
    if nseg < 1:
        raise ValueError("A path needs at least two high-symmetry points")
    if np.isscalar(nk_per_segment):
        nk_per_segment = [int(nk_per_segment)] * nseg
    if len(nk_per_segment) != nseg:
        raise ValueError(
            f"Got {len(nk_per_segment)} segment lengths for {nseg} segments"
        )
    kpoints = []
    node_at = {}
    for s in range(nseg):
        node_at[len(kpoints)] = s
        nk = int(nk_per_segment[s])
        kpoints.append(nodes[s])
        for i in range(1, nk + 1):
            kpoints.append(((nk + 1 - i) * nodes[s] + i * nodes[s + 1]) / (nk + 1))
    node_at[len(kpoints)] = nseg
    kpoints.append(nodes[-1])
    return np.array(kpoints), node_at


def spectrum_block(labels, nk_per_segment, kind="Band Pband eV"):
    """
    The green.inp spectrum block matching a generated path (same as the
    kpath tool prints to kpath.log).
    """
    if np.isscalar(nk_per_segment):
        nk_per_segment = [int(nk_per_segment)] * (len(labels) - 1)
    lines = [
        " spectrum",
        f" {kind}",
        " " + "".join(f"  {int(n):3d} " for n in nk_per_segment),
        " " + "".join(f"{lab:1s}     " for lab in labels),
    ]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def write_band_path(
    fname,
    cell,
    path_string,
    nk_per_segment,
    special_points=None,
    zero_weight_point=False,
):
    """
    Generate a band path and write it in spts.band format.

    zero_weight_point: append one zero-weight k-point, which disables
    RSPt's time-reversal consistency check for path files.

    Returns:
    ========
    labels: list[str], nk_per_segment: list[int] (for the spectrum block)
    """
    labels, nodes = resolve_path(cell, path_string, special_points)
    nseg = len(labels) - 1
    if np.isscalar(nk_per_segment):
        nk_per_segment = [int(nk_per_segment)] * nseg
    kpoints, node_at = interpolate_path(nodes, nk_per_segment)
    point_labels = {i: labels[n] for i, n in node_at.items()}
    weights = np.ones(len(kpoints), dtype=int)
    if zero_weight_point:
        kpoints = np.vstack([kpoints, kpoints[-1]])
        weights = np.append(weights, 0)
    write_spts_band(fname, kpoints, point_labels, weights)
    return labels, list(nk_per_segment)
