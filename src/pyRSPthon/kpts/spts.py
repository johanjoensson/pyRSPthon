"""
Readers/writers for RSPt k-point files (spts format).

Grid files (as written by cub):
    (i6)
       1
    (2i12)
    <nk> <iwsum>      <N1 N2 N3 onum1 onum2 onum3 oden1 oden2 oden3>
    (3f18.0, i6)
    <kx ky kz weight>    (k in reciprocal-lattice-fractional coords,
                          integer weights summing to iwsum)

Band-path files (as written by kpath):
    ( i6 )
       1
    (2i12)
    <nk> <nk>
    (3e22.14,i12)
    <kx ky kz 1>  [  #<label>]
"""

import numpy as np


def format_spts_number(x, decimals=15):
    """
    Format a float the way RSPt's decwrn1(x, 17) does: fixed decimal,
    suppressed leading zero, 17 characters right-justified.
    """
    s = f"{x:.{decimals}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return f"{s:>17s}"


def write_spts(fname, kpoints, weights, grid=None, shift_num=None, shift_den=None):
    """
    Write a k-point grid in spts format.

    Parameters:
    ===========
    fname: str - output file name (usually "spts")
    kpoints: (nk, 3) array - k-points in reciprocal-lattice-fractional coords
    weights: (nk,) int array - integer multiplicities
    grid, shift_num, shift_den: optional 3-int sequences recorded on the
        header line (grid divisions and rational offsets), like cub does.
    """
    kpoints = np.asarray(kpoints, dtype=float)
    weights = np.asarray(weights)
    if np.any(weights != weights.astype(int)):
        raise ValueError("spts weights must be integers")
    weights = weights.astype(int)
    iwsum = int(weights.sum())
    with open(fname, "w") as f:
        f.write(" (i6)\n     1\n")
        f.write(" (2i12)\n")
        f.write(f" {len(kpoints):11d} {iwsum:11d}")
        if grid is not None:
            shift_num = shift_num if shift_num is not None else (0, 0, 0)
            shift_den = shift_den if shift_den is not None else (1, 1, 1)
            trailer = " ".join(
                str(int(v)) for v in (*grid, *shift_num, *shift_den)
            )
            f.write(f"      {trailer}")
        f.write("\n")
        f.write(" (3f18.0, i6)\n")
        for k, w in zip(kpoints, weights):
            f.write(
                f" {format_spts_number(k[0])} {format_spts_number(k[1])}"
                f" {format_spts_number(k[2])} {w:5d}\n"
            )


def write_spts_band(fname, kpoints, labels, weights=None):
    """
    Write a band-path k-point file (spts.band format, as the kpath tool).

    Parameters:
    ===========
    fname: str - output file name (usually "spts.band")
    kpoints: (nk, 3) array - path points in reciprocal-lattice-fractional coords
    labels: dict[int, str] - point index -> high-symmetry label
    weights: optional (nk,) ints; defaults to all 1. A single zero-weight
        point can be used to disable RSPt's time-reversal consistency check.
    """
    kpoints = np.asarray(kpoints, dtype=float)
    if weights is None:
        weights = np.ones(len(kpoints), dtype=int)
    with open(fname, "w") as f:
        f.write(" ( i6 )\n")
        f.write(f"{1:6d}\n")
        f.write(" (2i12)\n")
        f.write(f"{len(kpoints):12d}{int(np.sum(weights)):12d}\n")
        f.write(" (3e22.14,i12)\n")
        for i, (k, w) in enumerate(zip(kpoints, weights)):
            line = f"{k[0]:22.14E}{k[1]:22.14E}{k[2]:22.14E}{int(w):12d}"
            if i in labels:
                line += f"   #{labels[i]}"
            f.write(line + "\n")


def read_spts(fname):
    """
    Read an spts file (grid or band-path variant).

    Returns:
    ========
    kpoints: (nk, 3) float array
    weights: (nk,) int array
    iwsum: int - total weight from the header
    labels: dict[int, str] - #label comments (band-path files)
    """
    with open(fname, "rt") as f:
        next(f)  # format
        nksets = int(next(f))
        if nksets != 1:
            raise ValueError(f"Only single k-set spts files supported, got {nksets}")
        next(f)  # format
        fields = next(f).split()
        nk, iwsum = int(fields[0]), int(fields[1])
        next(f)  # format
        kpoints = np.empty((nk, 3))
        weights = np.empty(nk, dtype=int)
        labels = {}
        for i in range(nk):
            line = next(f)
            data, _, comment = line.partition("#")
            fields = data.split()
            kpoints[i] = [float(x) for x in fields[:3]]
            weights[i] = int(fields[3])
            if comment.strip():
                labels[i] = comment.strip()
    return kpoints, weights, iwsum, labels
