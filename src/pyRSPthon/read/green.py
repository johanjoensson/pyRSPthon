"""
Convenience access to green.inp metadata for the plotting tools.

The heavy lifting lives in pyRSPthon.read.greeninp, which mirrors RSPt's
own parser (substring-matched keywords, progressive-optional data lines).
This module keeps the small RSPtGreen view the plot CLIs consume.
"""

from collections import namedtuple
import os

from .greeninp import parse_green_inp

RSPtGreen = namedtuple(
    "RSPtGreen", ["spectrum", "emesh", "matsubara", "kpath", "clusters"], defaults=None
)


def get_green(prefix="."):
    """
    Read green.inp from prefix and return an RSPtGreen namedtuple:
    spectrum: list of flag tokens (or None), emesh: (n, emin, emax, eim),
    matsubara: (nmats, head, log, tail), kpath: (nk, xticks, labels),
    clusters: list of cluster labels.
    """
    green, _ = parse_green_inp(os.path.join(prefix or ".", "green.inp"))

    clusters = []
    for cl in green.clusters:
        if cl.label:
            clusters.append(cl.label)
        elif cl.orbitals:
            orb = cl.orbitals[0]
            clusters.append(f"{orb.t}{orb.l}{orb.e}{orb.site}{orb.basis}")

    spectrum = None
    kpath = None
    if green.spectrum is not None:
        spectrum = green.spectrum.flags.split()
        if green.spectrum.gp_nkp:
            segments = green.spectrum.gp_nkp
            # kpath convention: sum(nk + 1) + 1 points along the path
            nk = sum(segments) + len(segments) + 1
            labels = green.spectrum.gp_labels or []
            xticks = [i + sum(segments[:i]) for i in range(len(labels))]
            kpath = (nk, xticks, labels)

    return RSPtGreen(
        spectrum=spectrum,
        emesh=green.energymesh[0] if green.energymesh else None,
        matsubara=green.matsubara[0] if green.matsubara else None,
        kpath=kpath,
        clusters=clusters,
    )
