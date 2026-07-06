"""
Unified band-structure reading for the RSPt output zoo.

Sources (all returned as BandStructure):
- DMFT spectral functions: binary band.data / pband-<cluster>.data, with
  metadata auto-detected from the sibling band.gpi gnuplot script and/or
  green.inp.
- DFT bands converted by evconv: bandfile_0..N (+ symline_* tick files).
- Fatbands: fatbands.tTT.eE.lL.mM files (x, energy, weight columns).
- Raw eigenvalues files (plot without running evconv at all).
"""

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .band import get_band, get_pband, peek_band_header
from .green import get_green

Ry_to_eV = 13.605703976


@dataclass
class BandStructure:
    """
    kind = "spectral": spectral[ne, nk, ncols] on the energies mesh;
    kind = "lines": bands[nk, nbands] eigenvalue curves over kdist.
    weights maps a label (e.g. "t01.e1.l2.m-1" or an orbital name) to a
    (nk, nbands) or (ne, nk) weight array of the same kind.
    """

    kind: str
    kdist: np.ndarray
    energies: np.ndarray | None = None
    spectral: np.ndarray | None = None
    bands: np.ndarray | None = None
    weights: dict = field(default_factory=dict)
    ticks: list = field(default_factory=list)
    tick_labels: list = field(default_factory=list)
    energy_unit: str = "Ry"


_GPI_XTICS = re.compile(r"set xtics\s*\((.*)\)")
_GPI_TIC = re.compile(r'"([^"]*)"\s+(-?\d+)')
_GPI_KY = re.compile(r"ky\(x\)\s*=\s*\(int\(x\)%\s*(\d+)\s*\)")
_GPI_RECORD = re.compile(r"binary record=\s*(\d+)")
_GPI_YLABEL = re.compile(r'set ylabel\s+"Energy \((\w+)\)"')


def parse_band_gpi(fname):
    """
    Extract plot metadata from an RSPt band.gpi gnuplot script.

    Returns a dict with (present when found): ne, nk, ticks, tick_labels,
    energy_unit, energies (linear mesh reconstructed from the ytics).
    """
    meta = {}
    ytic_pairs = []
    with open(fname, "rt") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            m = _GPI_KY.search(line)
            if m:
                meta["ne"] = int(m.group(1))
            m = _GPI_YLABEL.search(line)
            if m:
                meta["energy_unit"] = m.group(1)
            m = _GPI_RECORD.search(line)
            if m and "nk" not in meta:
                meta["record"] = int(m.group(1))
            m = _GPI_XTICS.search(line)
            if m:
                pairs = _GPI_TIC.findall(m.group(1))
                meta["ticks"] = [int(pos) for _, pos in pairs]
                meta["tick_labels"] = [
                    "Γ" if lab in ("{/Symbol G}", "G") else lab for lab, _ in pairs
                ]
            if "set ytics" in line and not ytic_pairs:
                ytic_pairs = [
                    (float(val), int(pos)) for val, pos in _GPI_TIC.findall(line)
                ]
    if "ne" in meta and "record" in meta:
        meta["nk"] = meta["record"] // meta["ne"]
    if "ne" in meta and len(ytic_pairs) >= 2:
        # The energy mesh is linear; fit e(i) = a + b i through the tic pairs
        vals = np.array([v for v, _ in ytic_pairs])
        idxs = np.array([i for _, i in ytic_pairs])
        b, a = np.polyfit(idxs, vals, 1)
        meta["energies"] = a + b * np.arange(meta["ne"])
    return meta


def _metadata_from_green(prefix):
    meta = {}
    try:
        green = get_green(prefix=prefix)
    except FileNotFoundError:
        return meta
    if green.spectrum is not None:
        meta["energy_unit"] = "eV" if "eV" in green.spectrum else "Ry"
    if green.kpath is not None:
        meta["nk"], meta["ticks"], meta["tick_labels"] = green.kpath
    if green.emesh is not None:
        ne, emin, emax = green.emesh[0], green.emesh[1], green.emesh[2]
        meta["ne"] = ne
        scale = Ry_to_eV if meta.get("energy_unit") == "eV" else 1.0
        meta["energies"] = np.linspace(emin * scale, emax * scale, ne)
    return meta


def read_spectral_bands(
    prefix=".",
    cluster=None,
    nk=None,
    ne=None,
    energies=None,
    ticks=None,
    tick_labels=None,
    energy_unit=None,
):
    """
    Read a binary band.data (or pband-<cluster>.data) spectral function.
    Metadata resolution order: explicit arguments > band.gpi > green.inp.
    """
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    basename = "band" if cluster is None else f"pband-{cluster}"
    meta = _metadata_from_green(prefix)
    gpi_file = f"{prefix}{basename}.gpi"
    if os.path.exists(gpi_file):
        meta.update(parse_band_gpi(gpi_file))
        
    # Check if a Band_header is present in the binary data and takes precedence
    data_file = f"{prefix}{basename}.data"
    header_meta = peek_band_header(data_file)
    if header_meta:
        meta["nk"] = header_meta["nk"]
        meta["ne"] = header_meta["ne"]
        
    nk = nk if nk is not None else meta.get("nk")
    ne = ne if ne is not None else meta.get("ne")
    if nk is None or ne is None:
        raise RuntimeError(
            f"Could not determine nk/ne for {basename}.data — no usable "
            f"{basename}.gpi or green.inp found, and no Band_header present in data file; pass them explicitly "
            "(plot_band: -nk/-ne)."
        )
    if energies is None:
        energies = meta.get("energies")
    if energies is None:
        energies = np.arange(ne, dtype=float)
    if cluster is None:
        spectral = get_band(nk=nk, ne=ne, prefix=prefix)
    else:
        spectral = get_pband(nk=nk, ne=ne, cluster=cluster, prefix=prefix)
    return BandStructure(
        kind="spectral",
        kdist=np.arange(nk, dtype=float),
        energies=np.asarray(energies, dtype=float),
        spectral=spectral,
        ticks=list(ticks) if ticks is not None else meta.get("ticks", []),
        tick_labels=(
            list(tick_labels)
            if tick_labels is not None
            else meta.get("tick_labels", [])
        ),
        energy_unit=energy_unit or meta.get("energy_unit", "Ry"),
    )


def read_bandfiles(prefix=".", energy_unit="eV"):
    """
    Read evconv output: bandfile_0..N (first column cumulative k-distance,
    up to 10 band columns each) and symline_* tick markers.
    """
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    fnames = sorted(
        glob.glob(f"{prefix}bandfile_*"),
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if not fnames:
        raise FileNotFoundError(f"No bandfile_* files found in {prefix or '.'}")
    kdist = None
    columns = []
    for fname in fnames:
        data = np.loadtxt(fname)
        if data.ndim == 1:
            data = data[:, None]
        if kdist is None:
            kdist = data[:, 0]
        columns.append(data[:, 1:])
    bands = np.hstack(columns)
    ticks = []
    for fname in sorted(glob.glob(f"{prefix}symline_*")):
        ticks.append(float(np.loadtxt(fname)[0, 0]))
    return BandStructure(
        kind="lines",
        kdist=kdist,
        bands=bands,
        ticks=sorted(ticks),
        tick_labels=[""] * len(ticks),
        energy_unit=energy_unit,
    )


_FATBAND_NAME = re.compile(r"fatbands\.(t\d+\.e\d+\.l\d+(?:\.m-?\d+)?)$")


def read_fatbands(prefix=".", energy_unit="eV"):
    """
    Read fatbands.tTT.eE.lL[.mM] files (columns: k-distance, energy, weight;
    xmgrace xydy layout). Returns a lines BandStructure with one weights
    entry per file, labeled by the projection part of the file name.
    """
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    fnames = sorted(glob.glob(f"{prefix}fatbands.*"))
    if not fnames:
        raise FileNotFoundError(f"No fatbands.* files found in {prefix or '.'}")
    bs = None
    weights = {}
    for fname in fnames:
        m = _FATBAND_NAME.search(os.path.basename(fname))
        if m is None:
            continue
        data = np.loadtxt(fname)
        kdist = np.unique(data[:, 0])
        nk = len(kdist)
        if data.shape[0] % nk != 0:
            raise ValueError(
                f"{fname}: {data.shape[0]} rows do not form bands over "
                f"{nk} k-points"
            )
        nbands = data.shape[0] // nk
        # rows are grouped per band (nk consecutive rows per band)
        energies = data[:, 1].reshape(nbands, nk).T
        weight = data[:, 2].reshape(nbands, nk).T
        if bs is None:
            bs = BandStructure(
                kind="lines",
                kdist=kdist,
                bands=energies,
                energy_unit=energy_unit,
            )
        weights[m.group(1)] = weight
    if bs is None:
        raise FileNotFoundError(f"No fatbands.* files with parsable names in {prefix}")
    bs.weights = weights
    return bs


def read_eigenvalues(fname="eigenvalues", reference=0.0, scale=1.0, nbands=None):
    """
    Read a raw RSPt eigenvalues file (new format: comment/header lines, then
    per k-point kx ky kz followed by the eigenvalues, free format).
    Detects symmetry-line breaks from changes in the k-path direction, like
    evconv does. Energies are returned as (e - reference) * scale.

    nbands: bands per k-point. If None, the smallest count that divides the
    data evenly is used — pass it explicitly if that guesses wrong.
    """
    with open(fname, "rt") as f:
        tokens = []
        for line in f:
            stripped = line.strip()
            if not stripped or stripped[0] in "#!c":
                continue
            try:
                row = [float(t) for t in stripped.split()]
            except ValueError:
                continue  # header line with words
            tokens.extend(row)
    values = np.array(tokens)
    # First record: 3 k components + nbands eigenvalues. Find nbands by
    # assuming all records are equally long.
    # Try increasing nbands until the total length divides evenly.
    total = len(values)
    if nbands is None:
        for nb in range(1, total - 2):
            if total % (3 + nb) == 0:
                nbands = nb
                break
    if nbands is None or total % (3 + nbands) != 0:
        raise ValueError(
            f"Could not infer the band count from {fname}; pass nbands explicitly"
        )
    records = values.reshape(-1, 3 + nbands)
    kvecs = records[:, :3]
    bands = (records[:, 3:] - reference) * scale
    # cumulative k-distance and direction breaks
    dk = np.diff(kvecs, axis=0)
    seglen = np.linalg.norm(dk, axis=1)
    kdist = np.concatenate([[0.0], np.cumsum(seglen)])
    ticks = [kdist[0]]
    for i in range(1, len(dk)):
        na, nb_ = seglen[i - 1], seglen[i]
        if na > 0 and nb_ > 0:
            cosang = np.dot(dk[i - 1], dk[i]) / (na * nb_)
            if cosang < 1.0 - 1e-6:
                ticks.append(kdist[i])
    ticks.append(kdist[-1])
    return BandStructure(
        kind="lines",
        kdist=kdist,
        bands=bands,
        ticks=ticks,
        tick_labels=[""] * len(ticks),
        energy_unit="eV" if abs(scale - Ry_to_eV) < 1e-2 else "Ry",
    )
