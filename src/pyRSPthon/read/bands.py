"""
Unified band-structure reading for the RSPt output zoo.

Sources:
- DMFT spectral functions: binary band.data / pband-<cluster>.data, with
  metadata from the optional Band_header, the sibling band.gpi gnuplot
  script and/or green.inp  -> SpectralBands
- DFT bands converted by evconv: bandfile_0..N (+ symline_* tick files,
  evconv.inp energy reference/scale)  -> LineBands
- Fatbands: fatbands.* files (k-distance, energy, weight; one block per
  band)  -> LineBands with Projections
- Raw eigenvalues files (plot without running evconv at all)  -> LineBands

read_bands() picks the source automatically. The models are immutable;
with_labels(), with_reference() and to_unit() return modified copies.
"""

import dataclasses
import glob
import os
import re
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pyRSPthon.orbitals import find_cluster_shells, projected_orbital_count
from pyRSPthon.units import RY_TO_EV, energy_scale

from ._band_binary import BandReadError, load_band_file, peek_band_header
from .green import get_green

__all__ = [
    "BandReadError",
    "KPath",
    "Projections",
    "SpectralBands",
    "LineBands",
    "read_bands",
    "detect_source",
    "read_spectral_bands",
    "read_bandfiles",
    "read_fatbands",
    "read_eigenvalues",
    "parse_band_gpi",
    "with_labels",
    "with_reference",
    "to_unit",
]


@dataclass(frozen=True)
class KPath:
    """
    Positions along the k-path (kdist) and the high-symmetry ticks on it.
    """

    kdist: np.ndarray
    ticks: tuple = ()
    tick_labels: tuple = ()


@dataclass(frozen=True)
class Projections:
    """
    Orbital (or other) projections of a band structure. weights holds one
    column per projection along its last axis: (ne, nk, norb) for spectral
    data, (nk, nbands, nproj) for line data. labels may be empty when they
    have to be derived from the shells (see pyRSPthon.orbitals.resolve_layout).
    """

    weights: np.ndarray
    labels: tuple = ()
    shells: tuple | None = None
    cfflag: bool = False
    fullrel: bool | None = None
    irrep: str | None = None


@dataclass(frozen=True)
class SpectralBands:
    """
    A spectral function A(k, E): total[ne, nk] on the energies mesh, plus
    the columns RSPt writes between the total and the orbital projections
    (extra[ne, nk, m]; the spin and orbital moment components sx, sy, sz,
    jx, jy, jz when present) and the orbital projections of a cluster.
    RSPt writes the mesh relative to the Fermi level, so shifted is True
    whenever the mesh is known.
    """

    kind: ClassVar[str] = "spectral"

    kpath: KPath
    energies: np.ndarray
    total: np.ndarray
    extra: np.ndarray | None = None
    projections: Projections | None = None
    energy_unit: str = "Ry"
    fermi_index: int | None = None
    shifted: bool = True

    @property
    def fermi_level(self):
        if self.fermi_index is not None:
            return float(self.energies[self.fermi_index])
        return 0.0 if self.shifted else None


@dataclass(frozen=True)
class LineBands:
    """
    Eigenvalue curves bands[nk, nbands] over kpath.kdist. shifted says
    whether the energies are relative to the Fermi level. source names the
    reader: each RSPt output measures kdist on its own scale (evconv
    bandfiles scaled to 0..1, eigenvalues from the k-vectors wrev writes
    divided by 2pi, fatbands from RSPt's dist(k) without that division), so
    only bands from the same source line up.
    """

    kind: ClassVar[str] = "lines"

    kpath: KPath
    bands: np.ndarray
    projections: Projections | None = None
    energy_unit: str = "Ry"
    shifted: bool = False
    source: str = ""

    @property
    def fermi_level(self):
        return 0.0 if self.shifted else None


# ---------------------------------------------------------------------------
# Transforms


def with_labels(model, ticks=None, labels=None):
    """
    Replace the tick positions and/or tick labels of a band model. Labels
    must match the number of ticks.
    """
    kpath = model.kpath
    new_ticks = tuple(float(t) for t in ticks) if ticks is not None else kpath.ticks
    if labels is not None:
        new_labels = tuple(labels)
    elif ticks is not None and len(new_ticks) != len(kpath.tick_labels):
        new_labels = ("",) * len(new_ticks)
    else:
        new_labels = kpath.tick_labels
    if new_labels and len(new_labels) != len(new_ticks):
        raise BandReadError(
            f"{len(new_labels)} tick labels given for {len(new_ticks)} ticks "
            f"(ticks at {', '.join(f'{t:g}' for t in new_ticks)})"
        )
    return dataclasses.replace(
        model, kpath=dataclasses.replace(kpath, ticks=new_ticks, tick_labels=new_labels)
    )


def with_reference(model, efermi):
    """
    Shift eigenvalue energies so that efermi (in the model's unit) is zero.
    """
    if model.kind != "lines":
        raise BandReadError("Spectral data are already relative to the Fermi level")
    return dataclasses.replace(model, bands=model.bands - efermi, shifted=True)


def to_unit(model, unit):
    """
    Convert eigenvalue energies to unit ("Ry" or "eV").
    """
    if model.energy_unit == unit:
        return model
    if model.kind != "lines":
        raise BandReadError(
            f"Spectral data are stored in {model.energy_unit}; they cannot be converted"
        )
    try:
        factor = energy_scale(model.energy_unit, unit)
    except ValueError as err:
        raise BandReadError(str(err)) from None
    return dataclasses.replace(model, bands=model.bands * factor, energy_unit=unit)


# ---------------------------------------------------------------------------
# Spectral functions (band.data / pband-<cluster>.data)

_GPI_XTICS = re.compile(r"set xtics\s*\((.*)\)")
_GPI_TIC = re.compile(r'"([^"]*)"\s+(-?\d+(?:\.\d*)?)')
_GPI_KY = re.compile(r"ky\(x\)\s*=\s*\(int\(x\)%\s*(\d+)\s*\)")
_GPI_RECORD = re.compile(r"binary record=\s*(\d+)")
_GPI_YLABEL = re.compile(r'set ylabel\s+"Energy \((\w+)\)"')
_GPI_ORB_LOOP = re.compile(r"do for \[ie=\s*(\d+)\s*:\s*(\d+)\s*\]")


def _float_or_none(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_band_gpi(fname):
    """
    Extract plot metadata from an RSPt band.gpi gnuplot script.

    Returns a dict with (present when found): ne, nk, ticks, tick_labels,
    energy_unit, energies (linear mesh reconstructed from the ytics),
    orb_start/orb_end (the orbital projection columns of a pband script).
    """
    meta = {}
    ytic_pairs = []
    with open(fname, "rt") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        m = _GPI_KY.search(line)
        if m:
            meta["ne"] = int(m.group(1))
        m = _GPI_YLABEL.search(line)
        if m:
            meta["energy_unit"] = m.group(1)
        m = _GPI_RECORD.search(line)
        if m and "record" not in meta:
            meta["record"] = int(m.group(1))
        m = _GPI_XTICS.search(line)
        if m:
            pairs = _GPI_TIC.findall(m.group(1))
            meta["ticks"] = [float(pos) for _, pos in pairs]
            meta["tick_labels"] = [
                "Γ" if lab in ("{/Symbol G}", "G") else lab for lab, _ in pairs
            ]
        if "set ytics" in line and not ytic_pairs:
            for val, pos in _GPI_TIC.findall(line):
                energy = _float_or_none(val)
                if energy is not None:
                    ytic_pairs.append((energy, float(pos)))
        m = _GPI_ORB_LOOP.search(line)
        if m and "orb_start" not in meta:
            if any("proj-" in nxt for nxt in lines[i + 1 : i + 4]):
                meta["orb_start"] = int(m.group(1)) - 1
                meta["orb_end"] = int(m.group(2))

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
        scale = RY_TO_EV if meta.get("energy_unit") == "eV" else 1.0
        meta["energies"] = np.linspace(emin * scale, emax * scale, ne)
    return meta


_IRREP_SUFFIX = re.compile(r"-irr(\d{4})$")


def _used_columns(data):
    """
    Number of data columns before the trailing all-zero padding.
    """
    nonzero = np.flatnonzero(np.any(data != 0, axis=(0, 1)))
    return int(nonzero[-1]) + 1 if len(nonzero) else data.shape[2]


def _read_fullrel(prefix):
    """Whether the run in prefix is fully relativistic (None if unknown)."""
    from .data import read_data

    try:
        return read_data(os.path.join(prefix, "data")).fullrel
    except FileNotFoundError:
        return None


def read_spectral_bands(prefix=".", cluster=None, nk=None, ne=None, orb_start=None):
    """
    Read a binary band.data (or pband-<cluster>.data) spectral function.
    Metadata resolution order: explicit arguments > Band_header > band.gpi >
    green.inp. For a cluster the orbital projection columns start at
    orb_start (default: from band.gpi, else from the header shells, else
    column 1) and run to the end of the gpi's orbital loop or, without a gpi,
    to the last column that is not all-zero padding.
    """
    basename = "band" if cluster is None else f"pband-{cluster}"
    data_file = os.path.join(prefix, f"{basename}.data")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"No {basename}.data in {prefix!r}")
    meta = _metadata_from_green(prefix)
    gpi_file = os.path.join(prefix, f"{basename}.gpi")
    if os.path.exists(gpi_file):
        meta.update(parse_band_gpi(gpi_file))

    header = peek_band_header(data_file)
    if header:
        if header["ndim"] != 3:
            raise BandReadError(
                f"{data_file} holds {header['ndim']}-dimensional (surface) "
                "data, which is not supported"
            )
        meta["nk"] = header["nk"]
        meta["ne"] = header["ne"]
        meta["energies"] = np.linspace(header["emin"], header["emax"], header["ne"])
        meta["energy_unit"] = "Ry" if abs(header["scale"] - 1.0) < 1e-3 else "eV"
        for key in ("fermi_index", "shells", "cfflag"):
            if key in header:
                meta[key] = header[key]

    nk = nk if nk is not None else meta.get("nk")
    ne = ne if ne is not None else meta.get("ne")
    if nk is None or ne is None:
        raise BandReadError(
            f"Could not determine nk/ne for {basename}.data: no usable "
            f"{basename}.gpi or green.inp found, and no Band_header in the data "
            "file; pass them explicitly (plot_band: -nk/-ne)."
        )
    if nk <= 0 or ne <= 0:
        raise BandReadError(f"nk and ne must be positive (got nk={nk}, ne={ne})")
    energies = meta.get("energies")
    energy_unit = meta.get("energy_unit", "Ry")
    shifted = True
    if energies is None or len(energies) != ne:
        energies = np.arange(ne, dtype=float)
        energy_unit = "mesh index"
        shifted = False
    fermi_index = meta.get("fermi_index")
    if fermi_index is not None and not 0 <= fermi_index < ne:
        fermi_index = None

    data = load_band_file(data_file, nk, ne)
    n_data = data.shape[2]
    projections = None
    extra = data[:, :, 1:] if n_data > 1 else None
    if cluster is not None:
        shells, cfflag = meta.get("shells"), meta.get("cfflag", False)
        # With the spectrum Cf flag RSPt writes one file per irreducible
        # representation, pband-<cluster>-irrNNab, holding only that block's
        # orbitals; its header still describes the whole cluster.
        m_irrep = _IRREP_SUFFIX.search(cluster)
        irrep = m_irrep.group(1) if m_irrep else None
        # RSPt sizes the orbital block of every pband record for the largest
        # cluster (or irrep block), so smaller ones end in all-zero padding
        n_used = _used_columns(data)
        orb_end = meta.get("orb_end")
        if orb_start is None:
            orb_start = meta.get("orb_start")
        if orb_start is None and shells and irrep is None:
            # the header lists every projected set, so it fixes the orbital
            # count (green.inp lookups below give only the correlated ones)
            norb = projected_orbital_count(shells)
            if norb < n_used:
                orb_start = n_used - norb
        if shells is None:
            base = cluster[: m_irrep.start()] if m_irrep else cluster
            shells, cfflag = find_cluster_shells(base, prefix)
        if orb_start is None:
            orb_start = 1
        if orb_end is None:
            orb_end = n_used
        if not 1 <= orb_start < orb_end <= n_data:
            raise BandReadError(
                f"Orbital columns {orb_start}..{orb_end - 1} do not fit in the "
                f"{n_data} data columns of {basename}.data"
            )
        projections = Projections(
            weights=data[:, :, orb_start:orb_end],
            shells=tuple(shells) if shells else None,
            cfflag=bool(cfflag),
            fullrel=_read_fullrel(prefix),
            irrep=irrep,
        )
        extra = data[:, :, 1:orb_start] if orb_start > 1 else None

    return SpectralBands(
        kpath=KPath(
            kdist=np.arange(nk, dtype=float),
            ticks=tuple(float(t) for t in meta.get("ticks", ())),
            tick_labels=tuple(meta.get("tick_labels", ())),
        ),
        energies=np.asarray(energies, dtype=float),
        total=data[:, :, 0],
        extra=extra,
        projections=projections,
        energy_unit=energy_unit,
        fermi_index=fermi_index,
        shifted=shifted,
    )


# ---------------------------------------------------------------------------
# Eigenvalue curves


def _unit_from_scale(scale):
    if abs(scale - RY_TO_EV) < 1e-2:
        return "eV"
    if abs(scale - 1.0) < 1e-6:
        return "Ry"
    return f"{scale:g} × Ry"


def _read_evconv_inp(prefix):
    """(reference_energy, energy_scale_factor) from evconv.inp, or None."""
    try:
        with open(os.path.join(prefix, "evconv.inp"), "rt") as f:
            values = f.read().split()
        return float(values[0]), float(values[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def read_bandfiles(prefix="."):
    """
    Read evconv output: bandfile_0..N (first column the k-distance, scaled to
    0..1, then up to 10 band columns each) and symline_* tick markers. The
    energy reference and unit come from evconv.inp when it is present;
    otherwise the bands are taken to be in eV relative to E_F.
    """
    fnames = sorted(
        glob.glob(os.path.join(prefix, "bandfile_*")),
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if not fnames:
        raise FileNotFoundError(f"No bandfile_* files found in {prefix!r}")
    kdist = None
    columns = []
    for fname in fnames:
        data = np.atleast_2d(np.loadtxt(fname, comments=("#", "&")))
        if kdist is None:
            kdist = data[:, 0]
        elif len(data) != len(kdist):
            raise BandReadError(
                f"{fname} has {len(data)} k-points, {fnames[0]} has {len(kdist)}"
            )
        columns.append(data[:, 1:])
    ticks = []
    for fname in glob.glob(os.path.join(prefix, "symline_*")):
        ticks.append(float(np.atleast_2d(np.loadtxt(fname))[0, 0]))
    ticks = [kdist[0]] + sorted(ticks) + [kdist[-1]]

    evconv = _read_evconv_inp(prefix)
    energy_unit = _unit_from_scale(evconv[1]) if evconv else "eV"
    return LineBands(
        kpath=KPath(kdist=kdist, ticks=tuple(ticks), tick_labels=("",) * len(ticks)),
        bands=np.hstack(columns),
        energy_unit=energy_unit,
        shifted=True,
        source="bandfiles",
    )


_FATBAND_NAME = re.compile(
    r"^fatbands\.((?:t\d+(?:\.e\d+\.l\d+(?:\.m\s*-?\d+)?)?)?(?:\.?s_(?:up|dn))?)$"
)


def _read_fatband_file(fname):
    """
    One fatbands file: blocks of (k-distance, energy, weight) rows, one block
    per band, separated by blank lines. Returns (kdist, energies[nk, nb],
    weights[nk, nb]).
    """
    blocks, current = [], []
    with open(fname, "rt") as f:
        for line in f:
            if line.strip():
                current.append([float(x) for x in line.split()[:3]])
            elif current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    if not blocks:
        raise BandReadError(f"{fname} holds no data")
    nk = len(blocks[0])
    if any(len(b) != nk for b in blocks):
        raise BandReadError(f"{fname}: bands have different numbers of k-points")
    data = np.array(blocks)  # (nbands, nk, 3)
    return data[0, :, 0], data[:, :, 1].T, data[:, :, 2].T


def read_fatbands(prefix="."):
    """
    Read the fatbands.* files RSPt writes (bandplots.F90): names
    fatbands.tTT[.eE.lL[.mMM]][.s_up|.s_dn] or fatbands.s_up/.s_dn. Energies
    are raw eigenvalues in Ry. Returns LineBands whose projections hold one
    column per file, labelled by the projection part of the file name.
    """
    fnames = sorted(glob.glob(os.path.join(prefix, "fatbands.*")))
    if not fnames:
        raise FileNotFoundError(f"No fatbands.* files found in {prefix!r}")
    kdist = bands = None
    weights, labels = [], []
    for fname in fnames:
        m = _FATBAND_NAME.match(os.path.basename(fname))
        if m is None or not m.group(1):
            continue
        kd, energies, weight = _read_fatband_file(fname)
        if bands is None:
            kdist, bands = kd, energies
        elif energies.shape != bands.shape:
            raise BandReadError(
                f"{fname} has {energies.shape[1]} bands over {energies.shape[0]} "
                f"k-points, expected {bands.shape[1]} over {bands.shape[0]}"
            )
        weights.append(weight)
        labels.append(m.group(1).replace(" ", ""))
    if bands is None:
        raise FileNotFoundError(f"No fatbands.* files with RSPt names in {prefix!r}")
    # A repeated k-distance marks a joint between two symmetry lines
    joints = [kdist[i] for i in range(1, len(kdist)) if kdist[i] == kdist[i - 1]]
    ticks = [kdist[0]] + joints + [kdist[-1]]
    return LineBands(
        kpath=KPath(kdist=kdist, ticks=tuple(ticks), tick_labels=("",) * len(ticks)),
        bands=bands,
        projections=Projections(
            weights=np.stack(weights, axis=-1), labels=tuple(labels)
        ),
        energy_unit="Ry",
        shifted=False,
        source="fatbands",
    )


def _kpath_from_kvecs(kvecs):
    """
    Cumulative k-distance and high-symmetry ticks for a list of k-points,
    following evconv's conventions: a repeated k-point is a joint between
    two lines (the duplicate is dropped and gets a tick), and a change of
    direction is a corner (tick at the corner point). Paths are taken to be
    connected, as RSPt's kpath tool writes them.

    Returns (keep, kdist, ticks): keep indexes the k-points that are kept and
    kdist has one entry per kept point.
    """
    kvecs = np.asarray(kvecs, dtype=float)
    keep = [0]
    tick_idx = {0}
    for i in range(1, len(kvecs)):
        if np.allclose(kvecs[i], kvecs[keep[-1]], atol=1e-9):
            tick_idx.add(len(keep) - 1)
        else:
            keep.append(i)
    k = kvecs[keep]
    dk = np.diff(k, axis=0)
    seglen = np.linalg.norm(dk, axis=1)
    kdist = np.concatenate([[0.0], np.cumsum(seglen)])
    tick_idx.add(len(k) - 1)
    for i in range(1, len(dk)):
        cosang = np.dot(dk[i - 1], dk[i]) / (seglen[i - 1] * seglen[i])
        if cosang < 1.0 - 1e-6:
            tick_idx.add(i)
    ticks = sorted({float(kdist[i]) for i in tick_idx})
    return np.array(keep), kdist, ticks


def _parse_wrev_records(fname, lines):
    """(kvec, eigenvalues) records of an eigenvalues file with '#' separators."""
    records = []
    current = None
    for lineno, stripped in lines:
        if stripped.startswith("#"):
            if current is not None and current[1]:
                records.append(current)
                current = None
            continue
        try:
            values = [float(t) for t in stripped.split()]
        except ValueError:
            raise BandReadError(f"{fname}:{lineno}: cannot parse {stripped!r}")
        if current is None:
            if len(values) != 3:
                raise BandReadError(
                    f"{fname}:{lineno}: expected a 'kx ky kz' line, got {stripped!r}"
                )
            current = (values, [])
        else:
            current[1].extend(values)
    if current is not None and current[1]:
        records.append(current)
    return records


def _parse_flat_records(fname, lines, nbands):
    """
    Records of an eigenvalues file without '#' separators (older or
    hand-made files): a stream of kx ky kz + nbands values per k-point,
    after any all-text header lines. The band count cannot be inferred.
    """
    if nbands is None:
        raise BandReadError(
            f"{fname} has no '#' lines separating the k-points (not RSPt's "
            "current eigenvalues format); pass the band count (--nbands)"
        )
    values = []
    for lineno, stripped in lines:
        tokens = stripped.split()
        numbers = [_float_or_none(t) for t in tokens]
        if all(v is None for v in numbers) and not values:
            continue  # header line
        if any(v is None for v in numbers):
            raise BandReadError(f"{fname}:{lineno}: cannot parse {stripped!r}")
        values.extend(numbers)
    width = 3 + nbands
    if not values or len(values) % width:
        raise BandReadError(
            f"{fname} holds {len(values)} numbers, not a whole number of "
            f"k-points with {nbands} bands ({width} numbers each)"
        )
    rows = np.array(values).reshape(-1, width)
    return [(row[:3], row[3:]) for row in rows]


def read_eigenvalues(fname="eigenvalues", nbands=None):
    """
    Read a raw RSPt eigenvalues file (bandplots.F90 wrev): per k-point a
    block of '#' comment lines, one line kx ky kz, then the eigenvalues four
    per line. Returns LineBands in Ry, not shifted; the band count comes
    from the records (nbands, if given, is checked against it). Files
    without the '#' lines need nbands to be split into k-points.
    Symmetry-line breaks are detected from the k-path (see _kpath_from_kvecs).
    """
    with open(fname, "rt") as f:
        lines = [(n, line.strip()) for n, line in enumerate(f, start=1) if line.strip()]
    if any(stripped.startswith("#") for _, stripped in lines):
        records = _parse_wrev_records(fname, lines)
    else:
        records = _parse_flat_records(fname, lines, nbands)
    if not records:
        raise BandReadError(f"No eigenvalue records found in {fname}")

    counts = {len(ev) for _, ev in records}
    if len(counts) != 1:
        raise BandReadError(
            f"{fname}: k-points have different numbers of eigenvalues {sorted(counts)}"
        )
    found = counts.pop()
    if nbands is not None and nbands != found:
        raise BandReadError(f"{fname} has {found} eigenvalues per k-point, not {nbands}")
    kvecs = np.array([kv for kv, _ in records])
    bands = np.array([ev for _, ev in records])
    keep, kdist, ticks = _kpath_from_kvecs(kvecs)
    return LineBands(
        kpath=KPath(kdist=kdist, ticks=tuple(ticks), tick_labels=("",) * len(ticks)),
        bands=bands[keep],
        energy_unit="Ry",
        shifted=False,
        source="eigenvalues",
    )


# ---------------------------------------------------------------------------
# Entry point

SOURCES = ("spectral", "bandfiles", "fatbands", "eigenvalues")


def detect_source(directory):
    """The band data source present in directory (see SOURCES)."""
    if os.path.exists(os.path.join(directory, "band.data")):
        return "spectral"
    if os.path.exists(os.path.join(directory, "bandfile_0")):
        return "bandfiles"
    if glob.glob(os.path.join(directory, "fatbands.*")):
        return "fatbands"
    if os.path.exists(os.path.join(directory, "eigenvalues")):
        return "eigenvalues"
    raise FileNotFoundError(
        f"No band data found in {directory!r} (looked for band.data, "
        "bandfile_0, fatbands.*, eigenvalues)"
    )


def read_bands(path=".", source=None, cluster=None, nk=None, ne=None, nbands=None, orb_start=None):
    """
    Read the band structure in directory path. source is one of SOURCES
    (default: auto-detect); cluster selects pband-<cluster>.data for the
    spectral source. Returns SpectralBands or LineBands.
    """
    source = source or ("spectral" if cluster else detect_source(path))
    if source == "spectral":
        return read_spectral_bands(prefix=path, cluster=cluster, nk=nk, ne=ne, orb_start=orb_start)
    if source == "bandfiles":
        return read_bandfiles(prefix=path)
    if source == "fatbands":
        return read_fatbands(prefix=path)
    if source == "eigenvalues":
        return read_eigenvalues(os.path.join(path, "eigenvalues"), nbands=nbands)
    raise BandReadError(f"Unknown band source {source!r} (expected one of {SOURCES})")
