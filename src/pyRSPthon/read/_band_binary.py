"""
Low-level access to RSPt's binary spectral files (band.data,
pband-<cluster>.data, surface.data): the optional Band_header and the
float32 payload. The public entry points are in pyRSPthon.read.bands.
"""

import os
import struct

import numpy as np


class BandReadError(ValueError):
    """Band data that cannot be read or does not match its metadata."""


def peek_band_header(dataname):
    """
    Check if an RSPt binary band file starts with a Band_header and parse it.

    Layout (4-byte little-endian integer slots, written by build_band_header
    in RSPt's green_spectrum.F90; header size = 4*max(16, 13+nshells) bytes):
     1     magic string 'RSPt'
     2     format version (1)
     3     header size in bytes
     4     number of dimensions of the data array (3, or 4 for surface.data)
     5-8   shape of the data array (Fortran order, unused = 0)
     9     first energy of the mesh (float32, output units)
    10     last energy of the mesh (float32, output units)
    11     mesh index of the Fermi level (1-based; negative if unknown)
    12     scaling factor Ry -> output unit (float32; 1.0 = Ry)
    13     packed bytes (cfflag, nshells, ncorr, 0); zeros if unprojected.
           nshells counts all projected sets of orbitals, ncorr the
           correlated ones (which come first)
    14...  one integer per projected shell: packed bytes (type, l, basis, 0)
    ...    zero padding up to at least 16 integers

    Returns a dict with 'ndim', 'shape', 'emin', 'emax', 'scale' and, when
    applicable, 'n_data'/'ne'/'nk' (3-dim data), 'fermi_index' (0-based),
    'cfflag', 'ncorr' and 'shells' ([{'type', 'l', 'basis'}, ...]). Headers
    written before RSPt stored ncorr hold 0 in that byte; 'ncorr' is then
    None (unknown). Returns None if the file is missing or has no header.
    """
    if not os.path.exists(dataname):
        return None
    with open(dataname, "rb") as f:
        header = f.read(16)
        if len(header) < 16 or header[:4] != b"RSPt":
            return None
        version, hdr_size, ndim = struct.unpack("<iii", header[4:])
        if hdr_size < 64:
            return None
        rest = f.read(hdr_size - 16)
    if len(rest) < hdr_size - 16:
        return None
    shape = struct.unpack("<4i", rest[:16])[:ndim]
    emin, emax, nfermi, scale = struct.unpack("<ffif", rest[16:32])
    res = {
        "ndim": ndim,
        "shape": shape,
        "emin": emin,
        "emax": emax,
        "scale": scale,
    }
    if ndim == 3:
        res.update({"n_data": shape[0], "ne": shape[1], "nk": shape[2]})
    if nfermi >= 1:
        res["fermi_index"] = nfermi - 1
    cfflag, nshells, ncorr, _ = struct.unpack("4B", rest[32:36])
    if nshells > 0 and hdr_size >= 4 * (13 + nshells):
        shells = []
        for n in range(nshells):
            t, l, basis, _ = struct.unpack("4B", rest[36 + 4 * n : 40 + 4 * n])
            shells.append({"type": t, "l": l, "basis": basis})
        res.update(
            {"cfflag": bool(cfflag), "ncorr": ncorr or None, "shells": shells}
        )
    return res


def load_band_file(dataname, nk, ne):
    """
    Load an RSPt binary band file (float32, C order (nk, ne, n_data),
    optionally preceded by a Band_header).

    Returns a writable array of shape (ne, nk, n_data), where n_data is the
    number of data columns per (k, e) point (total weight, spin/orbital
    moments, projections...).
    """
    with open(dataname, "rb") as f:
        magic = f.read(4)
        if magic == b"RSPt":
            _version, hdr_size, _ndim = struct.unpack("<iii", f.read(12))
            f.seek(hdr_size)
        else:
            f.seek(0)
        data = f.read()

    n_vals = len(data) // 4
    if len(data) % 4 != 0 or n_vals % (nk * ne) != 0:
        raise BandReadError(
            f"{dataname} holds {len(data)} bytes ({n_vals} float32 values), "
            f"which is not divisible by nk*ne = {nk}*{ne} = {nk * ne}. "
            "Check the nk/ne values (e.g. from green.inp or band.gpi)."
        )
    n_data = n_vals // (nk * ne)
    band = np.frombuffer(data, dtype="<f4").reshape((nk, ne, n_data))
    return np.ascontiguousarray(np.transpose(band, (1, 0, 2)))
