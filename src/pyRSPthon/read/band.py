import os
import struct
import numpy as np

def peek_band_header(dataname):
    """
    Check if an RSPt binary band file starts with a Band_header and parse it.

    Layout (4-byte little-endian integer slots, written by build_band_header
    in RSPt's green_spectrum.F90; header size = 4*max(16, 13+ncorr) bytes):
     1     magic string 'RSPt'
     2     format version (1)
     3     header size in bytes
     4     number of dimensions of the data array (3 or 4)
     5-8   shape of the data array (Fortran order, unused = 0)
     9     first energy of the mesh (float32, output units)
    10     last energy of the mesh (float32, output units)
    11     mesh index of the Fermi level (1-based; negative if unknown)
    12     scaling factor Ry -> output unit (float32; 1.0 = Ry)
    13     packed bytes (cfflag, ncorr, 0, 0); zeros if unprojected
    14...  one integer per correlated shell: packed bytes (type, l, basis, 0)
    ...    zero padding up to at least 16 integers

    Returns a dict with 'ndim', 'shape', 'emin', 'emax', 'scale' and, when
    applicable, 'n_data'/'ne'/'nk' (3-dim data), 'fermi_index' (0-based),
    'cfflag', 'ncorr' and 'shells' ([{'type', 'l', 'basis'}, ...]).
    Returns None if the file is missing or has no header.
    """
    if not os.path.exists(dataname):
        return None
    with open(dataname, "rb") as f:
        header = f.read(16)
        if len(header) < 16 or header[:4] != b"RSPt":
            return None
        version, hdr_size, ndim = struct.unpack("<iii", header[4:])
        rest = f.read(hdr_size - 16)
    if len(rest) < hdr_size - 16 or hdr_size < 64:
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
    cfflag, ncorr, _, _ = struct.unpack("4B", rest[32:36])
    if ncorr > 0 and hdr_size >= 4 * (13 + ncorr):
        shells = []
        for n in range(ncorr):
            t, l, basis, _ = struct.unpack("4B", rest[36 + 4 * n : 40 + 4 * n])
            shells.append({"type": t, "l": l, "basis": basis})
        res.update({"cfflag": bool(cfflag), "ncorr": ncorr, "shells": shells})
    return res


def _load_band_file(dataname, nk, ne):
    """
    Load an RSPt binary band file (float32, C order, written by the
    spectrum/gnuplot output of the green routines).

    Returns an array of shape (ne, nk, n_data) where n_data is the number of
    data columns per (k, e) point (energy/total weight/projections...).
    """
    with open(dataname, "rb") as f:
        magic = f.read(4)
        if magic == b"RSPt":
            version, hdr_size, ndim = struct.unpack("<iii", f.read(12))
            f.seek(hdr_size)
            data = f.read()
        else:
            f.seek(0)
            data = f.read()

    n_vals = len(data) // 4
    if len(data) % 4 != 0 or n_vals % (nk * ne) != 0:
        raise ValueError(
            f"{dataname} holds {len(data)} bytes ({n_vals} float32 values), "
            f"which is not divisible by nk*ne = {nk}*{ne} = {nk * ne}. "
            "Check the nk/ne values (e.g. from green.inp or band.gpi)."
        )
    n_data = n_vals // (nk * ne)
    band = np.frombuffer(data, dtype="f").reshape((nk, ne, n_data), order="C")
    return np.transpose(band, (1, 0, 2))


def get_band(nk, ne, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    return _load_band_file(f"{prefix}band.data", nk, ne)


def get_pband(nk, ne, cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    return _load_band_file(f"{prefix}pband-{cluster}.data", nk, ne)
