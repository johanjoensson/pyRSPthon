import os
import struct
import numpy as np

def peek_band_header(dataname):
    """
    Check if an RSPt binary band file has the new Band_header (starts with "RSPt").
    Returns a dict with dimensions {'n_data': ..., 'ne': ..., 'nk': ...} or None.
    """
    if not os.path.exists(dataname):
        return None
    with open(dataname, "rb") as f:
        magic = f.read(4)
        if magic == b"RSPt":
            version, hdr_size, ndim = struct.unpack("<iii", f.read(12))
            shape = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
            
            res = {}
            if ndim >= 3:
                res.update({'n_data': shape[0], 'ne': shape[1], 'nk': shape[2]})
            
            # The header was extended to include packed shell info at int 9 (offset 32)
            if hdr_size >= 36:
                f.seek(32)
                packed_bytes = f.read(4)
                if len(packed_bytes) == 4:
                    cfflag, ncorr, basis, l = struct.unpack("4B", packed_bytes)
                    if ncorr > 0:  # Only add if there are correlated shells (so we don't return 0s for band.data)
                        res.update({'cfflag': cfflag, 'ncorr': ncorr, 'basis': basis, 'l': l})
            
            if res:
                return res
    return None


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
