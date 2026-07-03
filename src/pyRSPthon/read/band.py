import numpy as np


def _load_band_file(dataname, nk, ne):
    """
    Load an RSPt binary band file (float32, C order, written by the
    spectrum/gnuplot output of the green routines).

    Returns an array of shape (ne, nk, n_data) where n_data is the number of
    data columns per (k, e) point (energy/total weight/projections...).
    """
    with open(dataname, "rb") as f:
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
