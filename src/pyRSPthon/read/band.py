import numpy as np


def get_band(nk, ne, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}band.data"
    data = None
    with open(dataname, "rb") as f:
        data = f.read()

    n_data = (len(data) // 4) // (nk * ne)
    band = np.frombuffer(data, dtype="f").reshape((nk, ne, n_data), order="C")
    # band = np.empty((ne, nk, n_data), dtype="f")
    # for d in range(0, len(data), 4 * n_data):
    #     row = (d // (4 * n_data)) % ne
    #     col = (d // (4 * n_data)) // ne
    #     band[row, col] = np.frombuffer(data[d : d + n_data * 4], dtype="f")

    return np.transpose(band, (1, 0, 2))


def get_pband(nk, ne, cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}pband-{cluster}.data"
    data = None
    with open(dataname, "rb") as f:
        data = f.read()

    n_data = (len(data) // 4) // (nk * ne)
    pband = np.frombuffer(data, dtype="f").reshape((nk, ne, n_data), order="C")
    # pband = np.empty((ne, nk, n_data), dtype="f")
    # for d in range(0, len(data), 4 * n_data):
    #     row = (d // (4 * n_data)) % ne
    #     col = (d // (4 * n_data)) // ne
    #     pband[row, col] = np.frombuffer(data[d : d + n_data * 4], dtype="f")

    return np.transpose(pband, (1, 0, 2))
