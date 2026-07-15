from collections import namedtuple
import re
import itertools
import numpy as np
import scipy as sp

RSPtDOS = namedtuple(
    "RSPtDOS", ["w", "sum", "up", "down", "s", "l", "j"], defaults=None
)
RSPtpDOS = namedtuple(
    "RSPtpDOS", ["w", "sum", "up", "down", "s", "l", "j", "orbitals"], defaults=None
)
RSPtDAT = namedtuple(
    "RSPtDAT", ["w", "sum", "up", "down", "orbitals", "blocks"], defaults=None
)

# Column-label patterns, tried in order. More specific patterns (tot+, spin up)
# must come before the generic ones (total, up).
_COLUMN_PATTERNS = (
    ("energy", ("energy", "frequency")),
    ("up", ("tot+", "spin up")),
    ("down", ("tot-", "spin dn", "spin down")),
    ("total", ("total", "sum", "tot")),
    ("sx", ("sx",)),
    ("sy", ("sy",)),
    ("sz", ("sz",)),
    ("lx", ("lx",)),
    ("ly", ("ly",)),
    ("lz", ("lz",)),
    ("jx", ("jx",)),
    ("jy", ("jy",)),
    ("jz", ("jz",)),
    ("orbitals", ("orbitals",)),
    ("up", ("up",)),
    ("down", ("down", "dn")),
)


def split_header_columns(header):
    """
    Split a '#'-prefixed RSPt column header line into column labels.
    """
    header = header.strip().strip("#")
    header = header.replace(",", "  ")
    return re.split("  +", header.strip())


def match_columns(columns):
    """
    Map RSPt header column labels to column indices.

    Parameters:
    ===========
    columns: list[str] - column labels from the header line

    Returns:
    ========
    dict[str, int] - keys among: energy, total, up, down, sx..sz, lx..lz,
                     jx..jz, orbitals (index of the first orbital column).
                     Only matched labels are present.
    """
    matched = {}
    for i, label in enumerate(columns):
        label_l = label.lower()
        if not label_l:
            continue
        for name, needles in _COLUMN_PATTERNS:
            if any(needle in label_l for needle in needles):
                # First match wins; both for the pattern and for the column.
                matched.setdefault(name, i)
                break
        else:
            print(f"Unknown column label {label!r}")
    return matched


def _read_header_line(dataname):
    with open(dataname, "r") as f:
        header = next(f)
    return header


def _vector_columns(data, cols, names):
    """
    Collect columns named in names (e.g. ("sx","sy","sz")) into an (N, 3)
    array, or None if none of them are present.
    """
    if not any(name in cols for name in names):
        return None
    out = np.zeros((data.shape[0], len(names)))
    for k, name in enumerate(names):
        if name in cols:
            out[:, k] = data[:, cols[name]]
    return out


def _extract_dos_columns(dataname):
    """
    Shared implementation for dos.dat and pdos-*.dat: parse the header,
    load the data and pull out the standard columns.
    """
    header = _read_header_line(dataname)
    cols = match_columns(split_header_columns(header))

    if "energy" not in cols:
        raise RuntimeError(
            f"{dataname} does not contain an energy mesh! "
            "(no 'Energy' or 'Frequency' column in the header)"
        )
    if "total" not in cols:
        raise RuntimeError(
            f"{dataname} does not contain a total DOS column! "
            "(no 'Total' column in the header)"
        )

    dos = np.loadtxt(dataname)
    return dos, cols


def extract_pdos(cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}pdos-{cluster}.dat"
    dos, cols = _extract_dos_columns(dataname)

    orbitals = dos[:, cols["orbitals"]:] if "orbitals" in cols else None
    return RSPtpDOS(
        w=dos[:, cols["energy"]],
        sum=dos[:, cols["total"]],
        up=dos[:, cols["up"]] if "up" in cols else None,
        down=dos[:, cols["down"]] if "down" in cols else None,
        s=_vector_columns(dos, cols, ("sx", "sy", "sz")),
        l=_vector_columns(dos, cols, ("lx", "ly", "lz")),
        j=_vector_columns(dos, cols, ("jx", "jy", "jz")),
        orbitals=orbitals,
    )


def extract_dos(prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}dos.dat"
    dos, cols = _extract_dos_columns(dataname)

    return RSPtDOS(
        w=dos[:, cols["energy"]],
        sum=dos[:, cols["total"]],
        up=dos[:, cols["up"]] if "up" in cols else None,
        down=dos[:, cols["down"]] if "down" in cols else None,
        s=_vector_columns(dos, cols, ("sx", "sy", "sz")),
        l=_vector_columns(dos, cols, ("lx", "ly", "lz")),
        j=_vector_columns(dos, cols, ("jx", "jy", "jz")),
    )


def _read_dat_header(fname):
    """
    Read the header line and the optional indexmap comment block of a
    real-*/imag-* dat file.
    """
    indexmap = None
    with open(fname, "r") as f:
        header = next(f)
        if "indexmap" in next(f, ""):
            indexmap = []
            for line in f:
                if line[0] != "#":
                    break
                row = [int(i) for i in line.strip("#").split()]
                indexmap.append(row)
    return header, indexmap


def extract_dat(dataname, cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    realname = f"{prefix}real-{dataname}-{cluster}.dat"
    imagname = f"{prefix}imag-{dataname}-{cluster}.dat"
    try:
        header, indexmap = _read_dat_header(realname)
    except FileNotFoundError:
        try:
            header, indexmap = _read_dat_header(imagname)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find either {realname} or {imagname}.")
    if indexmap is not None:
        indexmap = np.array(indexmap, dtype=int)

    columns = split_header_columns(header)
    cols = match_columns(columns)
    if "energy" not in cols:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain an energy mesh! "
            "(The header does not include 'Energy' or 'Frequency')"
        )
    e_col = cols["energy"]

    try:
        re_dat = np.loadtxt(realname)
    except FileNotFoundError:
        print(f"Could not find file {realname}. Setting real part to 0.")
        re_dat = None
    try:
        im_dat = np.loadtxt(imagname)
    except FileNotFoundError:
        print(f"Could not find file {imagname}. Setting imaginary part to 0.")
        im_dat = None
    if re_dat is None:
        dat = 1j * im_dat
        # The energy mesh is real, even when read from the imaginary-part file
        dat[:, e_col] = dat[:, e_col].imag
    elif im_dat is None:
        dat = re_dat.astype(complex)
    else:
        dat = re_dat + 1j * im_dat
        dat[:, e_col] = dat[:, e_col].real

    # Columns beyond the labeled ones hold orbital-resolved data
    orb_start = cols.get("orbitals")
    if orb_start is None and len(columns) < dat.shape[1]:
        orb_start = len(columns)

    orb_data = None
    if indexmap is not None:
        orb_data = np.zeros(
            (dat.shape[0], indexmap.shape[0], indexmap.shape[1]), dtype=complex
        )
        for i, j in itertools.product(
            range(indexmap.shape[0]), range(indexmap.shape[1])
        ):
            if indexmap[i, j] == 0:
                continue
            orb_data[:, i, j] = dat[:, indexmap[i, j] - 1]
    elif orb_start is not None and orb_start < dat.shape[1]:
        orb_data = dat[:, orb_start:]

    if "total" in cols:
        sum_data = dat[:, cols["total"]]
    elif orb_data is not None and orb_data.ndim == 3:
        sum_data = np.sum(np.diagonal(orb_data, axis1=1, axis2=2), axis=1)
    elif orb_data is not None:
        sum_data = np.sum(orb_data, axis=1)
    else:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain either summed up data "
            "(header field 'total') or orbital resolved data (indexmap in header)"
        )

    # With spin polarization but no explicit spin columns, the diagonal
    # orbital entries hold the two spin channels. RSPt lays the orbitals out
    # per correlated shell (a spin-down block followed by that shell's spin-up
    # block), so look the shells up in green.inp to pick out the down/up
    # columns; fall back to a global first-half/second-half split when the
    # shells are unknown (single-shell case, where the two layouts coincide).
    up_data = None
    dn_data = None
    orb_diag = None
    if orb_data is not None and orb_data.shape[1] > 0:
        orb_diag = (
            np.diagonal(orb_data, axis1=1, axis2=2) if orb_data.ndim == 3 else orb_data
        )
    down_idx = up_idx = None
    if orb_diag is not None and orb_diag.shape[1] % 2 == 0:
        from pyRSPthon.orbitals import find_cluster_shells, spin_split_indices

        shells, _ = find_cluster_shells(cluster, prefix)
        split = spin_split_indices(shells)
        if split is not None and len(split[0]) + len(split[1]) == orb_diag.shape[1]:
            down_idx, up_idx = split
        else:
            half = orb_diag.shape[1] // 2
            down_idx = list(range(half))
            up_idx = list(range(half, orb_diag.shape[1]))
    if "down" in cols:
        dn_data = dat[:, cols["down"]]
    elif down_idx is not None:
        print(
            f"{realname}, {imagname} do not contain any spin down projected data. "
            "Assuming I can sum the spin-down orbital terms."
        )
        dn_data = np.sum(orb_diag[:, down_idx], axis=1)
    if "up" in cols:
        up_data = dat[:, cols["up"]]
    elif up_idx is not None:
        print(
            f"{realname}, {imagname} do not contain any spin up projected data. "
            "Assuming I can sum the spin-up orbital terms."
        )
        up_data = np.sum(orb_diag[:, up_idx], axis=1)

    blocks = None
    if indexmap is not None:
        n_blocks, block_idxs = sp.sparse.csgraph.connected_components(
            csgraph=sp.sparse.csr_matrix(indexmap > 0),
            directed=False,
            return_labels=True,
        )
        blocks = [[] for _ in range(n_blocks)]
        for orb_i, block_i in enumerate(block_idxs):
            blocks[block_i].append(orb_i)
    return RSPtDAT(
        w=dat[:, e_col].real,
        sum=sum_data,
        up=up_data,
        down=dn_data,
        orbitals=orb_data,
        blocks=blocks,
    )
