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


def extract_pdos(cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}pdos-{cluster}.dat"
    with open(dataname, "r") as f:
        header = next(f)
    assert "orbitals" in header.lower()
    header = header.strip()
    header = header.strip("#")
    columns = re.split("  +", header)

    e_col = -1
    tot_col = -1
    up_col = -1
    dn_col = -1
    sx_col = -1
    sy_col = -1
    sz_col = -1
    lx_col = -1
    ly_col = -1
    lz_col = -1
    jx_col = -1
    jy_col = -1
    jz_col = -1
    orb_start = -1
    for i, label in reversed(list(enumerate(columns))):
        if "energy" in label.lower() or "frequency" in label.lower():
            e_col = i
        elif "total" in label.lower():
            tot_col = i
        elif "spin up" in label.lower():
            up_col = i
        elif "spin dn" in label.lower():
            dn_col = i
        elif "sx" in label.lower():
            sx_col = i
        elif "sy" in label.lower():
            sy_col = i
        elif "sz" in label.lower():
            sz_col = i
        elif "lx" in label.lower():
            lx_col = i
        elif "ly" in label.lower():
            ly_col = i
        elif "lz" in label.lower():
            lz_col = i
        elif "jx" in label.lower():
            jx_col = i
        elif "jy" in label.lower():
            jy_col = i
        elif "jz" in label.lower():
            jz_col = i
        elif "orbitals" in label.lower():
            orb_start = i
        else:
            print(f"Unknown label {label}")

    pdos = np.loadtxt(dataname)
    if sx_col != -1 or sy_col != -1 or sz_col != -1:
        s = np.zeros((pdos.shape[0], 3))
        if sx_col != -1:
            s[:, 0] = pdos[:, sx_col]
        if sy_col != -1:
            s[:, 1] = pdos[:, sy_col]
        if sz_col != -1:
            s[:, 2] = pdos[:, sz_col]
    else:
        s = None
    if lx_col != -1 or ly_col != -1 or lz_col != -1:
        l = np.zeros((pdos.shape[0], 3))
        if lx_col != -1:
            l[:, 0] = pdos[:, lx_col]
        if ly_col != -1:
            l[:, 1] = pdos[:, ly_col]
        if lz_col != -1:
            l[:, 2] = pdos[:, lz_col]
    else:
        l = None
    if jx_col != -1 or jy_col != -1 or jz_col != -1:
        j = np.zeros((pdos.shape[0], 3))
        if jx_col != -1:
            j[:, 0] = pdos[:, jx_col]
        if jy_col != -1:
            j[:, 1] = pdos[:, jy_col]
        if jz_col != -1:
            j[:, 2] = pdos[:, jz_col]
    else:
        j = None
    if orb_start != -1:
        orb_dos = pdos[:, orb_start:]
    else:
        orb_dos = None
    return RSPtpDOS(
        w=pdos[:, e_col],
        sum=pdos[:, tot_col],
        up=pdos[:, up_col] if up_col != -1 else None,
        down=pdos[:, dn_col] if dn_col != -1 else None,
        s=s,
        l=l,
        j=j,
        orbitals=orb_dos if orb_start != -1 else None,
    )


def extract_dos(prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    dataname = f"{prefix}dos.dat"
    with open(dataname, "r") as f:
        header = next(f)
    header = header.strip()
    header = header.strip("#")
    columns = re.split("  +", header)

    e_col = -1
    tot_col = -1
    up_col = -1
    dn_col = -1
    sx_col = -1
    sy_col = -1
    sz_col = -1
    lx_col = -1
    ly_col = -1
    lz_col = -1
    jx_col = -1
    jy_col = -1
    jz_col = -1
    for i, label in reversed(list(enumerate(columns))):
        if "energy" in label.lower() or "frequency" in label.lower():
            e_col = i
        elif "total" in label.lower():
            tot_col = i
        elif "spin up" in label.lower():
            up_col = i
        elif "spin dn" in label.lower():
            dn_col = i
        elif "sx" in label.lower():
            sx_col = i
        elif "sy" in label.lower():
            sy_col = i
        elif "sz" in label.lower():
            sz_col = i
        elif "lx" in label.lower():
            lx_col = i
        elif "ly" in label.lower():
            ly_col = i
        elif "lz" in label.lower():
            lz_col = i
        elif "jx" in label.lower():
            jx_col = i
        elif "jy" in label.lower():
            jy_col = i
        elif "jz" in label.lower():
            jz_col = i
        else:
            print(f"Unknown label {label}")

    print(f"{sz_col=}")
    pdos = np.loadtxt(dataname)
    if sx_col != -1 or sy_col != -1 or sz_col != -1:
        s = np.zeros((pdos.shape[0], 3))
        if sx_col != -1:
            s[:, 0] = pdos[:, sx_col]
        if sy_col != -1:
            s[:, 1] = pdos[:, sy_col]
        if sz_col != -1:
            s[:, 2] = pdos[:, sz_col]
    else:
        s = None
    if lx_col != -1 or ly_col != -1 or lz_col != -1:
        l = np.zeros((pdos.shape[0], 3))
        if lx_col != -1:
            l[:, 0] = pdos[:, lx_col]
        if ly_col != -1:
            l[:, 1] = pdos[:, ly_col]
        if lz_col != -1:
            l[:, 2] = pdos[:, lz_col]
    else:
        l = None
    if jx_col != -1 or jy_col != -1 or jz_col != -1:
        j = np.zeros((pdos.shape[0], 3))
        if jx_col != -1:
            j[:, 0] = pdos[:, jx_col]
        if jy_col != -1:
            j[:, 1] = pdos[:, jy_col]
        if jz_col != -1:
            j[:, 2] = pdos[:, jz_col]
    else:
        j = None
    return RSPtDOS(
        w=pdos[:, e_col],
        sum=pdos[:, tot_col],
        up=pdos[:, up_col] if up_col != -1 else None,
        down=pdos[:, dn_col] if dn_col != -1 else None,
        s=s,
        l=l,
        j=j,
    )


def extract_dat(dataname, cluster, prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    realname = f"{prefix}real-{dataname}-{cluster}.dat"
    imagname = f"{prefix}imag-{dataname}-{cluster}.dat"
    indexmap = None
    try:
        with open(realname, "r") as f:
            header = next(f)
            if "indexmap" in next(f):
                indexmap = []
                for line in f:
                    if line[0] != "#":
                        break
                    line = line.strip("#")
                    row = [int(i) for i in line.split()]
                    indexmap.append(row)
    except FileNotFoundError:
        try:
            with open(imagname, "r") as f:
                header = next(f)
                if "indexmap" in next(f):
                    indexmap = []
                    for line in f:
                        if line[0] != "#":
                            break
                        line = line.strip("#")
                        row = [int(i) for i in line.split()]
                        indexmap.append(row)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find either {realname} or {imagname}.")
    if indexmap is not None:
        indexmap = np.array(indexmap, dtype=int)

    header = header.strip()
    header = header.strip("#")
    header = header.replace(",", "  ")
    columns = re.split("  +", header)

    e_col = -1
    tot_col = -1
    up_col = -1
    dn_col = -1
    orb_start = len(columns)
    for i, label in reversed(list(enumerate(columns))):
        if "energy" in label.lower() or "frequency" in label.lower():
            e_col = i
        elif (
            "total" in label.lower() or "sum" in label.lower() or "tot" in label.lower()
        ):
            tot_col = i
        elif (
            "tot+" in label.lower()
            or "spin up" in label.lower()
            or "up" in label.lower()
        ):
            up_col = i
        elif (
            "tot-" in label.lower()
            or "spin down" in label.lower()
            or "down" in label.lower()
            or "dn" in label.lower()
        ):
            dn_col = i
        elif "orbitals" in label.lower():
            orb_start = i
        else:
            print(f"Unknown label {label}")
    if e_col == -1:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain an energy mesh! (The header does not include 'Energy' or 'Frequency')"
        )
    try:
        re_dat = np.loadtxt(realname, dtype=complex)
    except FileNotFoundError:
        print(f"Could not find file {realname}. Setting real part to 0.")
        re_dat = None
    try:
        im_dat = np.loadtxt(imagname, dtype=complex)
    except FileNotFoundError:
        print(f"Could not find file {imagname}. Setting imaginary part to 0.")
        im_dat = None
    # dat = re_dat + 1j * im_dat
    if re_dat is None:
        dat = 1j * im_dat
        dat[:, 0] = dat[:, 0].imag
    elif im_dat is None:
        dat = re_dat
    else:
        dat = re_dat + 1j * im_dat
        dat[:, 0] = dat[:, 0].real
    if orb_start == -1 and len(columns) < dat.shape[1]:
        orb_start = len(columns)

    sum_data = None
    up_data = None
    dn_data = None
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
    elif orb_start != -1:
        orb_data = dat[:, orb_start:]
    if tot_col != -1:
        sum_data = dat[:, tot_col]
    elif tot_col == -1 and orb_data is not None:
        sum_data = np.sum(np.diagonal(orb_data, axis1=1, axis2=2), axis=1)
    elif tot_col == -1:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain either summed up data (header fields 'total') or orbital resolved data (indexmap in header)"
        )
    if dn_col != -1:
        dn_data = dat[:, dn_col]
    elif dn_col == -1 and orb_data is not None and orb_data.shape[1] % 2 == 0:
        print(
            f"{realname}, {imagname} do not contain any spin down projected data. Assuming I can sum the first half of the diagonal orbital terms."
        )
        if len(orb_data.shape) == 3:
            dn_data = np.sum(
                np.diagonal(orb_data, axis1=1, axis2=2)[:, : orb_data.shape[1] // 2],
                axis=1,
            )
        else:
            dn_data = np.sum(orb_data[:, : orb_data.shape[1] // 2], axis=1)
    if up_col != -1:
        up_data = dat[:, up_col]
    elif up_col == -1 and orb_data is not None and orb_data.shape[1] % 2 == 0:
        print(
            f"{realname}, {imagname} do not contain any spin down projected data. Assuming I can sum the first half of the diagonal orbital terms."
        )
        if len(orb_data.shape) == 3:
            up_data = np.sum(
                np.diagonal(orb_data, axis1=1, axis2=2)[:, orb_data.shape[1] // 2 :],
                axis=1,
            )
        else:
            up_data = np.sum(orb_data[:, orb_data.shape[1] // 2 :], axis=1)

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
