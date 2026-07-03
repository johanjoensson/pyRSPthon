from collections import namedtuple
import re
import itertools
import numpy as np
import scipy as sp

RSPtGreen = namedtuple(
    "RSPtGreen", ["spectrum", "emesh", "matsubara", "kpath", "clusters"], defaults=None
)


id_pattern = re.compile(".*Id(.*)")


def remove_comments(line):
    idx = line.find("#")
    if idx != -1:
        line = line[:idx]
    idx = line.find("!")
    if idx != -1:
        line = line[:idx]
    return line


def extract_cluster(f):
    line = next(f)
    cluster = None
    line = remove_comments(line)
    if "Id" in line:
        match = re.match(id_pattern, line)
        assert match is not None, f"Could not extract cluster Id from {line}"
        cluster = match.group(1)
    else:
        line = next(f)
        line = remove_comments(line)
        tmp = line.split()
        cluster = "".join(tmp[:5])
    return cluster


def extract_spectrum(f):
    kpath = None
    line = next(f)
    line = remove_comments(line)
    res = line.split()
    if "band" in line.lower():
        kpath = extract_kpath(f)

    return res, kpath


def extract_emesh(f):
    line = next(f)
    line = remove_comments(line)
    tmp = line.split()
    res = (
        int(tmp[0]),
        float(tmp[1]),
        float(tmp[2]),
        float(tmp[3]),
    )
    return res


def extract_matsubara(f):
    line = next(f)
    line = remove_comments(line)
    tmp = line.split()
    res = (
        int(tmp[0]),
        int(tmp[1]),
        int(tmp[2]),
        int(tmp[3]),
    )
    return res


def extract_kpath(f):
    line = next(f)
    line = remove_comments(line)
    # Line segments first
    tmp = line.split()
    path_length = [int(i) for i in tmp]
    nk = sum(path_length) + len(path_length) + 1

    # Kpoint labels
    line = next(f)
    line = remove_comments(line)
    labels = line.split()
    xticks = [li for li in range(len(labels))]
    for i in range(1, len(labels)):
        xticks[i] += sum(path_length[:i])
    res = (nk, xticks, labels)
    return res


def get_green(prefix="."):
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    clusters = []
    spectrum = None
    emesh = None
    matsubara = None
    kpath = None
    with open(f"{prefix}green.inp", "r") as f:
        try:
            line = next(f)
            while line:
                fields = remove_comments(line).split()
                keyword = fields[0].lower() if fields else ""
                if keyword == "cluster":
                    clusters.append(extract_cluster(f))
                elif keyword == "spectrum":
                    spectrum, kpath = extract_spectrum(f)
                elif keyword == "energymesh":
                    emesh = extract_emesh(f)
                elif keyword == "matsubara":
                    matsubara = extract_matsubara(f)
                line = next(f)
        except StopIteration:
            pass
    return RSPtGreen(
        clusters=clusters,
        emesh=emesh,
        spectrum=spectrum,
        matsubara=matsubara,
        kpath=kpath,
    )
