from os.path import split
import re
from . import extract_dos, extract_pdos, extract_dat

# Assume the cluster label is the part between the last '-' and '.dat'
cluster_pattern = re.compile(r".+-([^-]+)\.dat$")

# dataset pattern: optional real-/imag- prefix, dataset name, cluster label
dataset_pattern = re.compile(r"(?:real-|imag-)?(.+)-(?:[^-]+)\.dat$")


def read(filepath):
    """
    Read an RSPt output data file, dispatching on the file name.
    Supported: dos.dat, pdos-<cluster>.dat, real-/imag-<dataset>-<cluster>.dat
    """
    prefix, fname = split(filepath)
    if fname == "dos.dat":
        return extract_dos(prefix)
    match = cluster_pattern.match(fname)
    if match is None:
        raise ValueError(f"Could not extract cluster label from {fname!r}")
    cluster = match.group(1)
    if fname == f"pdos-{cluster}.dat":
        return extract_pdos(cluster, prefix)
    match = dataset_pattern.match(fname)
    if match is None:
        raise ValueError(f"Could not extract dataset name from {fname!r}")
    dataset = match.group(1)
    return extract_dat(dataset, cluster, prefix)
