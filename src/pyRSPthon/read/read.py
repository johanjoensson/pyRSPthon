from os.path import split
import re
from . import extract_dos, extract_pdos, extract_dat

# Assume the cluster label is the part between a '-' and '.dat'
# Capture the cluster label
cluster_pattern = re.compile(".+-([^-]+).dat")

# dataset pattern
dataset_pattern = re.compile("(?:real-|imag-)?(.+)-(?:[^-]+).dat")


def read(filepath):
    prefix, fname = split(filepath)
    if fname == "dos.dat":
        return extract_dos(prefix)
    match = re.match(cluster_pattern, fname)
    assert match is not None, f"Could not extract cluster label from {fname}"
    cluster = match.group(1)
    if fname == f"pdos-{cluster}.dat":
        return extract_pdos(cluster, prefix)
    match = re.match(dataset_pattern, fname)
    assert match is not None, f"Could not extract dataset name from {fname}"
    dataset = match.group(1)
    return extract_dat(dataset, cluster, prefix)
