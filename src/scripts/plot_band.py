from pyRSPthon.read import band
from pyRSPthon.read import green
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import rc
import itertools
import numpy as np
from argparse import ArgumentParser
from os import getcwd

font = {"family": "sans-serif", "weight": "bold", "size": 20}
rc("font", **font)


def add_colors(c1, c2):
    return (
        c1[0] * c1[3] + c2[0] * c2[3],
        c1[1] * c1[3] + c2[1] * c2[3],
        c1[2] * c1[3] + c2[2] * c2[3],
        1.0,
    )


def get_color(dz2, dx2, dyz, dxz, dxy):
    col_dz2 = mc.to_rgb("blue")
    col_dx2 = mc.to_rgb("green")
    col_dyz = mc.to_rgb("red")
    col_dxz = mc.to_rgb("cyan")
    col_dxy = mc.to_rgb("magenta")

    max_val = np.max(
        np.abs(dz2) + np.abs(dx2) + np.abs(dyz) + np.abs(dxz) + np.abs(dxy)
    )
    adz2 = np.divide(dz2, max_val)
    adx2 = np.divide(dx2, max_val)
    adyz = np.divide(dyz, max_val)
    adxz = np.divide(dxz, max_val)
    adxy = np.divide(dxy, max_val)
    colors = np.zeros((dz2.shape[0], dz2.shape[1], 4), dtype=float)
    for row, col in itertools.product(range(dz2.shape[0]), range(dz2.shape[1])):
        colors[row, col] = [col_dz2[0], col_dz2[1], col_dz2[2], adz2[row, col]]
        colors[row, col] = add_colors(
            colors[row, col], (col_dx2[0], col_dx2[1], col_dx2[2], adx2[row, col])
        )
        colors[row, col] = add_colors(
            colors[row, col], [col_dyz[0], col_dyz[1], col_dyz[2], adyz[row, col]]
        )
        colors[row, col] = add_colors(
            colors[row, col], [col_dxz[0], col_dxz[1], col_dxz[2], adxz[row, col]]
        )
        colors[row, col] = add_colors(
            colors[row, col], [col_dxy[0], col_dxy[1], col_dxy[2], adxy[row, col]]
        )
    return colors


def get_kpath(num_k, xticks, labels, kpath):
    assert not (
        (num_k is None or xticks is None or labels is None) and kpath is None
    ), "num_k, xticks and/or labels are absent from the command line, and no kpath info is found in green.inp."
    if num_k is None:
        num_k = kpath[0]
    if xticks is None:
        xticks = kpath[1]
    if labels is None:
        labels = kpath[2]
    return num_k, xticks, labels


def run(read_green, num_k, num_e, emin, emax, xticks, labels, directory):
    if read_green:
        green_dat = green.get_green(prefix=directory)

        num_k, xticks, labels = get_kpath(num_k, xticks, labels, green_dat.kpath)
        assert (
            green_dat.emesh is not None
        ), f"Could not read energymesh information from green.inp in {directory}"
        if num_e is None:
            num_e = green_dat.emesh[0]
        if emin is None:
            emin = green_dat.emesh[1]
        if emax is None:
            emax = green_dat.emesh[2]
    assert num_k is not None
    assert xticks is not None
    assert labels is not None
    assert num_e is not None
    assert emin is not None
    assert emax is not None
    b = band.get_band(nk=num_k, ne=num_e, prefix=directory)
    plt.imshow(
        b[::-1, :, 0],
        extent=(0, num_k, emin, emax),
        aspect="auto",
    )
    plt.hlines(y=0, xmin=1, xmax=num_k, linestyle="dotted", color="gray")
    plt.vlines(x=xticks, ymin=emin, ymax=emax, linestyle="dotted", color="gray")
    plt.xlim(left=1, right=num_k)
    plt.xticks(xticks, labels)
    plt.show()


def main():
    parser = ArgumentParser(description="Plot bandtructure from bands.data file")
    parser.add_argument(
        "-d",
        "--directory",
        default=f"{getcwd()}",
        type=str,
        help="Look for files in directory",
    )
    parser.add_argument(
        "--read-green",
        action="store_true",
        help="Attempt to extract information from green.inp",
    )
    parser.add_argument(
        "-nk",
        "--num_k",
        default=None,
        type=int,
        help="Total number of kpoints in path",
    )
    parser.add_argument(
        "-ne",
        "--num_e",
        default=None,
        type=int,
        help="Total number of energy points in mesh",
    )
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Lower energy bound in plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Upper energy bound in plot",
    )
    parser.add_argument(
        "-x",
        "--xticks",
        default=None,
        type=int,
        nargs="+",
        help="Indices of special kpoints along path",
    )
    parser.add_argument(
        "-l",
        "--labels",
        default=None,
        type=str,
        nargs="+",
        help="Kpoint labels along path",
    )
    args = parser.parse_args()

    run(**vars(args))
