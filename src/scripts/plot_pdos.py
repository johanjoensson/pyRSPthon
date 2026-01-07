from rspt_utils.read import extract_pdos
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os import getcwd
from itertools import cycle

linestyles = ["-", "--", ":", "-.", "-*"]


def run(cluster, directory):
    dos = extract_pdos(cluster, prefix=directory)
    _ = plt.figure()
    plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
    if dos.up is not None:
        plt.plot(dos.w, dos.up, label=r"$\uparrow$")
        plt.plot(dos.w, dos.down, "--", label=r"$\downarrow$")
    plt.title("Projected dcensity of state")
    plt.xlabel(r"E - E$_F$")
    plt.ylabel(r"pDOS")
    plt.legend()
    if dos.s is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.s[:, 0], linestyles[0], label=r"S$_x$")
        plt.plot(dos.w, dos.s[:, 1], linestyles[1], label=r"S$_y$")
        plt.plot(dos.w, dos.s[:, 2], linestyles[2], label=r"S$_z$")
        plt.title("Projected dcensity of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    if dos.l is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.l[:, 0], linestyles[0], label=r"L$_x$")
        plt.plot(dos.w, dos.l[:, 1], linestyles[1], label=r"L$_y$")
        plt.plot(dos.w, dos.l[:, 2], linestyles[2], label=r"L$_z$")
        plt.title("Projected dcensity of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    if dos.j is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.j[:, 0], linestyles[0], label=r"J$_x$")
        plt.plot(dos.w, dos.j[:, 1], linestyles[1], label=r"J$_y$")
        plt.plot(dos.w, dos.j[:, 2], linestyles[2], label=r"J$_z$")
        plt.title("Projected dcensity of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    if dos.orbitals is not None:
        _ = plt.figure()
        linecycler = cycle(linestyles)
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        for orb in range(dos.orbitals.shape[1]):
            plt.plot(
                dos.w, dos.orbitals[:, orb], next(linecycler), label=f"Orbital {orb}"
            )
        plt.title("Orbital projected density of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    plt.show()


def main():
    parser = ArgumentParser(description="Plot dos.dat file")
    parser.add_argument("cluster", type=str, help="Cluster to plot.")
    parser.add_argument(
        "-d",
        "--directory",
        default=f"{getcwd()}",
        type=str,
        help="Look for files in directory.",
    )
    args = parser.parse_args()
    run(**vars(args))
