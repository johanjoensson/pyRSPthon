from pyRSPthon.read import extract_dos
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os import getcwd


def run(directory):
    dos = extract_dos(prefix=directory)
    _ = plt.figure()
    plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
    if dos.up is not None:
        plt.plot(dos.w, dos.up, label=r"$\uparrow$")
        plt.plot(dos.w, dos.down, "--", label=r"$\downarrow$")
    plt.title("Density of state")
    plt.xlabel(r"E - E$_F$")
    plt.ylabel(r"DOS")
    plt.legend()
    if dos.s is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.s[:, 2], label=r"S$_z$")
        plt.title("Density of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"DOS")
        plt.legend()
    if dos.l is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.fill_between(dos.w, dos.lum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.l[:, 2], label=r"L$_z$")
        plt.title("Density of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"DOS")
        plt.legend()
    if dos.j is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.fill_between(dos.w, dos.jum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.j[:, 2], label=r"J$_z$")
        plt.title("Density of state")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"DOS")
        plt.legend()
    plt.show()


def main():
    parser = ArgumentParser(description="Plot dos.dat file")
    parser.add_argument(
        "-d",
        "--directory",
        default=f"{getcwd()}",
        type=str,
        help="Look for files in directory.",
    )
    args = parser.parse_args()
    run(**vars(args))
