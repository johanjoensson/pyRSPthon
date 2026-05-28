from pyRSPthon.read import band
from pyRSPthon.read import green
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mp
from matplotlib import rc
import itertools
import numpy as np
from argparse import ArgumentParser
from os import getcwd

font = {"family": "sans-serif", "weight": "bold", "size": 20}
rc("font", **font)


def add_colors(c1, c2):
    c_out = np.empty_like(c1)
    c_out[3] = c1[3] + c2[3] * (1 - c1[3])
    c_out[:3] = c1[:3] * c1[np.newaxis, 3] + c2[:3] * (1 - c1[np.newaxis, 3])

    return c_out
    # return (
    #     (c1[0] * c1[3] + c2[0] * c2[3] * (1 - c1[3])) / a_out,
    #     (c1[1] * c1[3] + c2[1] * c2[3] * (1 - c1[3])) / a_out,
    #     (c1[2] * c1[3] + c2[2] * c2[3] * (1 - c1[3])) / a_out,
    #     a_out,
    # )
    # return (
    #     (c1[0] + c2[0] * (1 - c1[3])),
    #     (c1[1] + c2[1] * (1 - c1[3])),
    #     (c1[2] + c2[2] * (1 - c1[3])),
    #     a_out,
    # )


def get_color(alpha, colors):
    out_colors = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=float)
    for row, col in itertools.product(range(alpha.shape[0]), range(alpha.shape[1])):
        for i in range(len(colors)):
            out_colors[row, col] = add_colors(
                out_colors[row, col],
                (colors[i][0], colors[i][1], colors[i][2], alpha[row, col, i]),
            )
    return out_colors


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


Ry_to_eV = 13.605703976


def run(
    cluster,
    read_green,
    num_k,
    num_e,
    emin,
    emax,
    xticks,
    labels,
    orb_start,
    directory,
    energy_unit,
):
    if energy_unit is None:
        energy_unit = "Ry"
    if read_green:
        green_dat = green.get_green(prefix=directory)

        assert (
            cluster in green_dat.clusters
        ), f"Cluster {cluster} not found in green.inp file"

        num_k, xticks, labels = get_kpath(num_k, xticks, labels, green_dat.kpath)
        assert (
            green_dat.emesh is not None
        ), f"Could not read energymesh information from green.inp in {directory}"
        if num_e is None:
            num_e = green_dat.emesh[0]
        if "eV" in green_dat.spectrum:
            energy_unit = "eV"
        else:
            energy_unit = "Ry"
        if emin is None:
            emin = green_dat.emesh[1]
            if energy_unit == "eV":
                emin *= Ry_to_eV
        if emax is None:
            emax = green_dat.emesh[2]
            if energy_unit == "eV":
                emax *= Ry_to_eV

    assert num_k is not None
    assert xticks is not None
    assert labels is not None
    assert num_e is not None
    assert emin is not None
    assert emax is not None

    bp = band.get_pband(cluster=cluster, nk=num_k, ne=num_e, prefix=directory)
    plt.imshow(
        bp[::-1, :, 0] - np.min(bp[:, :, 0]),
        extent=(1, num_k, emin, emax),
        aspect="auto",
        cmap="Blues_r",
    )
    plt.hlines(y=0, xmin=1, xmax=num_k, linestyle="dotted", color="gray")
    plt.vlines(x=xticks, ymin=emin, ymax=emax, linestyle="dotted", color="gray")
    plt.xlim(left=1, right=num_k)
    plt.xticks(xticks, labels)
    plt.colorbar(pad=0.05, spacing="uniform")
    plt.ylabel(f"E - E$_F$ ({energy_unit})")
    plt.show(block=True)
    for i in range(orb_start):
        plt.imshow(
            bp[::-1, :, i] - np.min(bp[:, :, i]),
            extent=(1, num_k, emin, emax),
            aspect="auto",
            cmap="Blues_r",
        )
        plt.hlines(y=0, xmin=1, xmax=num_k, linestyle="dotted", color="gray")
        plt.vlines(x=xticks, ymin=emin, ymax=emax, linestyle="dotted", color="gray")
        plt.xlim(left=1, right=num_k)
        plt.xticks(xticks, labels)
        plt.colorbar(pad=0.05, spacing="uniform")
        plt.ylabel(f"E - E$_F$ ({energy_unit})")
        plt.show(block=True)

    orbital_character = np.empty((bp.shape[0], bp.shape[1], 5), dtype=float)
    orbital_character[:, :, 0] = np.sum(bp[:, :, [orb_start, orb_start + 5]], axis=2)
    orbital_character[:, :, 1] = np.sum(
        bp[:, :, [orb_start + 1, orb_start + 6]], axis=2
    )
    orbital_character[:, :, 2] = np.sum(
        bp[:, :, [orb_start + 2, orb_start + 7]], axis=2
    )
    orbital_character[:, :, 3] = np.sum(
        bp[:, :, [orb_start + 3, orb_start + 8]], axis=2
    )
    orbital_character[:, :, 4] = np.sum(
        bp[:, :, [orb_start + 4, orb_start + 9]], axis=2
    )
    orbital_character -= np.min(orbital_character)
    # orbital_character = np.log(orbital_character + 1)
    # orbital_character = orbital_character
    orbital_character = np.divide(orbital_character, np.max(orbital_character))
    colors = ["blue", "green", "red", "cyan", "magenta"]
    im = get_color(orbital_character, colors=[mc.to_rgb(c) for c in colors])
    im = np.multiply(im[:, :, :3], im[:, :, 3, np.newaxis])  # [:, :, np.newaxis])
    plt.imshow(im[::-1], extent=(1, num_k, emin, emax), aspect="auto")
    plt.hlines(y=0, xmin=1, xmax=num_k, linestyle="dotted", color="gray")
    plt.vlines(x=xticks, ymin=emin, ymax=emax, linestyle="dotted", color="gray")
    plt.xlim(left=1, right=num_k)
    plt.xticks(xticks, labels)
    plt.ylabel(f"E - E$_F$ ({energy_unit})")

    orbs = [r"d$_{z^2}$", r"d$_{x^2-y^2}$", r"d$_{yz}$", r"d$_{xz}$", r"d$_{xy}$"]
    patches = [mp.Patch(color=colors[i], label=orbs[i]) for i in range(len(orbs))]
    plt.legend(
        handles=patches,
        loc="upper right",
        borderaxespad=0,
        fontsize=12,
        title="Orbital character",
        title_fontsize=12,
        alignment="left",
    )
    plt.show()


def main():
    parser = ArgumentParser(description="Plot bandtructure from bands.data file")
    parser.add_argument("cluster", type=str, help="Cluster to plot.")
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
        type=int,
        default=None,
        help="Total number of kpoints in path",
    )
    parser.add_argument(
        "-ne",
        "--num_e",
        type=int,
        default=None,
        help="Total number of energy points in mesh",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=None,
        help="Lower energy bound in plot",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
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
    parser.add_argument(
        "--orb-start",
        default=1,
        type=int,
        help="Column of first local orbital",
    )
    parser.add_argument(
        "-eu",
        "--energy-unit",
        default=None,
        type=str,
        help="Used energy unit",
    )
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
