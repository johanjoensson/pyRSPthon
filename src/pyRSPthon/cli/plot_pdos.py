from argparse import ArgumentParser
from itertools import cycle

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots
from pyRSPthon.read import extract_pdos

linestyles = ["-", "--", ":", "-."]


def run(args):
    import matplotlib.pyplot as plt

    dos = extract_pdos(args.cluster, prefix=args.directory)
    _ = plt.figure()
    plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
    if dos.up is not None:
        plt.plot(dos.w, dos.up, label=r"$\uparrow$")
        plt.plot(dos.w, dos.down, "--", label=r"$\downarrow$")
    plt.title("Projected density of states")
    plt.xlabel(r"E - E$_F$")
    plt.ylabel(r"pDOS")
    plt.legend()
    for name, data in (("S", dos.s), ("L", dos.l), ("J", dos.j)):
        if data is None:
            continue
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        for i, sub in enumerate("xyz"):
            plt.plot(
                dos.w, data[:, i], linestyle=linestyles[i], label=rf"{name}$_{sub}$"
            )
        plt.title("Projected density of states")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    if dos.orbitals is not None:
        _ = plt.figure()
        linecycler = cycle(linestyles)
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        for orb in range(dos.orbitals.shape[1]):
            plt.plot(
                dos.w,
                dos.orbitals[:, orb],
                linestyle=next(linecycler),
                label=f"Orbital {orb}",
            )
        plt.title("Orbital projected density of states")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    finish_plots(args)


def main():
    parser = ArgumentParser(description="Plot pdos-<cluster>.dat file")
    add_plot_arguments(parser, cluster=True)
    args = parser.parse_args()
    apply_plot_style(args)
    run(args)


if __name__ == "__main__":
    main()
