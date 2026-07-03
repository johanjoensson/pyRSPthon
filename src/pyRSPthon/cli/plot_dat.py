from pyRSPthon.read import read, green
from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from itertools import cycle, product

linestyles = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))]


def plot_dos(dos, e_unit):
    _ = plt.figure()
    plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
    if dos.up is not None:
        plt.plot(dos.w, dos.up, label=r"$\uparrow$")
        plt.plot(dos.w, dos.down, "--", label=r"$\downarrow$")
    plt.title("Density of states")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(rf"DOS ($({e_unit}^{{-1}})$)")
    plt.legend()
    if dos.s is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.s[:, 0], linestyle=linestyles[0], label=r"S$_x$")
        plt.plot(dos.w, dos.s[:, 1], linestyle=linestyles[1], label=r"S$_y$")
        plt.plot(dos.w, dos.s[:, 2], linestyle=linestyles[2], label=r"S$_z$")
        plt.title("Density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"DOS ($({e_unit}^{{-1}})$)")
        plt.legend()
    if dos.l is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.l[:, 0], linestyle=linestyles[0], label=r"L$_x$")
        plt.plot(dos.w, dos.l[:, 1], linestyle=linestyles[1], label=r"L$_y$")
        plt.plot(dos.w, dos.l[:, 2], linestyle=linestyles[2], label=r"L$_z$")
        plt.title("Density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"DOS ($({e_unit}^{{-1}})$)")
        plt.legend()
    if dos.j is not None:
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, dos.j[:, 0], linestyle=linestyles[0], label=r"J$_x$")
        plt.plot(dos.w, dos.j[:, 1], linestyle=linestyles[1], label=r"J$_y$")
        plt.plot(dos.w, dos.j[:, 2], linestyle=linestyles[2], label=r"J$_z$")
        plt.title("Density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"DOS ($({e_unit}^{{-1}})$)")
        plt.legend()



def plot_pdos(pdos, e_unit):
    _ = plt.figure()
    plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
    if pdos.up is not None:
        plt.plot(pdos.w, pdos.up, label=r"$\uparrow$")
        plt.plot(pdos.w, pdos.down, "--", label=r"$\downarrow$")
    plt.title("Projected density of states")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
    plt.legend()
    if pdos.s is not None:
        _ = plt.figure()
        plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(pdos.w, pdos.s[:, 0], linestyle=linestyles[0], label=r"S$_x$")
        plt.plot(pdos.w, pdos.s[:, 1], linestyle=linestyles[1], label=r"S$_y$")
        plt.plot(pdos.w, pdos.s[:, 2], linestyle=linestyles[2], label=r"S$_z$")
        plt.title("Projected density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        plt.legend()
    if pdos.l is not None:
        _ = plt.figure()
        plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(pdos.w, pdos.l[:, 0], linestyle=linestyles[0], label=r"L$_x$")
        plt.plot(pdos.w, pdos.l[:, 1], linestyle=linestyles[1], label=r"L$_y$")
        plt.plot(pdos.w, pdos.l[:, 2], linestyle=linestyles[2], label=r"L$_z$")
        plt.title("Projected density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        plt.legend()
    if pdos.j is not None:
        _ = plt.figure()
        plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(pdos.w, pdos.j[:, 0], linestyle=linestyles[0], label=r"J$_x$")
        plt.plot(pdos.w, pdos.j[:, 1], linestyle=linestyles[1], label=r"J$_y$")
        plt.plot(pdos.w, pdos.j[:, 2], linestyle=linestyles[2], label=r"J$_z$")
        plt.title("Projected density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        plt.legend()
    if pdos.orbitals is not None:
        _ = plt.figure()
        linecycler = cycle(linestyles)
        plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
        for orb in range(pdos.orbitals.shape[1]):
            plt.plot(
                pdos.w,
                pdos.orbitals[:, orb],
                linestyle=next(linecycler),
                label=f"Orbital {orb}",
            )
        plt.title("Orbital projected density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        plt.legend()



def plot_dat(cluster, dataset, dat, e_unit):
    _ = plt.figure()
    plt.fill_between(dat.w, dat.sum.real, alpha=0.2, color="tab:gray", label="Total")
    if dat.up is not None:
        plt.plot(dat.w, dat.up.real, label=r"$\uparrow$")
        plt.plot(dat.w, dat.down.real, "--", label=r"$\downarrow$")
    plt.title(f"{cluster} {dataset}")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(f"Re{{{dataset}}}")
    plt.legend()
    _ = plt.figure()
    plt.fill_between(dat.w, dat.sum.imag, alpha=0.2, color="tab:gray", label="Total")
    if dat.up is not None:
        plt.plot(dat.w, dat.up.imag, label=r"$\uparrow$")
        plt.plot(dat.w, dat.down.imag, "--", label=r"$\downarrow$")
    plt.title(f"{cluster} {dataset}")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(f"Im{{{dataset}}}")
    plt.legend()
    if dat.orbitals is not None:
        for block in dat.blocks:
            fig, ax = plt.subplots(
                nrows=len(block),
                ncols=len(block),
                squeeze=False,
                sharex="all",
                sharey="all",
            )
            linecycler = cycle(linestyles)
            for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                ax[i, j].plot(
                    dat.w,
                    dat.orbitals[:, orb_i, orb_j].real,
                    # next(linecycler),
                )
                ax[i, j].set_title(f"{orb_i}_{orb_j}")
            fig.suptitle(f"Orbital projected {dataset}")
            fig.supxlabel(rf"E - E$_F$ ({e_unit})")
            fig.supylabel(rf"Re{{{dataset}}}")
        for block in dat.blocks:
            fig, ax = plt.subplots(
                nrows=len(block),
                ncols=len(block),
                squeeze=False,
                sharex="all",
                sharey="all",
            )
            linecycler = cycle(linestyles)
            for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                ax[i, j].plot(
                    dat.w,
                    dat.orbitals[:, orb_i, orb_j].imag,
                    # next(linecycler),
                )
                ax[i, j].set_title(f"{orb_i}_{orb_j}")
            fig.suptitle(f"Orbital projected {dataset}")
            fig.supxlabel(rf"E - E$_F$ ({e_unit})")
            fig.supylabel(rf"Im{{{dataset}}}")



def run(args):
    cluster, data, directory, eV = args.cluster, args.data, args.directory, args.eV
    if not eV:
        try:
            green_dat = green.get_green(prefix=directory)
            eV = "eV" in green_dat.spectrum
        except (FileNotFoundError, TypeError):
            print(
                "No green.inp found, energies assumed to be in Ry "
                "(use --eV to override)."
            )
    e_unit = "eV" if eV else "Ry"

    if data.lower() == "dos":
        dat = read(f"{directory}/dos.dat")
        plot_dos(dat, e_unit)
    elif data.lower() == "pdos":
        dat = read(f"{directory}/pdos-{cluster}.dat")
        plot_pdos(dat, e_unit)
    else:
        dat = read(f"{directory}/{data}-{cluster}.dat")
        plot_dat(cluster, data, dat, e_unit)
    finish_plots(args)


def main():
    parser = ArgumentParser(
        description="Plot RSPt data files (dos, pdos, real-/imag-<dataset>)."
    )
    add_plot_arguments(parser, cluster=True)
    parser.add_argument("data", type=str, help="Dataset to plot (dos, pdos, hyb, ...)")
    parser.add_argument("--eV", action="store_true", help="Unit of energy is eV.")
    args = parser.parse_args()
    apply_plot_style(args)
    run(args)


if __name__ == "__main__":
    main()
