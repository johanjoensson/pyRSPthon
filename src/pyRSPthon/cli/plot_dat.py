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



def plot_pdos(pdos, e_unit, args):
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
        from pyRSPthon.cli._common import parse_orbital_selection
        import numpy as np
        
        norb = pdos.orbitals.shape[1]
        selection = parse_orbital_selection(args.orbitals, norb)
        
        default_labels = [f"Orbital {i}" for i in range(norb)]
        labels = default_labels
        
        sel_data = []
        sel_labels = []
        for i, group in enumerate(selection):
            if len(group) == 1:
                idx = group[0]
                sel_data.append(pdos.orbitals[:, idx])
                sel_labels.append(labels[idx] if idx < len(labels) else f"Orbital {idx}")
            else:
                summed = np.sum(pdos.orbitals[:, group], axis=1)
                sel_data.append(summed)
                group_labels = [labels[idx] if idx < len(labels) else f"Orbital {idx}" for idx in group]
                sel_labels.append(" + ".join(group_labels))
                
        if args.orbital_labels:
            for i, lab in enumerate(args.orbital_labels):
                if i < len(sel_labels):
                    sel_labels[i] = lab
                    
        _ = plt.figure()
        linecycler = cycle(linestyles)
        plt.fill_between(pdos.w, pdos.sum, alpha=0.2, color="tab:gray", label="Total")
        for data_col, lab in zip(sel_data, sel_labels):
            plt.plot(
                pdos.w,
                data_col,
                linestyle=next(linecycler),
                label=lab,
            )
        plt.title("Orbital projected density of states")
        plt.xlabel(rf"E - E$_F$ ({e_unit})")
        plt.ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        plt.legend()



def plot_dat(cluster, dataset, dat, e_unit, args):
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
        from pyRSPthon.cli._common import parse_orbital_selection
        import numpy as np

        norb = dat.orbitals.shape[1]

        if dat.orbitals.ndim == 2:
            # Diagonal-only data (no indexmap): one line per selected column
            selection = parse_orbital_selection(args.orbitals, norb)
            sel_data = []
            sel_labels = []
            for group in selection:
                sel_data.append(np.sum(dat.orbitals[:, group], axis=1))
                sel_labels.append("+".join(str(idx) for idx in group))
            if args.orbital_labels:
                for i, lab in enumerate(args.orbital_labels):
                    if i < len(sel_labels):
                        sel_labels[i] = lab
            for part_name, part in (("Re", np.real), ("Im", np.imag)):
                _ = plt.figure()
                linecycler = cycle(linestyles)
                for data_col, lab in zip(sel_data, sel_labels):
                    plt.plot(
                        dat.w, part(data_col), linestyle=next(linecycler), label=lab
                    )
                plt.title(f"Orbital projected {dataset}")
                plt.xlabel(rf"E - E$_F$ ({e_unit})")
                plt.ylabel(f"{part_name}{{{dataset}}}")
                plt.legend()

        elif args.orbitals is None:
            # Fallback to RSPt's own block diagonal structure to avoid plotting zeros
            for block in dat.blocks:
                # Real part
                fig, ax = plt.subplots(
                    nrows=len(block), ncols=len(block), squeeze=False, sharex="all", sharey="all"
                )
                for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                    ax[i, j].plot(dat.w, dat.orbitals[:, orb_i, orb_j].real)
                    ax[i, j].set_title(f"{orb_i}_{orb_j}")
                fig.suptitle(f"Orbital projected {dataset}")
                fig.supxlabel(rf"E - E$_F$ ({e_unit})")
                fig.supylabel(rf"Re{{{dataset}}}")
                
                # Imag part
                fig, ax = plt.subplots(
                    nrows=len(block), ncols=len(block), squeeze=False, sharex="all", sharey="all"
                )
                for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                    ax[i, j].plot(dat.w, dat.orbitals[:, orb_i, orb_j].imag)
                    ax[i, j].set_title(f"{orb_i}_{orb_j}")
                fig.suptitle(f"Orbital projected {dataset}")
                fig.supxlabel(rf"E - E$_F$ ({e_unit})")
                fig.supylabel(rf"Im{{{dataset}}}")

        else:
            selection = parse_orbital_selection(args.orbitals, norb)
            default_labels = [f"{i}" for i in range(norb)]
            labels = default_labels

            sel_labels = []
            for i, group in enumerate(selection):
                group_labels = [labels[idx] if idx < len(labels) else f"{idx}" for idx in group]
                sel_labels.append("+".join(group_labels) if len(group) > 1 else group_labels[0])
                
            if args.orbital_labels:
                for i, lab in enumerate(args.orbital_labels):
                    if i < len(sel_labels):
                        sel_labels[i] = lab

            n_sel = len(selection)
            new_orbitals = np.zeros((dat.orbitals.shape[0], n_sel, n_sel), dtype=complex)
            for i, group_i in enumerate(selection):
                for j, group_j in enumerate(selection):
                    grid_i, grid_j = np.meshgrid(group_i, group_j, indexing='ij')
                    new_orbitals[:, i, j] = np.sum(dat.orbitals[:, grid_i, grid_j], axis=(1, 2))
                    
            fig, ax = plt.subplots(
                nrows=n_sel, ncols=n_sel, squeeze=False, sharex="all", sharey="all"
            )
            for i in range(n_sel):
                for j in range(n_sel):
                    ax[i, j].plot(dat.w, new_orbitals[:, i, j].real)
                    ax[i, j].set_title(f"{sel_labels[i]}_{sel_labels[j]}")
            fig.suptitle(f"Orbital projected {dataset}")
            fig.supxlabel(rf"E - E$_F$ ({e_unit})")
            fig.supylabel(rf"Re{{{dataset}}}")

            fig, ax = plt.subplots(
                nrows=n_sel, ncols=n_sel, squeeze=False, sharex="all", sharey="all"
            )
            for i in range(n_sel):
                for j in range(n_sel):
                    ax[i, j].plot(dat.w, new_orbitals[:, i, j].imag)
                    ax[i, j].set_title(f"{sel_labels[i]}_{sel_labels[j]}")
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
        plot_pdos(dat, e_unit, args)
    else:
        dat = read(f"{directory}/{data}-{cluster}.dat")
        plot_dat(cluster, data, dat, e_unit, args)
    finish_plots(args)


def main():
    parser = ArgumentParser(
        description="Plot RSPt data files (dos, pdos, real-/imag-<dataset>)."
    )
    add_plot_arguments(parser, cluster=True)
    parser.add_argument("data", type=str, help="Dataset to plot (dos, pdos, hyb, ...)")
    parser.add_argument("--eV", action="store_true", help="Unit of energy is eV.")
    parser.add_argument(
        "--orbitals",
        default=None,
        type=str,
        help='Orbitals to plot, e.g. "0,2,4", "0-4", or "0+1+2" (default: all)',
    )
    parser.add_argument(
        "--orbital-labels",
        default=None,
        type=str,
        nargs="+",
        help="Override the automatic orbital labels",
    )
    args = parser.parse_args()
    apply_plot_style(args)
    run(args)


if __name__ == "__main__":
    main()
