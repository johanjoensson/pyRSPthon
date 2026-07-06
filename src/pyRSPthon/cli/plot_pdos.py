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
        from pyRSPthon.cli._common import parse_orbital_selection
        from pyRSPthon.cli import _bandplot as bp
        import numpy as np

        norb = dos.orbitals.shape[1]
        selection = parse_orbital_selection(args.orbitals, norb)

        labels = [f"Orbital {i}" for i in range(norb)]
        shells, cfflag = bp.find_cluster_shells(args.cluster, args.directory)
        if shells:
            # pdos files carry no header, so infer the spin layout from the
            # column count; keep the generic labels on any mismatch.
            spin_guess = dos.up is not None
            for spin_split in (spin_guess, not spin_guess):
                candidate = bp.shell_labels(
                    shells, norb, spin_split=spin_split, cfflag=cfflag
                )
                if len(candidate) == norb:
                    labels = candidate
                    break

        sel_data = []
        sel_labels = []
        for i, group in enumerate(selection):
            if len(group) == 1:
                idx = group[0]
                sel_data.append(dos.orbitals[:, idx])
                sel_labels.append(labels[idx] if idx < len(labels) else f"Orbital {idx}")
            else:
                summed = np.sum(dos.orbitals[:, group], axis=1)
                sel_data.append(summed)
                group_labels = [labels[idx] if idx < len(labels) else f"Orbital {idx}" for idx in group]
                sel_labels.append(" + ".join(group_labels))
                
        if args.orbital_labels:
            for i, lab in enumerate(args.orbital_labels):
                if i < len(sel_labels):
                    sel_labels[i] = lab
                    
        _ = plt.figure()
        linecycler = cycle(linestyles)
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        
        for data_col, lab in zip(sel_data, sel_labels):
            plt.plot(
                dos.w,
                data_col,
                linestyle=next(linecycler),
                label=lab,
            )
            
        plt.title("Orbital projected density of states")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"pDOS")
        plt.legend()
    finish_plots(args)


def main():
    parser = ArgumentParser(description="Plot pdos-<cluster>.dat file")
    add_plot_arguments(parser, cluster=True)
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
