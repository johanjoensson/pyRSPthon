from argparse import ArgumentParser

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots

linestyles = ["-", "--", ":", "-."]


def run(args):
    import matplotlib.pyplot as plt
    import sys
    import numpy as np
    from pyRSPthon.cli._common import parse_orbital_selection, OrbitalSelectionError
    from pyRSPthon.cli._common import resolve_energy_unit, apply_unit_conversion, prepare_plot_data, extract_true_total_dos
    from pyRSPthon.cli import _bandplot as bp

    multi = len(args.cluster) > 1

    e_unit, conversion_factor = resolve_energy_unit(args.directory, getattr(args, 'eV', False))

    dos_list, valid_clusters, valid_orbitals = prepare_plot_data(args.cluster, args.directory, getattr(args, 'orbitals', None), data_type="pdos")
    
    if not dos_list:
        return
        
    for d in dos_list:
        apply_unit_conversion(d, conversion_factor, is_dos=True, data_type="pdos")

    true_total_dos = extract_true_total_dos(args.directory, conversion_factor)

    fig_total = plt.figure()
    ax_total = fig_total.add_subplot(111)
    
    if true_total_dos is not None:
        ax_total.fill_between(true_total_dos.w, true_total_dos.sum, alpha=0.2, color="tab:gray", label="Total System")
        
    ax_total.set_title("Projected density of states")
    ax_total.set_xlabel(rf"E - E$_F$ ({e_unit})")
    ax_total.set_ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")

    for i, (dos, cluster) in enumerate(zip(dos_list, valid_clusters)):
        color = f"C{i}"
        if dos.up is not None:
            ax_total.plot(dos.w, dos.up, color=color, linestyle="-", label=f"{cluster}: " + r"$\uparrow$")
            ax_total.plot(dos.w, dos.down, color=color, linestyle="--", label=f"{cluster}: " + r"$\downarrow$")
        else:
            if multi:
                ax_total.plot(dos.w, dos.sum, color=color, linestyle="-", label=f"{cluster}")

    figs_slj = {}
    for attr, name in [("s", "S"), ("l", "L"), ("j", "J")]:
        if any(getattr(d, attr) is not None for d in dos_list):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if true_total_dos is not None:
                ax.fill_between(true_total_dos.w, true_total_dos.sum, alpha=0.2, color="tab:gray", label="Total System")
            ax.set_title("Projected density of states")
            ax.set_xlabel(rf"E - E$_F$ ({e_unit})")
            ax.set_ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
            figs_slj[name] = (fig, ax)
            for i, (dos, cluster) in enumerate(zip(dos_list, valid_clusters)):
                data = getattr(dos, attr)
                if data is None: continue
                color = f"C{i}"
                for j, sub in enumerate("xyz"):
                    ax.plot(
                        dos.w, data[:, j], color=color, linestyle=linestyles[j % len(linestyles)], label=rf"{cluster}: {name}$_{sub}$"
                    )

    orbitals_dos = [(d, c, o, i) for i, (d, c, o) in enumerate(zip(dos_list, valid_clusters, valid_orbitals)) if d.orbitals is not None]
    if orbitals_dos:
        fig_orb = plt.figure()
        ax_orb = fig_orb.add_subplot(111)
        if true_total_dos is not None:
            ax_orb.fill_between(true_total_dos.w, true_total_dos.sum, alpha=0.2, color="tab:gray", label="Total System")
        ax_orb.set_title("Orbital projected density of states")
        ax_orb.set_xlabel(rf"E - E$_F$ ({e_unit})")
        ax_orb.set_ylabel(rf"pDOS ({e_unit}$^{{-1}}$)")
        
        for dos, cluster, orb_sel, cluster_idx in orbitals_dos:
            norb = dos.orbitals.shape[1]
            try:
                selection = parse_orbital_selection(orb_sel, norb)
            except OrbitalSelectionError as e:
                print(f"Warning: --orbitals selection invalid for {cluster} (norb={norb}): {e}. Skipping.", file=sys.stderr)
                continue
                
            labels = [f"Orbital {idx}" for idx in range(norb)]
            shells, cfflag = bp.find_cluster_shells(cluster, args.directory)
            if shells:
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
            for j, group in enumerate(selection):
                valid_group = [idx for idx in group if idx < norb]
                if not valid_group: continue
                
                if len(valid_group) == 1:
                    idx = valid_group[0]
                    sel_data.append(dos.orbitals[:, idx])
                    sel_labels.append(labels[idx] if idx < len(labels) else f"Orbital {idx}")
                else:
                    summed = np.sum(dos.orbitals[:, valid_group], axis=1)
                    sel_data.append(summed)
                    group_labels = [labels[idx] if idx < len(labels) else f"Orbital {idx}" for idx in valid_group]
                    sel_labels.append(" + ".join(group_labels))
                    
            if args.orbital_labels:
                for j, lab in enumerate(args.orbital_labels):
                    if j < len(sel_labels):
                        sel_labels[j] = lab
                        
            color = f"C{cluster_idx}"
            for j, (data_col, lab) in enumerate(zip(sel_data, sel_labels)):
                ax_orb.plot(
                    dos.w,
                    data_col,
                    color=color,
                    linestyle=linestyles[j % len(linestyles)],
                    label=f"{cluster}: {lab}",
                )

        ax_orb.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_total.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for name, (fig, ax) in figs_slj.items():
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    finish_plots(args)


def main():
    parser = ArgumentParser(description="Plot pdos-<cluster>.dat file")
    add_plot_arguments(parser, cluster=True, multiple_clusters=True)
    parser.add_argument(
        "--orbitals",
        default=None,
        type=str,
        nargs="+",
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
