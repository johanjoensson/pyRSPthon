from argparse import ArgumentParser

from pyRSPthon.cli._common import add_orbital_arguments, add_plot_arguments, cli_main, finish_plots

linestyles = ["-", "--", ":", "-."]


def run(args):
    import matplotlib.pyplot as plt
    import sys
    from pyRSPthon.cli._common import resolve_energy_unit, apply_unit_conversion, prepare_plot_data, extract_true_total_dos
    from pyRSPthon.orbitals import (
        OrbitalSelectionError,
        find_cluster_shells,
        select_orbitals,
        shell_labels,
    )

    multi = len(args.cluster) > 1

    e_unit, conversion_factor = resolve_energy_unit(args.directory, getattr(args, 'eV', False))

    dos_list, valid_clusters, valid_orbitals = prepare_plot_data(args.cluster, args.directory, getattr(args, 'orbitals', None), data_type="pdos")
    
    if not dos_list:
        return
        
    dos_list = [
        apply_unit_conversion(d, conversion_factor, is_dos=True, data_type="pdos")
        for d in dos_list
    ]

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
            labels = [f"Orbital {idx}" for idx in range(norb)]
            shells, cfflag = find_cluster_shells(cluster, args.directory)
            if shells:
                spin_guess = dos.up is not None
                for spin_split in (spin_guess, not spin_guess):
                    candidate = shell_labels(
                        shells, norb, spin_split=spin_split, cfflag=cfflag
                    )
                    if len(candidate) == norb:
                        labels = candidate
                        break
            try:
                sel_data, sel_labels = select_orbitals(
                    dos.orbitals, labels, orb_sel, args.orbital_labels
                )
            except OrbitalSelectionError as e:
                print(f"Warning: --orbitals selection invalid for {cluster} (norb={norb}): {e}. Skipping.", file=sys.stderr)
                continue

            color = f"C{cluster_idx}"
            for j, lab in enumerate(sel_labels):
                ax_orb.plot(
                    dos.w,
                    sel_data[:, j],
                    color=color,
                    linestyle=linestyles[j % len(linestyles)],
                    label=f"{cluster}: {lab}",
                )
        ax_orb.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_orb.subplots_adjust(right=0.75)

    ax_total.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_total.subplots_adjust(right=0.75)
    for name, (fig, ax) in figs_slj.items():
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.subplots_adjust(right=0.75)
        
    finish_plots(args)


def main():
    parser = ArgumentParser(prog="plot_pdos", description="Plot pdos-<cluster>.dat file")
    add_plot_arguments(parser, cluster=True, multiple_clusters=True)
    add_orbital_arguments(parser, multiple=True)
    cli_main(parser, run)


if __name__ == "__main__":
    main()
