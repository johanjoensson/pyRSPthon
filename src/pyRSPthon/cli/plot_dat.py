from pyRSPthon.read import read, green
from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from itertools import product

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



def plot_dat(clusters, dataset, dat_list, e_unit, args, valid_orbitals):
    import numpy as np
    import sys
    from pyRSPthon.cli._common import parse_orbital_selection, OrbitalSelectionError

    multi = len(clusters) > 1

    _ = plt.figure()
    plt.title(f"{dataset}")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(f"Re{{{dataset}}}")
    for i, (dat, cluster) in enumerate(zip(dat_list, clusters)):
        color = f"C{i}"
        if dat.up is not None:
            plt.plot(dat.w, dat.up.real, color=color, linestyle="-", label=f"{cluster}: " + r"$\uparrow$")
            plt.plot(dat.w, dat.down.real, color=color, linestyle="--", label=f"{cluster}: " + r"$\downarrow$")
        else:
            if multi:
                plt.plot(dat.w, dat.sum.real, color=color, linestyle="-", label=f"{cluster}")
            else:
                plt.plot(dat.w, dat.sum.real, color=color, linestyle="-", label="Total")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    _ = plt.figure()
    plt.title(f"{dataset}")
    plt.xlabel(rf"E - E$_F$ ({e_unit})")
    plt.ylabel(f"Im{{{dataset}}}")
    for i, (dat, cluster) in enumerate(zip(dat_list, clusters)):
        color = f"C{i}"
        if dat.up is not None:
            plt.plot(dat.w, dat.up.imag, color=color, linestyle="-", label=f"{cluster}: " + r"$\uparrow$")
            plt.plot(dat.w, dat.down.imag, color=color, linestyle="--", label=f"{cluster}: " + r"$\downarrow$")
        else:
            if multi:
                plt.plot(dat.w, dat.sum.imag, color=color, linestyle="-", label=f"{cluster}")
            else:
                plt.plot(dat.w, dat.sum.imag, color=color, linestyle="-", label="Total")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    orbitals_dat_2d = []
    orbitals_dat_3d = []
    
    for i, (d, c, o) in enumerate(zip(dat_list, clusters, valid_orbitals)):
        if d.orbitals is not None:
            if d.orbitals.ndim == 2:
                orbitals_dat_2d.append((d, c, o, i))
            else:
                orbitals_dat_3d.append((d, c, o, i))

    if orbitals_dat_2d:
        for part_name, part_func in (("Re", np.real), ("Im", np.imag)):
            _ = plt.figure()
            for dat, cluster, orb_sel, cluster_idx in orbitals_dat_2d:
                norb = dat.orbitals.shape[1]
                try:
                    selection = parse_orbital_selection(orb_sel, norb)
                except OrbitalSelectionError as e:
                    print(f"Warning: --orbitals selection invalid for {cluster} (norb={norb}): {e}. Skipping.", file=sys.stderr)
                    continue
                
                sel_data = []
                sel_labels = []
                for group in selection:
                    sel_data.append(np.sum(dat.orbitals[:, group], axis=1))
                    sel_labels.append("+".join(str(idx) for idx in group))
                if args.orbital_labels:
                    for i_lab, lab in enumerate(args.orbital_labels):
                        if i_lab < len(sel_labels):
                            sel_labels[i_lab] = lab
                            
                color = f"C{cluster_idx}"
                for j, (data_col, lab) in enumerate(zip(sel_data, sel_labels)):
                    plt.plot(
                        dat.w, part_func(data_col), color=color, linestyle=linestyles[j % len(linestyles)], label=f"{cluster}: {lab}"
                    )
            plt.title(f"Orbital projected {dataset} (2D Diagonal)")
            plt.xlabel(rf"E - E$_F$ ({e_unit})")
            plt.ylabel(f"{part_name}{{{dataset}}}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if orbitals_dat_3d:
        if getattr(args, 'orbitals', None) is None:
            # Group by blocks to avoid mashing mismatched physics
            from collections import defaultdict
            grouped_by_blocks = defaultdict(list)
            for dat, cluster, orb_sel, cluster_idx in orbitals_dat_3d:
                block_repr = tuple(tuple(b) for b in dat.blocks)
                grouped_by_blocks[block_repr].append((dat, cluster, cluster_idx))
                
            for block_repr, group_dats in grouped_by_blocks.items():
                max_blocks = len(block_repr)
                for block_idx in range(max_blocks):
                    n_max = len(block_repr[block_idx])
                    if n_max == 0: continue
                    
                    fig_re, ax_re = plt.subplots(nrows=n_max, ncols=n_max, squeeze=False, sharex="all", sharey="all")
                    fig_im, ax_im = plt.subplots(nrows=n_max, ncols=n_max, squeeze=False, sharex="all", sharey="all")

                    for dat, cluster, cluster_idx in group_dats:
                        block = dat.blocks[block_idx]
                        color = f"C{cluster_idx}"
                        
                        for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                            ax_re[i, j].plot(dat.w, dat.orbitals[:, orb_i, orb_j].real, color=color, label=f"{cluster}" if i==0 and j==0 else "")
                            ax_im[i, j].plot(dat.w, dat.orbitals[:, orb_i, orb_j].imag, color=color, label=f"{cluster}" if i==0 and j==0 else "")
                            if cluster_idx == group_dats[0][2] or not ax_re[i,j].get_title():
                                ax_re[i, j].set_title(f"{orb_i}_{orb_j}")
                                ax_im[i, j].set_title(f"{orb_i}_{orb_j}")

                    fig_re.suptitle(f"Orbital projected {dataset} (Re) - Block {block_idx}")
                    fig_re.supxlabel(rf"E - E$_F$ ({e_unit})")
                    fig_re.supylabel(rf"Re{{{dataset}}}")
                    if len(group_dats) > 1: fig_re.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    fig_im.suptitle(f"Orbital projected {dataset} (Im) - Block {block_idx}")
                    fig_im.supxlabel(rf"E - E$_F$ ({e_unit})")
                    fig_im.supylabel(rf"Im{{{dataset}}}")
                    if len(group_dats) > 1: fig_im.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Using --orbitals selection
            # Just extract the physical TRACE for grouped elements to prevent physical nonsense
            parsed_selections = []
            for dat, cluster, orb_sel, cluster_idx in orbitals_dat_3d:
                norb = dat.orbitals.shape[1]
                try:
                    selection = parse_orbital_selection(orb_sel, norb)
                    parsed_selections.append((dat, cluster, cluster_idx, selection))
                except OrbitalSelectionError as e:
                    print(f"Warning: --orbitals selection invalid for {cluster} (norb={norb}): {e}. Skipping.", file=sys.stderr)
            
            if not parsed_selections: return
            
            for part_name, part_func in (("Re", np.real), ("Im", np.imag)):
                _ = plt.figure()
                for dat, cluster, cluster_idx, selection in parsed_selections:
                    sel_data = []
                    sel_labels = []
                    for group in selection:
                        # Extract the physical trace (sum of diagonals) for this group
                        trace_data = np.sum([dat.orbitals[:, idx, idx] for idx in group], axis=0)
                        sel_data.append(trace_data)
                        sel_labels.append("+".join(str(idx) for idx in group))
                        
                    if getattr(args, 'orbital_labels', None):
                        for i_lab, lab in enumerate(args.orbital_labels):
                            if i_lab < len(sel_labels):
                                sel_labels[i_lab] = lab

                    color = f"C{cluster_idx}"
                    for j, (data_col, lab) in enumerate(zip(sel_data, sel_labels)):
                        plt.plot(
                            dat.w, part_func(data_col), color=color, linestyle=linestyles[j % len(linestyles)], label=f"{cluster}: {lab}"
                        )
                plt.title(f"Orbital projected {dataset} (Trace)")
                plt.xlabel(rf"E - E$_F$ ({e_unit})")
                plt.ylabel(f"{part_name}{{{dataset}}}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def run(args):
    from pyRSPthon.cli._common import finish_plots, resolve_energy_unit, apply_unit_conversion, prepare_plot_data
    import sys
    
    e_unit, conversion_factor = resolve_energy_unit(args.directory, getattr(args, 'eV', False))

    if args.data.lower() == "dos":
        # dos doesn't have clusters, but we use the helper to load it
        from pyRSPthon.read import read as read_dat
        from pyRSPthon.cli.plot_dat import plot_dos
        dat = read_dat(f"{args.directory}/dos.dat")
        apply_unit_conversion(dat, conversion_factor, is_dos=True, data_type="dos")
        plot_dos(dat, e_unit)
    elif args.data.lower() == "pdos":
        from pyRSPthon.cli import plot_pdos
        plot_pdos.run(args)
        return
    else:
        dat_list, valid_clusters, valid_orbitals = prepare_plot_data(args.cluster, args.directory, getattr(args, 'orbitals', None), data_type=args.data)
        for d in dat_list:
            apply_unit_conversion(d, conversion_factor, is_dos=False, data_type=args.data)
            
        if dat_list:
            plot_dat(valid_clusters, args.data, dat_list, e_unit, args, valid_orbitals)
            
    finish_plots(args)

def main():
    parser = ArgumentParser(
        description="Plot RSPt data files (dos, pdos, real-/imag-<dataset>)."
    )
    add_plot_arguments(parser, cluster=True, multiple_clusters=True)
    parser.add_argument("data", type=str, help="Dataset to plot (dos, pdos, hyb, ...)")
    parser.add_argument("--eV", action="store_true", help="Unit of energy is eV.")
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
