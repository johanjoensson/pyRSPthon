"""
Plot cluster-projected RSPt spectral functions (pband-<cluster>.data).

Column 0 is the total spectral weight; the remaining columns are the
orbital projections (two spin blocks when spin polarized: first half down,
second half up).
"""

from argparse import ArgumentParser
import sys

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots, parse_orbital_selection
from pyRSPthon.cli.plot_band import add_band_arguments


def run(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from pyRSPthon.read import bands as B
    from pyRSPthon.cli import _bandplot as bp

    bs = B.read_spectral_bands(
        prefix=args.directory,
        cluster=args.cluster,
        nk=args.num_k,
        ne=args.num_e,
        ticks=args.xticks,
        tick_labels=args.labels,
    )
    
    shells = None
    cfflag = False
    fullrel = False
    
    # Try to determine if the calculation is fully relativistic
    try:
        from pyRSPthon.read.data import read_data
        prefix = args.directory
        if prefix != "" and prefix[-1] != "/":
            prefix = prefix + "/"
        rspt_data = read_data(f"{prefix}data")
        if rspt_data and rspt_data.fullrel is not None:
            fullrel = rspt_data.fullrel
    except Exception:
        pass
    
    # Check if the Band_header carries the correlated shells directly
    try:
        from pyRSPthon.read.band import peek_band_header
        prefix = args.directory
        if prefix != "" and prefix[-1] != "/":
            prefix = prefix + "/"
        data_file = f"{prefix}pband-{args.cluster}.data"
        header_meta = peek_band_header(data_file)
        if header_meta and header_meta.get("shells"):
            shells = header_meta["shells"]
            cfflag = bool(header_meta.get("cfflag", False))
    except Exception:
        pass

    if shells is None:
        shells, cfflag = bp.find_cluster_shells(args.cluster, args.directory)

    jj_bases = sorted(
        {sh["basis"] for sh in shells or []} & {9, 10, 11}
    )
    if args.spin_sum and jj_bases:
        raise SystemExit(f"--spin-sum is invalid for a JJ basis (basis={jj_bases}) because it would mix distinct j-manifolds.")
        
    if args.spin_sum and cfflag and fullrel:
        raise SystemExit("--spin-sum is invalid for fully relativistic crystal field (Cf) states. The states are mixed spinors, not spin-down/spin-up pairs.")

    n_data = bs.spectral.shape[2]
    
    orb_start = args.orb_start
    if orb_start is None:
        orb_start = bs.orb_start if bs.orb_start is not None else 1
        
    orb_end = bs.orb_end if bs.orb_end is not None else n_data
    norb = orb_end - orb_start
    
    # In fully relativistic Cf mode, states are 2N independent spinors. Do not split them into down/up blocks.
    is_spin_split = (norb % 2 == 0)
    if cfflag and fullrel:
        is_spin_split = False

    if args.spin_sum and is_spin_split:
        # Sum the two spin blocks (first half down, second half up)
        half = norb // 2
        summed = (
            bs.spectral[:, :, orb_start : orb_start + half]
            + bs.spectral[:, :, orb_start + half : orb_start + norb]
        )
        bs.spectral = np.concatenate([bs.spectral[:, :, :1], summed], axis=2)
        orb_columns = list(range(1, 1 + half))
        default_labels = bp.shell_labels(shells, half, spin_split=False, cfflag=cfflag)
    else:
        orb_columns = list(range(orb_start, orb_end))
        default_labels = bp.shell_labels(shells, norb, spin_split=is_spin_split, cfflag=cfflag)
    labels = default_labels
    selection = parse_orbital_selection(args.orbitals, len(orb_columns))
    
    columns = []
    sel_labels = []
    for i, group in enumerate(selection):
        if len(group) == 1:
            idx = group[0]
            columns.append(orb_columns[idx])
            sel_labels.append(labels[idx] if idx < len(labels) else f"orb {idx}")
        else:
            group_cols = [orb_columns[idx] for idx in group]
            summed_col = np.sum(bs.spectral[:, :, group_cols], axis=2, keepdims=True)
            bs.spectral = np.concatenate([bs.spectral, summed_col], axis=2)
            columns.append(bs.spectral.shape[2] - 1)
            group_labels = [labels[idx] if idx < len(labels) else f"orb {idx}" for idx in group]
            sel_labels.append(" + ".join(group_labels))
            
    if args.orbital_labels:
        for i, lab in enumerate(args.orbital_labels):
            if i < len(sel_labels):
                sel_labels[i] = lab

    # Total spectral weight
    fig, ax = plt.subplots()
    im = bp.plot_spectral(ax, bs, column=0, log=args.log)
    bp.decorate_axes(ax, bs, emin=args.emin, emax=args.emax)
    ax.set_title(f"{args.cluster}: total")
    fig.colorbar(im, ax=ax, pad=0.02)

    if args.composite:
        fig, ax = plt.subplots()
        bp.plot_composite(ax, bs, columns, gamma=args.gamma)
        bp.decorate_axes(ax, bs, emin=args.emin, emax=args.emax)
        ax.set_title(f"{args.cluster}: orbital character")
        patches = [
            mpatches.Patch(color=bp.ORBITAL_COLORS[i % len(bp.ORBITAL_COLORS)], label=lab)
            for i, lab in enumerate(sel_labels)
        ]
        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize="small",
            title="Orbital character",
        )
    else:
        for col, lab in zip(columns, sel_labels):
            fig, ax = plt.subplots()
            im = bp.plot_spectral(ax, bs, column=col, log=args.log)
            bp.decorate_axes(ax, bs, emin=args.emin, emax=args.emax)
            ax.set_title(f"{args.cluster}: {lab}")
            fig.colorbar(im, ax=ax, pad=0.02)

    finish_plots(args)


def main():
    parser = ArgumentParser(description=__doc__)
    add_plot_arguments(parser, cluster=True)
    add_band_arguments(parser)
    parser.add_argument(
        "--orb-start",
        default=None,
        type=int,
        help="Column of the first orbital projection (default 1)",
    )
    parser.add_argument(
        "--orbitals",
        default=None,
        type=str,
        help='Orbitals to plot, e.g. "0,2,4" or "0-4" (default: all)',
    )
    parser.add_argument(
        "--orbital-labels",
        default=None,
        type=str,
        nargs="+",
        help="Override the automatic orbital labels",
    )
    parser.add_argument(
        "--spin-sum",
        action="store_true",
        help="Sum the two spin blocks of the orbital columns",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="One RGB-composite image of the selected orbitals instead of "
        "one panel per orbital",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction for the composite weights (<1 boosts weak bands)",
    )
    args = parser.parse_args()
    apply_plot_style(args)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError) as err:
        print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
