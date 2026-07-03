"""
Plot cluster-projected RSPt spectral functions (pband-<cluster>.data).

Column 0 is the total spectral weight; the remaining columns are the
orbital projections (two spin blocks when spin polarized: first half down,
second half up).
"""

from argparse import ArgumentParser
import sys

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots
from pyRSPthon.cli.plot_band import add_band_arguments


def parse_orbital_selection(spec, norb):
    if spec is None:
        return list(range(norb))
    picked = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-")
            picked.extend(range(int(lo), int(hi) + 1))
        else:
            picked.append(int(part))
    bad = [i for i in picked if i < 0 or i >= norb]
    if bad:
        raise SystemExit(f"Orbital indices {bad} out of range (0..{norb - 1})")
    return picked


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
    n_data = bs.spectral.shape[2]
    norb = n_data - args.orb_start
    if args.spin_sum and norb % 2 == 0:
        # Sum the two spin blocks (first half down, second half up)
        half = norb // 2
        summed = (
            bs.spectral[:, :, args.orb_start : args.orb_start + half]
            + bs.spectral[:, :, args.orb_start + half : args.orb_start + norb]
        )
        bs.spectral = np.concatenate([bs.spectral[:, :, :1], summed], axis=2)
        orb_columns = list(range(1, 1 + half))
        default_labels = bp.orbital_labels(half)
    else:
        orb_columns = list(range(args.orb_start, n_data))
        default_labels = bp.orbital_labels(norb, spin_split=norb % 2 == 0)
    labels = args.orbital_labels or default_labels
    selection = parse_orbital_selection(args.orbitals, len(orb_columns))
    columns = [orb_columns[i] for i in selection]
    sel_labels = [labels[i] if i < len(labels) else f"orb {i}" for i in selection]

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
        default=1,
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
