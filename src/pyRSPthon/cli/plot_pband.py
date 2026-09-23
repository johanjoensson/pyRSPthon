"""
Plot cluster-projected RSPt spectral functions (pband-<cluster>.data).

Column 0 is the total spectral weight; the orbital projections follow the
spin/orbital moment columns (per shell a spin-down block then its spin-up
block, or one j block for jj bases).
"""

from argparse import ArgumentParser
import sys

from pyRSPthon.cli._common import (
    add_orbital_arguments,
    add_plot_arguments,
    cli_main,
    finish_plots,
)
from pyRSPthon.cli.plot_band import add_band_arguments, load


def run(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pyRSPthon.orbitals import resolve_layout, select_orbitals, spin_sum
    from pyRSPthon.plot import bands as bp

    model = load(args.directory, args, cluster=args.cluster)
    proj = model.projections
    layout = resolve_layout(
        proj.shells, proj.cfflag, proj.fullrel, proj.weights.shape[-1], irrep=proj.irrep
    )
    if args.spin_sum:
        weights, labels = spin_sum(proj.weights, layout)
    else:
        weights, labels = proj.weights, list(layout.labels)
    weights, sel_labels = select_orbitals(weights, labels, args.orbitals, args.orbital_labels)

    cmap = args.cmap or bp.SPECTRAL_CMAP

    # Total spectral weight
    fig, ax = plt.subplots()
    im = bp.plot_spectral(ax, model, log=args.log, cmap=cmap)
    bp.decorate_axes(ax, model, emin=args.emin, emax=args.emax)
    ax.set_title(f"{args.cluster}: total")
    fig.colorbar(im, ax=ax, pad=0.02)

    if args.composite:
        if len(sel_labels) > len(bp.COMPOSITE):
            print(
                f"warning: {len(sel_labels)} orbitals share {len(bp.COMPOSITE)} "
                "composite colours, so some are indistinguishable; reduce them "
                "with --spin-sum or --orbitals, or drop --composite for one "
                "panel per orbital",
                file=sys.stderr,
            )
        fig, ax = plt.subplots()
        bp.plot_composite(ax, model, weights, gamma=args.gamma)
        bp.decorate_axes(ax, model, emin=args.emin, emax=args.emax)
        ax.set_title(f"{args.cluster}: orbital character")
        patches = [
            mpatches.Patch(color=bp.COMPOSITE[i % len(bp.COMPOSITE)], label=lab)
            for i, lab in enumerate(sel_labels)
        ]
        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize="small",
            title="Orbital character",
        )
    else:
        for i, lab in enumerate(sel_labels):
            fig, ax = plt.subplots()
            im = bp.plot_spectral(ax, model, data=weights[:, :, i], log=args.log, cmap=cmap)
            bp.decorate_axes(ax, model, emin=args.emin, emax=args.emax)
            ax.set_title(f"{args.cluster}: {lab}")
            fig.colorbar(im, ax=ax, pad=0.02)

    finish_plots(args)


def build_parser():
    parser = ArgumentParser(prog="plot_pband", description=__doc__)
    add_plot_arguments(parser, cluster=True)
    add_band_arguments(parser, source=False)
    add_orbital_arguments(parser)
    parser.add_argument(
        "--orb-start",
        default=None,
        type=int,
        help="Column of the first orbital projection (default: from band.gpi "
        "or the Band_header shells, else 1)",
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
    return parser


def main():
    cli_main(build_parser(), run)


if __name__ == "__main__":
    main()
