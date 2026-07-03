"""
Plot RSPt band structures / spectral functions.

Sources are auto-detected in the working directory (override with --source):
- band.data (+ band.gpi / green.inp metadata): DMFT spectral function
- bandfile_0..N (+ symline_*): DFT bands from evconv
- fatbands.*: orbital-projected DFT bands
- eigenvalues: raw eigenvalue file (no evconv needed)
"""

from argparse import ArgumentParser
import glob
import os
import sys

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots


def detect_source(directory):
    if os.path.exists(os.path.join(directory, "band.data")):
        return "spectral"
    if os.path.exists(os.path.join(directory, "bandfile_0")):
        return "bandfiles"
    if glob.glob(os.path.join(directory, "fatbands.*")):
        return "fatbands"
    if os.path.exists(os.path.join(directory, "eigenvalues")):
        return "eigenvalues"
    raise SystemExit(
        f"No band data found in {directory!r} (looked for band.data, "
        "bandfile_0, fatbands.*, eigenvalues)."
    )


def load(args, directory=None):
    from pyRSPthon.read import bands as B

    directory = directory if directory is not None else args.directory
    source = args.source or detect_source(directory)
    if source == "spectral":
        return B.read_spectral_bands(
            prefix=directory,
            nk=args.num_k,
            ne=args.num_e,
            ticks=args.xticks,
            tick_labels=args.labels,
        )
    if source == "bandfiles":
        return B.read_bandfiles(prefix=directory)
    if source == "fatbands":
        return B.read_fatbands(prefix=directory)
    if source == "eigenvalues":
        scale = B.Ry_to_eV if args.eV else 1.0
        return B.read_eigenvalues(
            fname=os.path.join(directory, "eigenvalues"),
            reference=args.efermi,
            scale=scale,
            nbands=args.nbands,
        )
    raise SystemExit(f"Unknown source {source!r}")


def run(args):
    import matplotlib.pyplot as plt
    from pyRSPthon.cli import _bandplot as bp

    bs = load(args)
    directories = [args.directory] + (args.compare or [])
    structures = [bs] + [load(args, d) for d in (args.compare or [])]

    npanels = len(structures) if bs.kind == "spectral" else 1
    ncols = npanels + (1 if args.dos_panel else 0)
    width_ratios = [3] * npanels + ([1] if args.dos_panel else [])
    fig, axes = plt.subplots(
        ncols=ncols,
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = axes[0]

    if bs.kind == "spectral":
        for ax, s, d in zip(axes, structures, directories):
            im = bp.plot_spectral(ax, s, column=0, log=args.log)
            bp.decorate_axes(ax, s, emin=args.emin, emax=args.emax)
            if len(structures) > 1:
                ax.set_title(d)
        fig.colorbar(im, ax=axes[npanels - 1], pad=0.02)
    else:
        ax = axes[0]
        if bs.weights:
            bp.plot_fatbands(ax, bs)
        else:
            for s, d, color in zip(
                structures, directories, ["#0072B2", "#D55E00", "#009E73"]
            ):
                bp.plot_lines(
                    ax, s, color=color, label=d if len(structures) > 1 else None
                )
            if len(structures) > 1:
                ax.legend(loc="upper right", fontsize="small")
        bp.decorate_axes(ax, bs, emin=args.emin, emax=args.emax)

    if args.dos_panel:
        from pyRSPthon.read import extract_dos

        dos = extract_dos(prefix=args.directory)
        dax = axes[-1]
        dax.fill_betweenx(dos.w, dos.sum, alpha=0.3, color="tab:gray")
        dax.set_xlabel("DOS")
        dax.set_xlim(left=0)

    finish_plots(args)


def add_band_arguments(parser):
    parser.add_argument(
        "--source",
        choices=["spectral", "bandfiles", "fatbands", "eigenvalues"],
        default=None,
        help="Data source (default: auto-detect)",
    )
    parser.add_argument(
        "-nk",
        "--num_k",
        default=None,
        type=int,
        help="k-points in the path (spectral; usually auto-detected)",
    )
    parser.add_argument(
        "-ne",
        "--num_e",
        default=None,
        type=int,
        help="energy mesh points (spectral; usually auto-detected)",
    )
    parser.add_argument("--emin", default=None, type=float, help="Lower plot bound")
    parser.add_argument("--emax", default=None, type=float, help="Upper plot bound")
    parser.add_argument(
        "-x",
        "--xticks",
        default=None,
        type=int,
        nargs="+",
        help="Symmetry-point indices (spectral; usually auto-detected)",
    )
    parser.add_argument(
        "-l",
        "--labels",
        default=None,
        type=str,
        nargs="+",
        help="Symmetry-point labels (usually auto-detected)",
    )
    parser.add_argument("--log", action="store_true", help="Logarithmic intensity")


def main():
    parser = ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    add_band_arguments(parser)
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        metavar="DIR",
        help="Also plot the band structure from these directories",
    )
    parser.add_argument(
        "--dos-panel", action="store_true", help="Attach a DOS side panel"
    )
    parser.add_argument(
        "--efermi",
        type=float,
        default=0.0,
        help="Reference energy subtracted from raw eigenvalues (Ry)",
    )
    parser.add_argument(
        "--eV", action="store_true", help="Scale raw eigenvalues Ry -> eV"
    )
    parser.add_argument(
        "--nbands",
        type=int,
        default=None,
        help="Bands per k-point in a raw eigenvalues file (usually inferred)",
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
