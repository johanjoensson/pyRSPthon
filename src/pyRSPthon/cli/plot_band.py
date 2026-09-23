"""
Plot RSPt band structures / spectral functions.

Sources are auto-detected in the working directory (override with --source):
- band.data (+ Band_header / band.gpi / green.inp metadata): DMFT spectral function
- bandfile_0..N (+ symline_*, evconv.inp): DFT bands from evconv
- fatbands.*: orbital-projected DFT bands
- eigenvalues: raw eigenvalue file (no evconv needed)
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys

from pyRSPthon.cli._common import add_plot_arguments, cli_main, finish_plots


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def apply_band_options(model, args):
    """
    Apply the energy reference/unit and tick options to a loaded band model,
    warning about options that do not apply to its kind.
    """
    from pyRSPthon.read import bands as B

    efermi = getattr(args, "efermi", None)
    eV = getattr(args, "eV", False)
    if model.kind == "lines":
        if efermi is not None:
            model = B.with_reference(model, efermi)
        if eV:
            model = B.to_unit(model, "eV")
        if args.num_k is not None or args.num_e is not None:
            warn("-nk/-ne only apply to spectral (band.data) sources; ignored")
    else:
        ignored = [
            flag
            for flag, given in (
                ("--efermi", efermi is not None),
                ("--eV", eV),
                ("--nbands", getattr(args, "nbands", None) is not None),
            )
            if given
        ]
        if ignored:
            warn(
                f"{', '.join(ignored)} only apply to eigenvalue sources; spectral "
                "data are already relative to E_F in RSPt's output unit"
            )
    if args.xticks is not None or args.labels is not None:
        model = B.with_labels(model, ticks=args.xticks, labels=args.labels)
    return model


def load(directory, args, cluster=None):
    from pyRSPthon.read.bands import read_bands

    model = read_bands(
        directory,
        source=getattr(args, "source", None),
        cluster=cluster,
        nk=args.num_k,
        ne=args.num_e,
        nbands=getattr(args, "nbands", None),
        orb_start=getattr(args, "orb_start", None),
    )
    return apply_band_options(model, args)


def plot_dos_panel(ax, directory, model):
    """
    Total DOS next to the bands, in the band model's energy unit.
    """
    from pyRSPthon.cli._common import apply_unit_conversion, resolve_energy_unit
    from pyRSPthon.read import extract_dos
    from pyRSPthon.units import energy_scale

    dos = extract_dos(prefix=directory)
    native_unit, _ = resolve_energy_unit(directory, False)
    if model.energy_unit in ("Ry", "eV"):
        factor = energy_scale(native_unit, model.energy_unit)
        dos = apply_unit_conversion(dos, factor, is_dos=True, data_type="dos")
    else:
        warn(f"bands are in {model.energy_unit!r}; the DOS panel is not converted")
    if not model.shifted:
        warn("the bands are not relative to E_F (pass --efermi) but the DOS is")
    ax.fill_betweenx(dos.w, dos.sum, alpha=0.3, color="tab:gray")
    ax.set_xlabel("DOS")
    ax.set_xlim(left=0)


def run(args):
    import matplotlib.pyplot as plt
    import numpy as np
    from pyRSPthon.plot import bands as bp, palette
    from pyRSPthon.read.bands import BandReadError

    directories = [args.directory] + (args.compare or [])
    models = [load(d, args) for d in directories]
    sources = {getattr(m, "source", m.kind) for m in models}
    if len(sources) > 1:
        # the sources differ in kind or measure the k-path on different scales
        raise BandReadError(
            "--compare needs the same kind of data from the same source "
            "everywhere; got "
            + ", ".join(f"{d}: {getattr(m, 'source', m.kind)}" for d, m in zip(directories, models))
        )
    model = models[0]
    compare = len(models) > 1

    npanels = len(models) if model.kind == "spectral" else 1
    ncols = npanels + (1 if args.dos_panel else 0)
    width_ratios = [3] * npanels + ([1] if args.dos_panel else [])
    fig, axes = plt.subplots(
        ncols=ncols,
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = axes[0]

    if model.kind == "spectral":
        # one colour scale for all panels, so they can be compared
        top = max(float(np.max(m.total)) for m in models)
        if args.log:
            vmin, vmax = np.log(top * 1e-6), np.log(top)
        else:
            vmin, vmax = min(float(np.min(m.total)) for m in models), top
        for ax, m, d in zip(axes, models, directories):
            im = bp.plot_spectral(
                ax, m, log=args.log, vmin=vmin, vmax=vmax, cmap=args.cmap or bp.SPECTRAL_CMAP
            )
            bp.decorate_axes(ax, m, emin=args.emin, emax=args.emax)
            if compare:
                ax.set_title(d)
        fig.colorbar(im, ax=list(axes[:npanels]), pad=0.02)
    else:
        ax = axes[0]
        if not compare and model.projections is not None:
            bp.plot_fatbands(ax, model)
        else:
            if any(m.projections is not None for m in models):
                warn("--compare draws plain bands; fatband weights are not shown")
            for i, (m, d) in enumerate(zip(models, directories)):
                bp.plot_lines(
                    ax,
                    m,
                    color=palette.color(i),
                    linestyle=palette.linestyle(i),
                    label=d if compare else None,
                )
            if compare:
                ax.legend(loc="upper right", fontsize="small")
        bp.decorate_axes(ax, model, emin=args.emin, emax=args.emax)

    if args.dos_panel:
        plot_dos_panel(axes[-1], args.directory, model)

    finish_plots(args)


def add_band_arguments(parser, source=True):
    band_group = parser.add_argument_group("Band Data Options")
    if source:
        band_group.add_argument(
            "--source",
            choices=["spectral", "bandfiles", "fatbands", "eigenvalues"],
            default=None,
            help="Data source (default: auto-detect)",
        )
    band_group.add_argument(
        "-nk",
        "--num_k",
        default=None,
        type=int,
        metavar="INT",
        help="k-points in the path (spectral; usually auto-detected)",
    )
    band_group.add_argument(
        "-ne",
        "--num_e",
        default=None,
        type=int,
        metavar="INT",
        help="energy mesh points (spectral; usually auto-detected)",
    )

    fmt_group = parser.add_argument_group("Band Formatting Options")
    fmt_group.add_argument("--emin", default=None, type=float, metavar="FLOAT", help="Lower plot bound")
    fmt_group.add_argument("--emax", default=None, type=float, metavar="FLOAT", help="Upper plot bound")
    fmt_group.add_argument(
        "-x",
        "--xticks",
        default=None,
        type=float,
        nargs="+",
        metavar="FLOAT",
        help="Symmetry-point positions along the path (k index for spectral "
        "data, k-distance otherwise; usually auto-detected)",
    )
    fmt_group.add_argument(
        "-l",
        "--labels",
        default=None,
        type=str,
        nargs="+",
        metavar="STR",
        help="Symmetry-point labels, one per tick including the path ends",
    )
    fmt_group.add_argument("--log", action="store_true", help="Logarithmic intensity")
    fmt_group.add_argument(
        "--cmap",
        default=None,
        metavar="NAME",
        help="Matplotlib colormap for intensity plots (default: YlGnBu)",
    )


def build_parser():
    parser = ArgumentParser(
        prog="plot_band",
        description=__doc__ + "\n\n"
                    "Physics Conventions:\n"
                    "- For pure DFT (bandfiles/fatbands/eigenvalues), plots sharp eigenvalues (ε_nk).\n"
                    "- For DFT+DMFT (band.data), plots the continuous interacting spectral function A(k,ω).\n"
                    "- Spectral data are relative to E_F. Raw eigenvalues and fatbands are absolute "
                    "(in Ry) unless --efermi is given.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
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
        default=None,
        help="Fermi energy subtracted from eigenvalue sources, in the data's "
        "own unit (Ry for eigenvalues/fatbands), before any --eV conversion",
    )
    parser.add_argument(
        "--eV", action="store_true", help="Convert eigenvalue sources Ry -> eV"
    )
    parser.add_argument(
        "--nbands",
        type=int,
        default=None,
        help="Expected bands per k-point in a raw eigenvalues file (checked; "
        "read from the file)",
    )
    return parser


def main():
    cli_main(build_parser(), run)


if __name__ == "__main__":
    main()
