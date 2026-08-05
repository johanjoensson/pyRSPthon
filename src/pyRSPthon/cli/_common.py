"""
Shared command line options and figure handling for the plot tools.
"""

import matplotlib


def add_plot_arguments(parser, cluster=False):
    """
    Add the options shared by all plot CLIs: directory, output file and
    basic figure styling.
    """
    io_group = parser.add_argument_group("I/O Options")
    if cluster:
        io_group.add_argument("cluster", type=str, help="Cluster to plot.")
    io_group.add_argument(
        "-d",
        "--directory",
        default=".",
        type=str,
        metavar="DIR",
        help="Look for files in this directory.",
    )
    io_group.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        metavar="FILE",
        help="Save the figure(s) to this file instead of showing them "
        "(multiple figures get numbered suffixes).",
    )
    
    fmt_group = parser.add_argument_group("Plot Formatting Options")
    fmt_group.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    fmt_group.add_argument("--dpi", type=float, default=None, metavar="INT", help="Figure DPI.")
    fmt_group.add_argument(
        "--font-size", type=float, default=None, metavar="PT", help="Base font size."
    )


def apply_plot_style(args):
    """
    Apply the shared style options. Must run before pyplot figures are made.
    """
    if args.output is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.figsize is not None:
        plt.rcParams["figure.figsize"] = args.figsize
    if args.dpi is not None:
        plt.rcParams["figure.dpi"] = args.dpi
        plt.rcParams["savefig.dpi"] = args.dpi
    if args.font_size is not None:
        plt.rcParams["font.size"] = args.font_size


def finish_plots(args):
    """
    Show the figures, or save them to args.output. With several open
    figures the output name gets a numbered suffix per figure.
    """
    import matplotlib.pyplot as plt

    if args.output is None:
        plt.show()
        return
    fignums = plt.get_fignums()
    if len(fignums) == 1:
        plt.figure(fignums[0])
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Wrote {args.output}")
        return
    stem, dot, ext = args.output.rpartition(".")
    if not dot:
        stem, ext = args.output, "png"
    for i, num in enumerate(fignums, start=1):
        fname = f"{stem}-{i}.{ext}"
        plt.figure(num)
        plt.savefig(fname, bbox_inches="tight")
        print(f"Wrote {fname}")

def parse_orbital_selection(spec, norb):
    """
    Parse an orbital selection like "0,2,4", "0-4" or "0+1,3-5" into a list
    of index groups. Comma-separated entries plot separately (a range gives
    one entry per index); '+'-joined indices/ranges form one summed group.
    With spec None every orbital is its own group.
    """
    if spec is None:
        return [[i] for i in range(norb)]

    def expand(token):
        token = token.strip()
        if "-" in token and not token.startswith("-"):
            lo, hi = (int(s) for s in token.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty range {token!r}")
            return list(range(lo, hi + 1))
        return [int(token)]

    picked = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "+" in part:
                group = []
                for sub in part.split("+"):
                    if sub.strip():
                        group.extend(expand(sub))
                if group:
                    picked.append(group)
            else:
                picked.extend([i] for i in expand(part))
        except ValueError:
            raise SystemExit(
                f"Malformed orbital selection {part!r} "
                '(expected e.g. "0,2,4", "0-4" or "0+1+2")'
            )
    bad = sorted({i for group in picked for i in group if i < 0 or i >= norb})
    if bad:
        raise SystemExit(f"Orbital indices {bad} out of range (0..{norb - 1})")
    return picked
