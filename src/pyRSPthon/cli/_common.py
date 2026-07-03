"""
Shared command line options and figure handling for the plot tools.
"""

import matplotlib


def add_plot_arguments(parser, cluster=False):
    """
    Add the options shared by all plot CLIs: directory, output file and
    basic figure styling.
    """
    if cluster:
        parser.add_argument("cluster", type=str, help="Cluster to plot.")
    parser.add_argument(
        "-d",
        "--directory",
        default=".",
        type=str,
        help="Look for files in this directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Save the figure(s) to this file instead of showing them "
        "(multiple figures get numbered suffixes).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument("--dpi", type=float, default=None, help="Figure DPI.")
    parser.add_argument(
        "--font-size", type=float, default=None, help="Base font size."
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
