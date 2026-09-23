"""
Shared command line options and figure handling for the plot tools.
"""

import sys

import matplotlib

from pyRSPthon.read import extract_dos, extract_pdos, green
from pyRSPthon.read import read as read_dat
from pyRSPthon.orbitals import OrbitalSelectionError
from pyRSPthon.units import RY_TO_EV


def add_plot_arguments(parser, cluster=False, multiple_clusters=False):
    """
    Add the options shared by all plot CLIs: directory, output file and
    basic figure styling.
    """
    io_group = parser.add_argument_group("I/O Options")
    if cluster:
        if multiple_clusters:
            io_group.add_argument("cluster", type=str, nargs="+", help="Cluster(s) to plot.")
        else:
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

def add_orbital_arguments(parser, multiple=False):
    """
    The orbital selection options shared by the projected plot CLIs. With
    multiple, --orbitals takes one selection per cluster.
    """
    group = parser.add_argument_group("Orbital Options")
    group.add_argument(
        "--orbitals",
        default=None,
        type=str,
        nargs="+" if multiple else None,
        help='Orbitals to plot, e.g. "0,2,4", "0-4" or "0+1+2" (summed); '
        "default: all" + (" (one selection, or one per cluster)" if multiple else ""),
    )
    group.add_argument(
        "--orbital-labels",
        default=None,
        type=str,
        nargs="+",
        help="Override the automatic orbital labels",
    )


def cli_main(parser, run):
    """
    Parse the arguments, apply the plot style and run. Errors in the input
    data (unreadable or inconsistent files, bad selections) are reported as
    one line on stderr with exit status 1 instead of a traceback; anything
    else is a bug and keeps its traceback.
    """
    from pyRSPthon.read.bands import BandReadError

    args = parser.parse_args()
    apply_plot_style(args)
    try:
        run(args)
    except (BandReadError, OrbitalSelectionError, OSError) as err:
        print(f"{parser.prog}: error: {err}", file=sys.stderr)
        sys.exit(1)



def resolve_energy_unit(directory, args_eV):
    native_eV = False
    try:
        green_dat = green.get_green(prefix=directory)
        native_eV = "eV" in green_dat.spectrum
    except (FileNotFoundError, TypeError):
        pass
        
    # If user explicitly asks for eV, but it's natively Ry, we must convert.
    # If user explicitly asks for Ry (default is args_eV=False) but it's natively eV, we must convert.
    # Wait, the argparse default is False. So if args_eV is False, does the user WANT Ry, or are they just accepting the default?
    # Usually, if args.eV is False, we just use native. If args.eV is True, we FORCE eV.
    target_eV = True if args_eV else native_eV
    
    conversion_factor = 1.0
    if target_eV and not native_eV:
        conversion_factor = RY_TO_EV
    elif not target_eV and native_eV:
        conversion_factor = 1.0 / RY_TO_EV
        
    e_unit = "eV" if target_eV else "Ry"
    return e_unit, conversion_factor

def apply_unit_conversion(dat_obj, factor, is_dos=False, data_type=None):
    """
    Return a copy of a DOS/dat namedtuple with its energies scaled by factor
    and its data scaled accordingly: sig, hyb and pt carry units of energy,
    everything else (dos, pdos, Green's functions) units of 1/energy.
    """
    if factor == 1.0:
        return dat_obj
    y_factor = factor if data_type in ["sig", "hyb", "pt"] else 1.0 / factor
    scaled = ["sum", "up", "down", "orbitals"] + (["s", "l", "j"] if is_dos else [])
    changes = {"w": dat_obj.w * factor}
    for attr in scaled:
        val = getattr(dat_obj, attr, None)
        if val is not None:
            changes[attr] = val * y_factor
    return dat_obj._replace(**changes)

def prepare_plot_data(clusters, directory, orbitals_arg, data_type="pdos"):
    valid_data = []
    valid_clusters = []
    valid_orbitals = []
    
    orb_list = orbitals_arg if orbitals_arg is not None else [None] * len(clusters)
    if len(orb_list) == 1:
        orb_list = orb_list * len(clusters)
    elif len(orb_list) != len(clusters):
        raise OrbitalSelectionError(
            f"--orbitals takes one selection or one per cluster ({len(clusters)}), "
            f"got {len(orb_list)}"
        )

    for cluster, orb_sel in zip(clusters, orb_list):
        try:
            if data_type == "pdos":
                d = extract_pdos(cluster, prefix=directory)
            else:
                d = read_dat(f"{directory}/{data_type}-{cluster}.dat")
            valid_data.append(d)
            valid_clusters.append(cluster)
            valid_orbitals.append(orb_sel)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not read {data_type} for {cluster}: {e}", file=sys.stderr)
            
    return valid_data, valid_clusters, valid_orbitals

def extract_true_total_dos(directory, conversion_factor):
    try:
        dos_total = extract_dos(prefix=directory)
        return apply_unit_conversion(dos_total, conversion_factor, is_dos=True, data_type="dos")
    except (FileNotFoundError, ValueError, RuntimeError):
        return None

