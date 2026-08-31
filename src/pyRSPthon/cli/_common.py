class OrbitalSelectionError(Exception):
    pass

"""
Shared command line options and figure handling for the plot tools.
"""

import matplotlib


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
            raise OrbitalSelectionError(
                f"Malformed orbital selection {part!r} "
                '(expected e.g. "0,2,4", "0-4" or "0+1+2")'
            )
    bad = sorted({i for group in picked for i in group if i < 0 or i >= norb})
    if bad:
        raise OrbitalSelectionError(f"Orbital indices {bad} out of range (0..{norb - 1})")
    return picked

import sys
import numpy as np
from pyRSPthon.read import extract_pdos, extract_dos
from pyRSPthon.read import read as read_dat
from pyRSPthon.read import green

RY_TO_EV = 13.605693122994

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
    if factor == 1.0:
        return
    # Energy scales by factor
    dat_obj.w = dat_obj.w * factor
    
    # Identify y-axis scaling
    # sig (self-energy), hyb (hybridization), and pt (perturbation theory) have units of Energy
    # dos, pdos, grn (Green's function) have units of 1/Energy
    if data_type in ["sig", "hyb", "pt"]:
        y_factor = factor
    else:
        y_factor = 1.0 / factor

    if dat_obj.sum is not None: dat_obj.sum = dat_obj.sum * y_factor
    if dat_obj.up is not None: dat_obj.up = dat_obj.up * y_factor
    if getattr(dat_obj, 'down', None) is not None: dat_obj.down = dat_obj.down * y_factor
    if getattr(dat_obj, 'orbitals', None) is not None: dat_obj.orbitals = dat_obj.orbitals * y_factor
    
    if is_dos:
        for attr in ['s', 'l', 'j']:
            val = getattr(dat_obj, attr, None)
            if val is not None:
                setattr(dat_obj, attr, val * y_factor)

def prepare_plot_data(clusters, directory, orbitals_arg, data_type="pdos"):
    valid_data = []
    valid_clusters = []
    valid_orbitals = []
    
    orb_list = orbitals_arg if orbitals_arg is not None else [None] * len(clusters)
    if len(orb_list) == 1:
        orb_list = orb_list * len(clusters)
    elif len(orb_list) != len(clusters):
        print("Error: --orbitals must have 1 argument or one for each cluster.", file=sys.stderr)
        sys.exit(1)

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
        apply_unit_conversion(dos_total, conversion_factor, is_dos=True, data_type="dos")
        return dos_total
    except (FileNotFoundError, ValueError, RuntimeError):
        return None

