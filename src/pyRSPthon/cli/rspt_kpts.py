"""
Generate RSPt k-point files: meshes (spts) and band paths (spts.band).

Non-interactive replacement for the cub and kpath tools. Meshes are reduced
ONLY with the symmetry operations RSPt knows (parsed from symcof); see
pyRSPthon.kpts.grid for why this matters.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import sys

import numpy as np


def add_cell_arguments(parser):
    parser.add_argument(
        "--symt",
        type=str,
        default=None,
        help="Read the cell from this symt.inp (default: try sym/symt.inp, symt.inp)",
    )


def load_atoms(args):
    from pyRSPthon.ase import read_symt

    candidates = (
        [args.symt] if args.symt else ["sym/symt.inp", "symt.inp", "sym/symt.inp.bak"]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return read_symt(candidate)
    raise FileNotFoundError(
        "Could not find a symt.inp to read the structure from; pass --symt"
    )


def load_cell(args):
    return load_atoms(args).cell[:]


def grid_command(args):
    from pyRSPthon.kpts.grid import generate_grid, get_kpts, fold_to_bz
    from pyRSPthon.kpts.spts import write_spts

    if (args.min_distance is None) == (args.nk is None):
        raise SystemExit("grid: give exactly one of -n/--nk or --min-distance")
    if args.min_distance is not None:
        from pyRSPthon.kpts.grid import get_kpts_min_distance

        atoms = load_atoms(args)
        kpoints, weights = get_kpts_min_distance(
            atoms,
            args.min_distance,
            symcof_path=args.symcof,
            time_reversal=args.time_reversal,
        )
        write_spts(args.output, kpoints, weights)
        print(
            f"Wrote {args.output}: {len(kpoints)} irreducible k-points "
            f"(kpLib grid, symcof-reduced), total weight {int(np.sum(weights))}"
        )
        return
    cell = load_cell(args)
    map_matrix = None
    if args.map is not None:
        map_matrix = np.array(args.map, dtype=int).reshape(3, 3)
    if args.unreduced or not os.path.exists(args.symcof):
        if not args.unreduced:
            print(
                f"No {args.symcof} found: writing the FULL (unreduced) grid. "
                "This is always correct, just larger; run 'symt -all' first "
                "to enable reduction.",
                file=sys.stderr,
            )
        points, D, _ = generate_grid(
            args.nk, args.shift, args.shift_den, map_matrix
        )
        kpoints = fold_to_bz(points / D, cell)
        weights = np.ones(len(kpoints), dtype=int)
    else:
        kpoints, weights = get_kpts(
            cell,
            args.nk,
            shift_num=args.shift,
            shift_den=args.shift_den,
            map_matrix=map_matrix,
            symcof_path=args.symcof,
            time_reversal=args.time_reversal,
        )
    write_spts(
        args.output,
        kpoints,
        weights,
        grid=args.nk,
        shift_num=args.shift,
        shift_den=args.shift_den,
    )
    print(
        f"Wrote {args.output}: {len(kpoints)} irreducible k-points, "
        f"total weight {int(np.sum(weights))}"
    )


def path_command(args):
    from pyRSPthon.kpts.path import write_band_path, spectrum_block

    cell = load_cell(args)
    labels, nks = write_band_path(
        args.output,
        cell,
        args.path,
        args.points_per_segment,
        zero_weight_point=args.zero_weight_point,
    )
    total = sum(nk + 1 for nk in nks) + 1
    print(f"Wrote {args.output}: {total} k-points along {'-'.join(labels)}")
    print("green.inp spectrum block:")
    print(spectrum_block(labels, nks))


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    grid = sub.add_parser("grid", help="Generate a k-point mesh (spts)", formatter_class=ArgumentDefaultsHelpFormatter)
    add_cell_arguments(grid)
    grid.add_argument(
        "-n", "--nk", nargs=3, type=int, default=None, metavar=("N1", "N2", "N3"),
        help="Generate a standard Monkhorst-Pack grid with N1 x N2 x N3 k-points.",
    )
    grid.add_argument(
        "--min-distance",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Generate grid using kpLib ensuring this minimum periodic distance (Å) "
        "between k-points (scales inversely with real-space lattice). "
        "Warning: kpLib search can stall for highly skewed/large primitive cells; prefer -n if it hangs.",
    )
    grid.add_argument(
        "--shift",
        nargs=3,
        type=int,
        default=[0, 0, 0],
        metavar=("O1", "O2", "O3"),
        help="Shift numerators (offset = shift/shift-den per axis)",
    )
    grid.add_argument(
        "--shift-den",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        metavar=("D1", "D2", "D3"),
        help="Shift denominators",
    )
    grid.add_argument(
        "--map",
        nargs=9,
        type=int,
        default=None,
        metavar="INT",
        help="Supercell map matrix (9 ints, row-major; cub.inp M). Default: identity",
    )
    grid.add_argument(
        "--symcof",
        type=str,
        default="symcof",
        metavar="FILE",
        help="Symmetry file used for reduction to the Irreducible Brillouin Zone (IBZ). "
        "Crucial for SCF integrations to have correct weights. "
        "Do not use external symmetries not present in symcof.",
    )
    grid.add_argument(
        "--time-reversal",
        action="store_true",
        help="Also reduce with time reversal (-k). WARNING: Disable this if calculating magnetic structures "
        "where time-reversal symmetry is broken, otherwise you may artificially force zero magnetization.",
    )
    grid.add_argument(
        "--unreduced", action="store_true", help="Write the full grid, no reduction"
    )
    grid.add_argument("-o", "--output", type=str, default="spts")
    grid.set_defaults(func=grid_command)

    path = sub.add_parser("path", help="Generate a band path (spts.band)", formatter_class=ArgumentDefaultsHelpFormatter)
    add_cell_arguments(path)
    path.add_argument(
        "path",
        type=str,
        help='High-symmetry path, e.g. "GXWLG" or "G,X,W,L,G"',
    )
    path.add_argument(
        "-p",
        "--points-per-segment",
        type=int,
        default=20,
        help="Interior k-points per path segment (gp_nkp)",
    )
    path.add_argument(
        "--zero-weight-point",
        action="store_true",
        help="Append a zero-weight point (disables RSPt's TR consistency check)",
    )
    path.add_argument("-o", "--output", type=str, default="spts.band")
    path.set_defaults(func=path_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
